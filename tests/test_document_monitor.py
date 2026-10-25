import hashlib
import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backend.ai import OllamaUnavailableError
from backend.config import Settings
from backend.database import DocumentRepository
from backend.document_monitor import DocumentMonitor, DocumentWorker


ANALYSIS_RESPONSE = json.dumps(
    {
        "summary": "Prepare the launch.",
        "tasks": [{"description": "Prepare launch checklist"}],
        "events": [],
        "potential_blockers": [],
    }
)


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 10, 25, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class FakeClient:
    def __init__(self, fail_count: int = 0) -> None:
        self.fail_count = fail_count
        self.calls = 0

    def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise OllamaUnavailableError("local model is temporarily unavailable")
        return ANALYSIS_RESPONSE


def make_settings(
    database_path: Path,
    *,
    stability_seconds: float = 2,
    max_size_mb: int = 5,
    max_attempts: int = 3,
    recursive: bool = False,
) -> Settings:
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        ollama_timeout_seconds=5,
        max_document_size_mb=max_size_mb,
        database_path=database_path,
        document_poll_interval_seconds=0.03,
        document_stability_seconds=stability_seconds,
        document_recursive=recursive,
        document_max_attempts=max_attempts,
    )


def test_detects_new_file_and_skips_unchanged_content(tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    document = folder / "brief.txt"
    document.write_text("Prepare the launch.", encoding="utf-8")
    clock = MutableClock()
    repository = DocumentRepository(tmp_path / "assistant.db")
    client = FakeClient()
    monitor = DocumentMonitor(
        folder, repository, client, make_settings(tmp_path / "assistant.db"), clock=clock
    )

    assert monitor.run_once() == 0
    clock.advance(3)
    assert monitor.run_once() == 1
    assert monitor.run_once() == 0
    assert client.calls == 1
    assert repository.analysis_count() == 1
    item = repository.list_processing_files(monitor.directory_id)[0]
    assert item["status"] == "succeeded"
    assert item["file_hash"] == hashlib.sha256(document.read_bytes()).hexdigest()


def test_changed_file_gets_a_new_analysis_without_deleting_history(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    document = folder / "brief.txt"
    document.write_text("First version", encoding="utf-8")
    clock = MutableClock()
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    monitor = DocumentMonitor(
        folder, repository, FakeClient(), make_settings(database_path), clock=clock
    )

    monitor.run_once()
    clock.advance(3)
    assert monitor.run_once() == 1
    original_mtime = document.stat().st_mtime_ns
    document.write_text("Changed version", encoding="utf-8")
    os.utime(document, ns=(document.stat().st_atime_ns, original_mtime + 2_000_000))
    clock.advance(1)
    assert monitor.run_once() == 0
    clock.advance(3)
    assert monitor.run_once() == 1

    assert repository.analysis_count() == 2
    assert [item["source_text"] for item in (
        repository.get_analysis(row["analysis_id"])
        for row in repository.list_processing_files(monitor.directory_id)
    )] == ["Changed version"]


def test_same_content_with_changed_mtime_is_not_reprocessed(tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    document = folder / "brief.txt"
    document.write_text("Unchanged bytes", encoding="utf-8")
    clock = MutableClock()
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    client = FakeClient()
    monitor = DocumentMonitor(
        folder, repository, client, make_settings(database_path), clock=clock
    )
    monitor.run_once()
    clock.advance(3)
    assert monitor.run_once() == 1
    original_mtime = document.stat().st_mtime_ns
    os.utime(document, ns=(document.stat().st_atime_ns, original_mtime + 5_000_000))
    clock.advance(1)
    assert monitor.run_once() == 0
    clock.advance(3)

    assert monitor.run_once() == 0
    assert client.calls == 1
    assert repository.analysis_count() == 1


def test_oversized_file_is_recorded_as_permanent_failure(tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    (folder / "large.txt").write_bytes(b"x" * (1024 * 1024 + 1))
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    monitor = DocumentMonitor(
        folder,
        repository,
        FakeClient(),
        make_settings(database_path, stability_seconds=0, max_size_mb=1),
    )

    monitor.run_once()
    monitor.run_once()

    record = repository.list_processing_files(monitor.directory_id)[0]
    assert record["status"] == "failed"
    assert record["error_category"] == "DocumentError"
    assert record["retry_count"] == 1
    assert repository.list_processing_runs(
        monitor.directory_id, failed_only=True
    )[0]["error_category"] == "DocumentError"


def test_changed_hash_retries_a_previously_permanent_validation_failure(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    document = folder / "brief.txt"
    document.write_bytes(b"\xff")
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    monitor = DocumentMonitor(
        folder,
        repository,
        FakeClient(),
        make_settings(database_path, stability_seconds=0),
    )

    monitor.run_once()
    record = repository.list_processing_files(monitor.directory_id)[0]
    assert record["status"] == "failed"
    assert record["error_category"] == "DocumentError"

    document.write_text("corrected", encoding="utf-8")
    monitor.run_once()
    assert repository.list_processing_files(monitor.directory_id)[0]["status"] == "succeeded"
    assert repository.analysis_count() == 1


def test_unreadable_file_is_recorded_without_stopping_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    unreadable = folder / "unreadable.txt"
    readable = folder / "readable.txt"
    unreadable.write_text("private content", encoding="utf-8")
    readable.write_text("valid content", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path == unreadable:
            raise PermissionError("test permission failure")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    monitor = DocumentMonitor(
        folder,
        repository,
        FakeClient(),
        make_settings(database_path, stability_seconds=0),
    )

    assert monitor.run_once() == 2
    records = {
        item["relative_path"]: item
        for item in repository.list_processing_files(monitor.directory_id)
    }
    assert records["unreadable.txt"]["status"] == "failed"
    assert records["readable.txt"]["status"] == "succeeded"


def test_transient_model_failure_is_retried_after_backoff(tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    (folder / "brief.txt").write_text("source", encoding="utf-8")
    clock = MutableClock()
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    client = FakeClient(fail_count=1)
    monitor = DocumentMonitor(
        folder,
        repository,
        client,
        make_settings(database_path, stability_seconds=0, max_attempts=3),
        clock=clock,
    )

    assert monitor.run_once() == 0
    assert client.calls == 1
    assert repository.list_processing_files(monitor.directory_id)[0]["status"] == "failed"
    clock.advance(3)
    assert monitor.run_once() == 1
    assert client.calls == 2
    assert repository.analysis_count() == 1
    assert [run["outcome"] for run in repository.list_processing_runs(
        monitor.directory_id
    )] == ["succeeded", "failed"]


def test_transient_retries_are_bounded(tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    (folder / "brief.txt").write_text("source", encoding="utf-8")
    clock = MutableClock()
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    client = FakeClient(fail_count=99)
    monitor = DocumentMonitor(
        folder,
        repository,
        client,
        make_settings(database_path, stability_seconds=0, max_attempts=2),
        clock=clock,
    )

    monitor.run_once()
    clock.advance(3)
    monitor.run_once()
    clock.advance(3)
    monitor.run_once()

    record = repository.list_processing_files(monitor.directory_id)[0]
    assert client.calls == 2
    assert record["retry_count"] == 2
    assert record["status"] == "failed"
    assert repository.count_processing_runs(monitor.directory_id) == 2


def test_only_scans_subdirectories_when_configured(tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    (folder / "top.txt").write_text("top", encoding="utf-8")
    (nested / "nested.txt").write_text("nested", encoding="utf-8")
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    client = FakeClient()
    monitor = DocumentMonitor(
        folder,
        repository,
        client,
        make_settings(database_path, stability_seconds=0),
    )
    monitor.run_once()
    monitor.run_once()
    assert client.calls == 1
    assert repository.list_processing_files(monitor.directory_id)[0]["relative_path"] == "top.txt"

    recursive_monitor = DocumentMonitor(
        folder,
        repository,
        FakeClient(),
        make_settings(database_path, stability_seconds=0, recursive=True),
        recursive=True,
    )
    recursive_monitor.run_once()
    recursive_monitor.run_once()
    assert len(repository.list_processing_files(recursive_monitor.directory_id)) == 2


def test_database_claim_prevents_duplicate_concurrent_processing(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    (folder / "brief.txt").write_text("source", encoding="utf-8")
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    entered_model = threading.Event()
    release_model = threading.Event()

    class BlockingClient(FakeClient):
        def generate(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            entered_model.set()
            assert release_model.wait(5)
            return ANALYSIS_RESPONSE

    client = BlockingClient()
    settings = make_settings(database_path, stability_seconds=0)
    first = DocumentMonitor(folder, repository, client, settings)
    second = DocumentMonitor(folder, repository, client, settings)
    thread = threading.Thread(target=first.run_once)
    thread.start()
    assert entered_model.wait(5)
    second.run_once()
    release_model.set()
    thread.join(5)

    assert not thread.is_alive()
    assert client.calls == 1
    assert repository.analysis_count() == 1


def test_worker_starts_stops_and_rejects_duplicate_start(tmp_path: Path) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    settings = make_settings(database_path, stability_seconds=0)
    monitor = DocumentMonitor(folder, repository, FakeClient(), settings)
    worker = DocumentWorker(monitor, repository, 0.03)
    other_worker = DocumentWorker(
        DocumentMonitor(folder, repository, FakeClient(), settings),
        repository,
        0.03,
    )

    thread = worker.start()
    with pytest.raises(RuntimeError, match="already running"):
        worker.start()
    with pytest.raises(RuntimeError, match="directory is already running"):
        other_worker.run_forever()
    worker.stop(timeout=3)

    assert not thread.is_alive()
    assert repository.get_worker_status(monitor.directory_id)["status"] == "stopped"


def test_changed_file_during_model_call_is_not_saved_as_stale(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "inbox"
    folder.mkdir()
    document = folder / "brief.txt"
    document.write_text("original", encoding="utf-8")
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)

    class ChangingClient(FakeClient):
        def generate(self, prompt: str, **kwargs: object) -> str:
            self.calls += 1
            document.write_text("changed while analyzing", encoding="utf-8")
            return ANALYSIS_RESPONSE

    monitor = DocumentMonitor(
        folder,
        repository,
        ChangingClient(),
        make_settings(database_path, stability_seconds=0),
    )

    monitor.run_once()
    monitor.run_once()

    assert repository.analysis_count() == 0
    assert repository.list_processing_runs(monitor.directory_id)[0]["error_category"] == (
        "FileChangedDuringAnalysis"
    )
