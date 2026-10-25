import hashlib
import logging
import os
import signal
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from backend.ai import OllamaError, OllamaClient, OllamaModelNotFoundError
from backend.analyzer import AnalysisOutputError, TextGenerator, analyze_document
from backend.config import Settings
from backend.database import DatabaseError, DocumentRepository
from backend.document_service import DocumentError, read_text_bytes

logger = logging.getLogger(__name__)
_ACTIVE_WORKERS: set[str] = set()
_ACTIVE_WORKERS_LOCK = threading.Lock()


def directory_identifier(directory: str | Path) -> str:
    resolved = Path(directory).expanduser().resolve()
    return hashlib.sha256(os.fsencode(str(resolved))).hexdigest()


class MonitorConfigurationError(ValueError):
    """Raised when the explicit input directory is not usable."""


class DocumentMonitor:
    def __init__(
        self,
        input_directory: str | Path,
        repository: DocumentRepository,
        client: TextGenerator,
        settings: Settings,
        *,
        recursive: bool | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        root = Path(input_directory).expanduser()
        if not root.exists() or not root.is_dir():
            raise MonitorConfigurationError(
                "The configured document input directory must already exist."
            )
        self.root = root.resolve()
        self.directory_id = directory_identifier(self.root)
        self.repository = repository
        self.client = client
        self.settings = settings
        self.recursive = settings.document_recursive if recursive is None else recursive
        self.clock = clock or (lambda: datetime.now(UTC))
        self.analyze = lambda text, title: analyze_document(text, client, title)
        self._file_locks: dict[str, threading.Lock] = {}
        self._file_locks_guard = threading.Lock()

    def _candidates(self) -> list[Path]:
        try:
            paths = self.root.rglob("*") if self.recursive else self.root.iterdir()
            candidates = []
            for path in paths:
                if path.is_symlink() or path.suffix.lower() != ".txt":
                    continue
                try:
                    if path.is_file() and path.resolve().is_relative_to(self.root):
                        candidates.append(path)
                except OSError as exc:
                    logger.warning(
                        "Could not inspect document path=%s category=%s",
                        path.relative_to(self.root),
                        type(exc).__name__,
                    )
            return sorted(candidates, key=lambda item: item.as_posix())
        except OSError as exc:
            logger.error("Could not list watched directory category=%s", type(exc).__name__)
            raise MonitorConfigurationError(
                "The configured document input directory could not be read."
            ) from exc

    def run_once(self) -> int:
        processed = 0
        now = self.clock()
        present: set[str] = set()
        for path in self._candidates():
            relative_path = path.relative_to(self.root).as_posix()
            present.add(relative_path)
            try:
                before = path.stat()
                record = self.repository.observe_file(
                    self.directory_id,
                    relative_path,
                    before.st_size,
                    before.st_mtime_ns,
                    now,
                )
                stable_since = datetime.fromisoformat(str(record["stable_since"]))
                if (now - stable_since).total_seconds() < self.settings.document_stability_seconds:
                    continue
                if before.st_size > self.settings.max_document_size_bytes:
                    if self._process_unreadable_candidate(
                        record,
                        relative_path,
                        None,
                        DocumentError("The document exceeds the configured size limit."),
                    ):
                        processed += 1
                    continue

                try:
                    content: bytes | None = None
                    content = path.read_bytes()
                    after = path.stat()
                    if (
                        before.st_size != after.st_size
                        or before.st_mtime_ns != after.st_mtime_ns
                        or len(content) != after.st_size
                    ):
                        self.repository.observe_file(
                            self.directory_id,
                            relative_path,
                            after.st_size,
                            after.st_mtime_ns,
                            now,
                        )
                        logger.info("Deferred changing document path=%s", relative_path)
                        continue
                    digest = hashlib.sha256(content).hexdigest()
                    text = read_text_bytes(content, self.settings.max_document_size_bytes)
                except (OSError, DocumentError) as exc:
                    digest = hashlib.sha256(content).hexdigest() if content is not None else None
                    if self._process_unreadable_candidate(
                        record, relative_path, digest, exc
                    ):
                        processed += 1
                    continue

                if (
                    record["analysis_id"] is not None
                    and record["file_hash"] == digest
                    and record["status"] != "processing"
                ):
                    self.repository.mark_file_unchanged(record["id"], now)
                    continue
                if self._process_candidate(record, relative_path, digest, text):
                    processed += 1
            except DatabaseError as exc:
                logger.error(
                    "Document processing state update failed path=%s category=%s",
                    relative_path,
                    type(exc).__name__,
                )
            except OSError as exc:
                logger.warning(
                    "Could not access document path=%s category=%s",
                    relative_path,
                    type(exc).__name__,
                )

        for record in self.repository.list_processing_files(self.directory_id, limit=10000):
            relative_path = str(record["relative_path"])
            if relative_path not in present and record["status"] != "missing":
                self.repository.mark_file_missing(self.directory_id, relative_path)
        return processed

    def _process_unreadable_candidate(
        self,
        record: dict[str, object],
        relative_path: str,
        digest: str | None,
        error: Exception,
    ) -> bool:
        started_at = self.clock()
        lock = self._file_lock(relative_path)
        if not lock.acquire(blocking=False):
            return False
        try:
            claimed = self.repository.claim_file(
                self.directory_id,
                relative_path,
                digest,
                self.settings.document_max_attempts,
                self.settings.ollama_timeout_seconds + 60,
                started_at,
            )
            if claimed is None:
                return False
            self.repository.fail_file_processing(
                str(claimed["id"]),
                started_at,
                self.clock(),
                type(error).__name__,
                retryable=False,
                max_attempts=self.settings.document_max_attempts,
            )
            logger.warning(
                "Document rejected path=%s category=%s",
                relative_path,
                type(error).__name__,
            )
            return True
        finally:
            lock.release()

    def _process_candidate(
        self,
        record: dict[str, object],
        relative_path: str,
        digest: str,
        text: str,
    ) -> bool:
        lock = self._file_lock(relative_path)
        if not lock.acquire(blocking=False):
            return False
        try:
            started_at = self.clock()
            claimed = self.repository.claim_file(
                self.directory_id,
                relative_path,
                digest,
                self.settings.document_max_attempts,
                self.settings.ollama_timeout_seconds + 60,
                started_at,
            )
            if claimed is None:
                return False
            try:
                analysis = self.analyze(text, Path(relative_path).name)
                current_path = self.root / relative_path
                current_stat = current_path.stat()
                current_digest = (
                    hashlib.sha256(current_path.read_bytes()).hexdigest()
                    if current_stat.st_size <= self.settings.max_document_size_bytes
                    else None
                )
                if (
                    current_stat.st_size != record["file_size"]
                    or current_stat.st_mtime_ns != record["file_mtime_ns"]
                    or current_digest != digest
                ):
                    self.repository.fail_file_processing(
                        str(claimed["id"]),
                        started_at,
                        self.clock(),
                        "FileChangedDuringAnalysis",
                        retryable=True,
                        max_attempts=self.settings.document_max_attempts,
                    )
                    logger.info("Deferred changed document path=%s", relative_path)
                    return False
                self.repository.complete_file_processing(
                    str(claimed["id"]),
                    text,
                    analysis,
                    started_at,
                    self.clock(),
                )
            except OllamaModelNotFoundError as exc:
                self.repository.fail_file_processing(
                    str(claimed["id"]),
                    started_at,
                    self.clock(),
                    type(exc).__name__,
                    retryable=False,
                    max_attempts=self.settings.document_max_attempts,
                )
                logger.warning(
                    "Document processing failed path=%s category=%s",
                    relative_path,
                    type(exc).__name__,
                )
                return False
            except (OllamaError, AnalysisOutputError) as exc:
                self.repository.fail_file_processing(
                    str(claimed["id"]),
                    started_at,
                    self.clock(),
                    type(exc).__name__,
                    retryable=True,
                    max_attempts=self.settings.document_max_attempts,
                )
                logger.warning(
                    "Document processing failed path=%s category=%s",
                    relative_path,
                    type(exc).__name__,
                )
                return False
            except Exception as exc:
                self.repository.fail_file_processing(
                    str(claimed["id"]),
                    started_at,
                    self.clock(),
                    type(exc).__name__,
                    retryable=False,
                    max_attempts=self.settings.document_max_attempts,
                )
                logger.error(
                    "Unexpected document processing failure path=%s category=%s",
                    relative_path,
                    type(exc).__name__,
                )
                return False
            logger.info("Document processed path=%s", relative_path)
            return True
        finally:
            lock.release()

    def _file_lock(self, relative_path: str) -> threading.Lock:
        with self._file_locks_guard:
            return self._file_locks.setdefault(relative_path, threading.Lock())


class DocumentWorker:
    def __init__(
        self,
        monitor: DocumentMonitor,
        repository: DocumentRepository,
        poll_interval_seconds: float,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("Polling interval must be positive.")
        self.monitor = monitor
        self.repository = repository
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()

    def start(self) -> threading.Thread:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("The document worker is already running.")
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self.run_forever,
                name="sloth-document-worker",
                daemon=False,
            )
            self._thread.start()
            return self._thread

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def run_forever(self) -> None:
        with _ACTIVE_WORKERS_LOCK:
            if self.monitor.directory_id in _ACTIVE_WORKERS:
                raise RuntimeError("A worker for this document directory is already running.")
            _ACTIVE_WORKERS.add(self.monitor.directory_id)
        try:
            self.repository.set_worker_status(
                self.monitor.directory_id,
                str(self.monitor.root),
                os.getpid(),
                "running",
                self.monitor.clock(),
            )
            while not self._stop_event.is_set():
                try:
                    self.monitor.run_once()
                    self.repository.set_worker_status(
                        self.monitor.directory_id,
                        str(self.monitor.root),
                        os.getpid(),
                        "running",
                        self.monitor.clock(),
                    )
                except Exception as exc:
                    logger.exception(
                        "Document worker poll failed category=%s", type(exc).__name__
                    )
                self._stop_event.wait(self.poll_interval_seconds)
        finally:
            try:
                self.repository.set_worker_status(
                    self.monitor.directory_id,
                    str(self.monitor.root),
                    os.getpid(),
                    "stopped",
                    self.monitor.clock(),
                )
            finally:
                with _ACTIVE_WORKERS_LOCK:
                    _ACTIVE_WORKERS.discard(self.monitor.directory_id)


def resolve_input_directory(
    settings: Settings,
    repository: DocumentRepository,
) -> Path:
    configured = repository.get_setting("document_input_dir")
    directory = Path(configured).expanduser() if configured else settings.document_input_dir
    if directory is None:
        raise MonitorConfigurationError(
            "Configure DOCUMENT_INPUT_DIR or choose a folder in the Sloth AI dashboard."
        )
    return directory


def run_worker() -> None:
    settings = Settings.from_env()
    repository = DocumentRepository(settings.database_path)
    input_directory = resolve_input_directory(settings, repository)
    monitor = DocumentMonitor(
        input_directory,
        repository,
        OllamaClient(settings),
        settings,
    )
    worker = DocumentWorker(
        monitor,
        repository,
        settings.document_poll_interval_seconds,
    )
    stop_event = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        logger.info("Received shutdown signal=%s", signum)
        stop_event.set()
        worker.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    logger.info(
        "Starting document worker directory=%s interval_seconds=%s recursive=%s",
        input_directory,
        settings.document_poll_interval_seconds,
        settings.document_recursive,
    )
    worker.run_forever()


if __name__ == "__main__":
    run_worker()
