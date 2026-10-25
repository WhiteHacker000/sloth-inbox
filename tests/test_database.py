import sqlite3
from pathlib import Path

import pytest

from backend.database import (
    AnalysisNotFoundError,
    DatabaseError,
    DocumentRepository,
    TaskNotFoundError,
)
from backend.schemas import DocumentAnalysis


def sample_analysis() -> DocumentAnalysis:
    return DocumentAnalysis.model_validate(
        {
            "summary": "The report is due tomorrow.",
            "tasks": [
                {
                    "description": "Submit report",
                    "deadline_text": "tomorrow",
                    "priority": "high",
                    "evidence": "Submit report by tomorrow.",
                }
            ],
            "events": [
                {"description": "Project review", "date_text": "March 2"}
            ],
            "potential_blockers": ["Testing is not complete."],
        }
    )


def test_persists_analysis_tasks_and_source(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "assistant.db"
    repository = DocumentRepository(database_path)
    analysis_id = repository.save_analysis(
        "Submit report by tomorrow.",
        sample_analysis(),
        "Weekly report",
    )

    saved = DocumentRepository(database_path).get_analysis(analysis_id)
    assert saved["title"] == "Weekly report"
    assert saved["source_text"] == "Submit report by tomorrow."
    assert saved["analysis"].summary == "The report is due tomorrow."

    tasks = repository.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["description"] == "Submit report"
    assert tasks[0]["source_document_id"] == analysis_id
    assert len(repository.list_tasks_for_analysis(analysis_id)) == 1
    assert repository.list_analyses()[0]["task_count"] == 1


def test_task_completion_reopening_and_filtering(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "assistant.db")
    repository.save_analysis("text", sample_analysis())
    task_id = repository.list_tasks()[0]["id"]

    repository.set_task_status(task_id, "completed")
    assert repository.list_tasks("completed")[0]["id"] == task_id
    assert repository.list_tasks("pending") == []
    assert repository.task_counts() == {
        "pending": 0,
        "in_progress": 0,
        "completed": 1,
    }

    repository.set_task_status(task_id, "pending")
    assert repository.list_tasks("pending")[0]["id"] == task_id


def test_missing_analysis_and_task_raise_clear_errors(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "assistant.db")
    with pytest.raises(AnalysisNotFoundError):
        repository.get_analysis("missing")
    with pytest.raises(TaskNotFoundError):
        repository.set_task_status("missing", "completed")


def test_repeated_analysis_saves_create_separate_records(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "assistant.db")
    first = repository.save_analysis("text", sample_analysis())
    second = repository.save_analysis("text", sample_analysis())

    assert first != second
    assert len(repository.list_analyses()) == 2
    assert repository.analysis_count() == 2


def test_analysis_count_is_not_capped_by_history_limit(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "assistant.db")
    for _ in range(101):
        repository.save_analysis("text", DocumentAnalysis(summary="Summary"))

    assert len(repository.list_analyses()) == 100
    assert repository.analysis_count() == 101


def test_failed_task_insert_rolls_back_analysis_transaction(tmp_path: Path) -> None:
    database_path = tmp_path / "assistant.db"
    repository = DocumentRepository(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_task_insert
            BEFORE INSERT ON tasks
            BEGIN
                SELECT RAISE(ABORT, 'test failure');
            END;
            """
        )

    with pytest.raises(DatabaseError, match="database operation failed"):
        repository.save_analysis("source", sample_analysis())

    assert repository.list_analyses() == []
    assert repository.list_tasks() == []


def test_initialization_adds_processing_tables_without_replacing_existing_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "assistant.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE analyses (
                id TEXT PRIMARY KEY,
                title TEXT,
                source_text TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                source_document_id TEXT NOT NULL REFERENCES analyses(id),
                description TEXT NOT NULL,
                owner TEXT,
                deadline_text TEXT,
                priority TEXT,
                status TEXT NOT NULL,
                evidence TEXT,
                extracted_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO analyses VALUES (
                'existing', 'Old record', 'source',
                '{"summary":"Saved before Phase 2","tasks":[],"events":[],"potential_blockers":[]}',
                '2026-10-20T10:00:00+00:00'
            );
            """
        )

    repository = DocumentRepository(database_path)

    assert repository.get_analysis("existing")["analysis"].summary == (
        "Saved before Phase 2"
    )
    assert repository.processing_file_counts("some-directory")["pending"] == 0
