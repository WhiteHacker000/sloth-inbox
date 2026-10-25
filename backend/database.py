import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from backend.schemas import DocumentAnalysis, TaskStatus

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Raised when the local SQLite repository cannot complete an operation."""


class AnalysisNotFoundError(LookupError):
    """Raised when an analysis identifier does not exist."""


class TaskNotFoundError(LookupError):
    """Raised when a task identifier does not exist."""


class DocumentRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser()
        self.initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=10)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
        except (OSError, sqlite3.Error) as exc:
            raise DatabaseError("Could not open the local SQLite database.") from exc
        try:
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise DatabaseError("A local database operation failed.") from exc
        finally:
            connection.close()

    def initialize(self) -> None:
        try:
            with self._connection() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS analyses (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        source_text TEXT NOT NULL,
                        analysis_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        source_document_id TEXT NOT NULL
                            REFERENCES analyses(id) ON DELETE CASCADE,
                        description TEXT NOT NULL,
                        owner TEXT,
                        deadline_text TEXT,
                        priority TEXT,
                        status TEXT NOT NULL
                            CHECK (status IN ('pending', 'in_progress', 'completed')),
                        evidence TEXT,
                        extracted_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_tasks_status
                        ON tasks(status);
                    CREATE INDEX IF NOT EXISTS idx_tasks_source_document
                        ON tasks(source_document_id);
                    CREATE INDEX IF NOT EXISTS idx_analyses_created_at
                        ON analyses(created_at DESC);
                    CREATE TABLE IF NOT EXISTS watched_files (
                        id TEXT PRIMARY KEY,
                        directory_id TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        file_hash TEXT,
                        file_size INTEGER NOT NULL,
                        file_mtime_ns INTEGER NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'pending', 'processing', 'succeeded', 'failed', 'missing'
                            )
                        ),
                        stable_since TEXT NOT NULL,
                        last_processed_at TEXT,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        next_retry_at TEXT,
                        error_category TEXT,
                        analysis_id TEXT REFERENCES analyses(id),
                        lease_expires_at TEXT,
                        updated_at TEXT NOT NULL,
                        UNIQUE (directory_id, relative_path)
                    );
                    CREATE INDEX IF NOT EXISTS idx_watched_files_status
                        ON watched_files(status, next_retry_at);
                    CREATE TABLE IF NOT EXISTS processing_runs (
                        id TEXT PRIMARY KEY,
                        source_file_id TEXT REFERENCES watched_files(id),
                        directory_id TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        file_hash TEXT,
                        started_at TEXT NOT NULL,
                        finished_at TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        error_category TEXT,
                        analysis_id TEXT REFERENCES analyses(id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_processing_runs_started
                        ON processing_runs(started_at DESC);
                    CREATE TABLE IF NOT EXISTS automation_workers (
                        directory_id TEXT PRIMARY KEY,
                        directory_path TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('running', 'stopped')),
                        started_at TEXT NOT NULL,
                        heartbeat_at TEXT NOT NULL,
                        stopped_at TEXT,
                        last_error_category TEXT
                    );
                    CREATE TABLE IF NOT EXISTS app_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
        except DatabaseError:
            raise

    def save_analysis(
        self,
        source_text: str,
        analysis: DocumentAnalysis,
        title: str | None = None,
    ) -> str:
        analysis_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        document_title = title or analysis.source_title
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO analyses (id, title, source_text, analysis_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_id,
                        document_title,
                        source_text,
                        analysis.model_dump_json(),
                        now,
                    ),
                )
                for task in analysis.tasks:
                    connection.execute(
                        """
                        INSERT INTO tasks (
                            id, source_document_id, description, owner, deadline_text,
                            priority, status, evidence, extracted_json, created_at,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            analysis_id,
                            task.description,
                            task.owner,
                            task.deadline_text,
                            task.priority,
                            task.status,
                            task.evidence,
                            task.model_dump_json(),
                            now,
                            now,
                        ),
                    )
        except DatabaseError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseError("Could not save the document analysis.") from exc
        logger.info("Saved document analysis id=%s", analysis_id)
        return analysis_id

    def list_analyses(self, limit: int = 100) -> list[dict[str, object]]:
        if limit <= 0:
            raise ValueError("Analysis limit must be a positive integer.")
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT a.id, a.title, a.created_at,
                           json_extract(a.analysis_json, '$.summary') AS summary,
                           (SELECT COUNT(*) FROM tasks t
                            WHERE t.source_document_id = a.id) AS task_count
                    FROM analyses a
                    ORDER BY a.created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not list saved analyses.") from exc
        return [dict(row) for row in rows]

    def analysis_count(self) -> int:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM analyses"
                ).fetchone()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not count saved analyses.") from exc
        return row["total"]

    def get_analysis(self, analysis_id: str) -> dict[str, object]:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM analyses WHERE id = ?",
                    (analysis_id,),
                ).fetchone()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not retrieve the saved analysis.") from exc
        if row is None:
            raise AnalysisNotFoundError("The requested analysis was not found.")
        try:
            analysis = DocumentAnalysis.model_validate_json(row["analysis_json"])
        except (ValidationError, ValueError) as exc:
            raise DatabaseError("The saved analysis data is invalid.") from exc
        return {
            "id": row["id"],
            "title": row["title"],
            "source_text": row["source_text"],
            "analysis": analysis,
            "created_at": row["created_at"],
        }

    def list_tasks(
        self,
        status: TaskStatus | None = None,
    ) -> list[dict[str, object]]:
        if status is not None and status not in {
            "pending",
            "in_progress",
            "completed",
        }:
            raise ValueError("Invalid task status.")
        query = """
            SELECT t.*, a.title AS source_title
            FROM tasks t
            JOIN analyses a ON a.id = t.source_document_id
        """
        parameters: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE t.status = ?"
            parameters = (status,)
        query += " ORDER BY t.created_at DESC, t.id"
        try:
            with self._connection() as connection:
                rows = connection.execute(query, parameters).fetchall()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not list tasks.") from exc
        return [dict(row) for row in rows]

    def list_tasks_for_analysis(self, analysis_id: str) -> list[dict[str, object]]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM tasks
                    WHERE source_document_id = ?
                    ORDER BY created_at, id
                    """,
                    (analysis_id,),
                ).fetchall()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not retrieve tasks for the analysis.") from exc
        return [dict(row) for row in rows]

    def set_task_status(self, task_id: str, status: TaskStatus) -> None:
        if status not in {"pending", "in_progress", "completed"}:
            raise ValueError("Invalid task status.")
        now = datetime.now(UTC).isoformat()
        try:
            with self._connection() as connection:
                result = connection.execute(
                    """
                    UPDATE tasks SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, now, task_id),
                )
                if result.rowcount == 0:
                    raise TaskNotFoundError("The requested task was not found.")
        except TaskNotFoundError:
            raise
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not update the task.") from exc

    def task_counts(self) -> dict[str, int]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT status, COUNT(*) AS total FROM tasks GROUP BY status"
                ).fetchall()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not count tasks.") from exc
        counts = {"pending": 0, "in_progress": 0, "completed": 0}
        counts.update({row["status"]: row["total"] for row in rows})
        return counts

    def get_setting(self, key: str) -> str | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT value FROM app_settings WHERE key = ?",
                    (key,),
                ).fetchone()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not read application settings.") from exc
        return row["value"] if row is not None else None

    def set_setting(self, key: str, value: str) -> None:
        now = datetime.now(UTC).isoformat()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (key, value, now),
                )
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not save application settings.") from exc

    def delete_setting(self, key: str) -> None:
        try:
            with self._connection() as connection:
                connection.execute("DELETE FROM app_settings WHERE key = ?", (key,))
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not reset application settings.") from exc

    def observe_file(
        self,
        directory_id: str,
        relative_path: str,
        file_size: int,
        file_mtime_ns: int,
        now: datetime,
    ) -> dict[str, object]:
        timestamp = now.isoformat()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO watched_files (
                        id, directory_id, relative_path, file_size, file_mtime_ns,
                        status, stable_since, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                    ON CONFLICT(directory_id, relative_path) DO UPDATE SET
                        file_size = excluded.file_size,
                        file_mtime_ns = excluded.file_mtime_ns,
                        file_hash = watched_files.file_hash,
                        status = CASE
                            WHEN watched_files.file_size != excluded.file_size
                              OR watched_files.file_mtime_ns != excluded.file_mtime_ns
                            THEN CASE WHEN watched_files.status = 'processing'
                                THEN 'processing' ELSE 'pending' END
                            WHEN watched_files.status = 'missing'
                            THEN 'pending'
                            ELSE watched_files.status END,
                        stable_since = CASE
                            WHEN watched_files.file_size != excluded.file_size
                              OR watched_files.file_mtime_ns != excluded.file_mtime_ns
                            THEN excluded.stable_since
                            ELSE watched_files.stable_since END,
                        retry_count = CASE
                            WHEN watched_files.file_size != excluded.file_size
                              OR watched_files.file_mtime_ns != excluded.file_mtime_ns
                            THEN 0 ELSE watched_files.retry_count END,
                        next_retry_at = CASE
                            WHEN watched_files.file_size != excluded.file_size
                              OR watched_files.file_mtime_ns != excluded.file_mtime_ns
                            THEN NULL ELSE watched_files.next_retry_at END,
                        error_category = CASE
                            WHEN watched_files.file_size != excluded.file_size
                              OR watched_files.file_mtime_ns != excluded.file_mtime_ns
                            THEN NULL ELSE watched_files.error_category END,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(uuid.uuid4()),
                        directory_id,
                        relative_path,
                        file_size,
                        file_mtime_ns,
                        timestamp,
                        timestamp,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT * FROM watched_files
                    WHERE directory_id = ? AND relative_path = ?
                    """,
                    (directory_id, relative_path),
                ).fetchone()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not record the observed document.") from exc
        return dict(row)

    def mark_file_missing(self, directory_id: str, relative_path: str) -> None:
        now = datetime.now(UTC).isoformat()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    UPDATE watched_files SET status = 'missing', updated_at = ?
                    WHERE directory_id = ? AND relative_path = ?
                      AND status != 'processing'
                    """,
                    (now, directory_id, relative_path),
                )
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not mark the document as missing.") from exc

    def mark_file_unchanged(self, source_file_id: str, now: datetime) -> None:
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    UPDATE watched_files SET status = 'succeeded', updated_at = ?
                    WHERE id = ? AND analysis_id IS NOT NULL
                    """,
                    (now.isoformat(), source_file_id),
                )
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not update unchanged document state.") from exc

    def claim_file(
        self,
        directory_id: str,
        relative_path: str,
        file_hash: str | None,
        max_attempts: int,
        lease_seconds: float,
        now: datetime,
    ) -> dict[str, object] | None:
        timestamp = now.isoformat()
        lease_until = datetime.fromtimestamp(
            now.timestamp() + lease_seconds, UTC
        ).isoformat()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT * FROM watched_files
                    WHERE directory_id = ? AND relative_path = ?
                    """,
                    (directory_id, relative_path),
                ).fetchone()
                if row is None:
                    return None
                if row["status"] == "succeeded" and row["file_hash"] == file_hash:
                    return None
                if row["status"] == "processing" and row["lease_expires_at"]:
                    if row["lease_expires_at"] > timestamp:
                        return None
                if (
                    row["status"] != "processing"
                    and file_hash is not None
                    and row["file_hash"] != file_hash
                ):
                    connection.execute(
                        """
                        UPDATE watched_files SET status = 'pending',
                            retry_count = 0, next_retry_at = NULL,
                            error_category = NULL, file_hash = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (file_hash, timestamp, row["id"]),
                    )
                    row = connection.execute(
                        "SELECT * FROM watched_files WHERE id = ?",
                        (row["id"],),
                    ).fetchone()
                if row["status"] == "failed":
                    if row["retry_count"] >= max_attempts or row["next_retry_at"] is None:
                        return None
                if row["next_retry_at"] and row["next_retry_at"] > timestamp:
                    return None
                if row["retry_count"] >= max_attempts and row["status"] == "processing":
                    connection.execute(
                        """
                        UPDATE watched_files SET status = 'failed',
                            error_category = 'WorkerLeaseExpired',
                            next_retry_at = NULL, lease_expires_at = NULL,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (timestamp, row["id"]),
                    )
                    connection.execute(
                        """
                        INSERT INTO processing_runs (
                            id, source_file_id, directory_id, relative_path, file_hash,
                            started_at, finished_at, outcome, error_category
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'failed', 'WorkerLeaseExpired')
                        """,
                        (
                            str(uuid.uuid4()),
                            row["id"],
                            row["directory_id"],
                            row["relative_path"],
                            row["file_hash"],
                            row["updated_at"],
                            timestamp,
                        ),
                    )
                    return None
                connection.execute(
                    """
                    UPDATE watched_files SET
                        file_hash = ?, status = 'processing',
                        retry_count = retry_count + 1,
                        error_category = NULL, next_retry_at = NULL,
                        lease_expires_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (file_hash, lease_until, timestamp, row["id"]),
                )
                return {
                    **dict(row),
                    "file_hash": file_hash,
                    "status": "processing",
                    "retry_count": row["retry_count"] + 1,
                    "lease_expires_at": lease_until,
                }
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not claim the document for processing.") from exc

    def complete_file_processing(
        self,
        source_file_id: str,
        source_text: str,
        analysis: DocumentAnalysis,
        started_at: datetime,
        finished_at: datetime,
    ) -> str:
        analysis_id = str(uuid.uuid4())
        now = finished_at.isoformat()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                file_row = connection.execute(
                    "SELECT * FROM watched_files WHERE id = ?",
                    (source_file_id,),
                ).fetchone()
                if file_row is None or file_row["status"] != "processing":
                    raise DatabaseError("The document is no longer claimed for processing.")
                connection.execute(
                    """
                    INSERT INTO analyses (id, title, source_text, analysis_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_id,
                        Path(file_row["relative_path"]).name,
                        source_text,
                        analysis.model_dump_json(),
                        now,
                    ),
                )
                for task in analysis.tasks:
                    connection.execute(
                        """
                        INSERT INTO tasks (
                            id, source_document_id, description, owner, deadline_text,
                            priority, status, evidence, extracted_json, created_at,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            analysis_id,
                            task.description,
                            task.owner,
                            task.deadline_text,
                            task.priority,
                            task.status,
                            task.evidence,
                            task.model_dump_json(),
                            now,
                            now,
                        ),
                    )
                connection.execute(
                    """
                    UPDATE watched_files SET
                        status = 'succeeded', analysis_id = ?,
                        last_processed_at = ?, error_category = NULL,
                        lease_expires_at = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (analysis_id, now, now, source_file_id),
                )
                connection.execute(
                    """
                    INSERT INTO processing_runs (
                        id, source_file_id, directory_id, relative_path, file_hash,
                        started_at, finished_at, outcome, analysis_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'succeeded', ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        source_file_id,
                        file_row["directory_id"],
                        file_row["relative_path"],
                        file_row["file_hash"],
                        started_at.isoformat(),
                        now,
                        analysis_id,
                    ),
                )
        except DatabaseError:
            raise
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseError("Could not persist document processing results.") from exc
        return analysis_id

    def fail_file_processing(
        self,
        source_file_id: str,
        started_at: datetime,
        finished_at: datetime,
        error_category: str,
        *,
        retryable: bool,
        max_attempts: int,
    ) -> None:
        now = finished_at.isoformat()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM watched_files WHERE id = ?",
                    (source_file_id,),
                ).fetchone()
                if row is None:
                    raise DatabaseError("The processing record no longer exists.")
                retry_count = row["retry_count"]
                can_retry = retryable and retry_count < max_attempts
                next_retry = (
                    datetime.fromtimestamp(
                        finished_at.timestamp() + min(2 ** (retry_count - 1), 300),
                        UTC,
                    ).isoformat()
                    if can_retry
                    else None
                )
                connection.execute(
                    """
                    UPDATE watched_files SET
                        status = 'failed', error_category = ?,
                        next_retry_at = ?, lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (error_category, next_retry, now, source_file_id),
                )
                connection.execute(
                    """
                    INSERT INTO processing_runs (
                        id, source_file_id, directory_id, relative_path, file_hash,
                        started_at, finished_at, outcome, error_category
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'failed', ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        source_file_id,
                        row["directory_id"],
                        row["relative_path"],
                        row["file_hash"],
                        started_at.isoformat(),
                        now,
                        error_category,
                    ),
                )
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not persist the processing failure.") from exc

    def list_processing_files(
        self,
        directory_id: str | None = None,
        limit: int = 100,
        *,
        status: str | None = None,
    ) -> list[dict[str, object]]:
        if limit <= 0:
            raise ValueError("File limit must be a positive integer.")
        query = "SELECT * FROM watched_files"
        conditions: list[str] = []
        parameters: tuple[object, ...] = ()
        if directory_id is not None:
            conditions.append("directory_id = ?")
            parameters = (directory_id,)
        if status is not None:
            if status not in {"pending", "processing", "succeeded", "failed", "missing"}:
                raise ValueError("Invalid document processing status.")
            conditions.append("status = ?")
            parameters += (status,)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC LIMIT ?"
        parameters += (limit,)
        try:
            with self._connection() as connection:
                rows = connection.execute(query, parameters).fetchall()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not list processed documents.") from exc
        return [dict(row) for row in rows]

    def processing_file_counts(self, directory_id: str) -> dict[str, int]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT status, COUNT(*) AS total FROM watched_files
                    WHERE directory_id = ? GROUP BY status
                    """,
                    (directory_id,),
                ).fetchall()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not count document processing states.") from exc
        counts = {
            "pending": 0,
            "processing": 0,
            "succeeded": 0,
            "failed": 0,
            "missing": 0,
        }
        counts.update({row["status"]: row["total"] for row in rows})
        return counts

    def list_processing_runs(
        self,
        directory_id: str | None = None,
        limit: int = 100,
        *,
        failed_only: bool = False,
    ) -> list[dict[str, object]]:
        conditions: list[str] = []
        parameters: tuple[object, ...] = ()
        if directory_id is not None:
            conditions.append("directory_id = ?")
            parameters += (directory_id,)
        if failed_only:
            conditions.append("outcome = 'failed'")
        query = "SELECT * FROM processing_runs"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY started_at DESC LIMIT ?"
        parameters += (limit,)
        try:
            with self._connection() as connection:
                rows = connection.execute(query, parameters).fetchall()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not list document processing runs.") from exc
        return [dict(row) for row in rows]

    def count_processing_runs(self, directory_id: str) -> int:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS total FROM processing_runs
                    WHERE directory_id = ?
                    """,
                    (directory_id,),
                ).fetchone()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not count processing runs.") from exc
        return row["total"]

    def set_worker_status(
        self,
        directory_id: str,
        directory_path: str,
        pid: int,
        status: str,
        now: datetime,
    ) -> None:
        timestamp = now.isoformat()
        if status not in {"running", "stopped"}:
            raise ValueError("Invalid worker status.")
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO automation_workers (
                        directory_id, directory_path, pid, status, started_at,
                        heartbeat_at, stopped_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(directory_id) DO UPDATE SET
                        directory_path = excluded.directory_path,
                        pid = excluded.pid,
                        status = excluded.status,
                        started_at = CASE WHEN automation_workers.status != 'running'
                            AND excluded.status = 'running'
                            THEN excluded.started_at
                            ELSE automation_workers.started_at END,
                        heartbeat_at = excluded.heartbeat_at,
                        stopped_at = excluded.stopped_at
                    """,
                    (
                        directory_id,
                        directory_path,
                        pid,
                        status,
                        timestamp,
                        timestamp,
                        timestamp if status == "stopped" else None,
                    ),
                )
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not update the automation worker status.") from exc

    def get_worker_status(self, directory_id: str) -> dict[str, object] | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    """
                    SELECT * FROM automation_workers WHERE directory_id = ?
                    """,
                    (directory_id,),
                ).fetchone()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not read the automation worker status.") from exc
        return dict(row) if row is not None else None

    def get_briefing_data(
        self,
        since: datetime,
        today: datetime,
        limit: int = 20,
    ) -> dict[str, object]:
        try:
            with self._connection() as connection:
                counts_rows = connection.execute(
                    """
                    SELECT status, COUNT(*) AS total FROM tasks GROUP BY status
                    """
                ).fetchall()
                tasks = connection.execute(
                    """
                    SELECT t.*, a.title AS source_title
                    FROM tasks t
                    JOIN analyses a ON a.id = t.source_document_id
                    ORDER BY t.created_at DESC
                    """
                ).fetchall()
                analyses = connection.execute(
                    """
                    SELECT id, title, created_at, analysis_json
                    FROM analyses
                    WHERE created_at >= ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (since.isoformat(), limit),
                ).fetchall()
                totals = connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM analyses) AS analyses,
                        (SELECT COUNT(*) FROM tasks) AS tasks,
                        (SELECT COUNT(*) FROM processing_runs
                         WHERE outcome = 'succeeded') AS successful_runs,
                        (SELECT COUNT(*) FROM processing_runs
                         WHERE outcome = 'failed') AS failed_runs,
                        (SELECT COUNT(*) FROM analyses
                         WHERE created_at >= ?) AS analyses_today,
                        (SELECT COUNT(*) FROM processing_runs
                         WHERE started_at >= ?) AS runs_today
                    """
                    ,
                    (today.isoformat(), today.isoformat()),
                ).fetchone()
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError("Could not load productivity briefing data.") from exc
        counts = {"pending": 0, "in_progress": 0, "completed": 0}
        counts.update({row["status"]: row["total"] for row in counts_rows})
        return {
            "counts": counts,
            "tasks": [dict(row) for row in tasks],
            "recent_analyses": [dict(row) for row in analyses],
            "totals": dict(totals),
        }