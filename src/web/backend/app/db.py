from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


TERMINAL_STATES = {"succeeded", "failed", "canceled"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thin SQLite wrapper used by task queue and API handlers."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    input_payload TEXT NOT NULL,
                    output_payload TEXT NOT NULL DEFAULT '{}',
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS task_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status_created
                ON tasks(status, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_task_logs_task_created
                ON task_logs(task_id, created_at DESC);
                """
            )
            conn.commit()

    def create_task(self, task_id: str, task_type: str, input_payload: dict[str, Any]) -> None:
        now = utc_now_iso()
        with self._write_lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(
                    task_id, task_type, status, progress, message,
                    input_payload, output_payload, created_at
                ) VALUES(?, ?, 'queued', 0, 'queued', ?, '{}', ?)
                """,
                (task_id, task_type, json.dumps(input_payload, ensure_ascii=False), now),
            )
            conn.commit()

    def log(self, level: str, source: str, message: str, task_id: str | None = None) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_logs(task_id, level, source, message, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (task_id, level.upper(), source, message, utc_now_iso()),
            )
            conn.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def get_tasks_by_status(self, statuses: list[str]) -> list[dict[str, Any]]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM tasks
                WHERE status IN ({placeholders})
                ORDER BY created_at ASC
                """,
                statuses,
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def list_tasks(
        self,
        page: int,
        page_size: int,
        task_type: str | None,
        status: str | None,
    ) -> dict[str, Any]:
        filters = []
        values: list[Any] = []
        if task_type:
            filters.append("task_type = ?")
            values.append(task_type)
        if status:
            filters.append("status = ?")
            values.append(status)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        offset = (page - 1) * page_size

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(1) AS cnt FROM tasks {where_clause}",
                values,
            ).fetchone()["cnt"]
            rows = conn.execute(
                f"""
                SELECT * FROM tasks
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*values, page_size, offset],
            ).fetchall()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [self._row_to_task(row) for row in rows],
        }

    def update_task_state(
        self,
        task_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error_message: str | None = None,
        output_payload: dict[str, Any] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        cancel_requested: bool | None = None,
    ) -> None:
        fields: list[str] = []
        values: list[Any] = []

        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if progress is not None:
            fields.append("progress = ?")
            values.append(max(0.0, min(progress, 1.0)))
        if message is not None:
            fields.append("message = ?")
            values.append(message)
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)
        if output_payload is not None:
            fields.append("output_payload = ?")
            values.append(json.dumps(output_payload, ensure_ascii=False))
        if started_at is not None:
            fields.append("started_at = ?")
            values.append(started_at)
        if finished_at is not None:
            fields.append("finished_at = ?")
            values.append(finished_at)
        if cancel_requested is not None:
            fields.append("cancel_requested = ?")
            values.append(1 if cancel_requested else 0)

        if not fields:
            return

        values.append(task_id)
        with self._write_lock, self._connect() as conn:
            conn.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE task_id = ?",
                values,
            )
            conn.commit()

    def fetch_logs(
        self,
        page: int,
        page_size: int,
        task_id: str | None,
        keyword: str | None,
    ) -> dict[str, Any]:
        filters = []
        values: list[Any] = []
        if task_id:
            filters.append("task_id = ?")
            values.append(task_id)
        if keyword:
            filters.append("message LIKE ?")
            values.append(f"%{keyword}%")

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        offset = (page - 1) * page_size

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(1) AS cnt FROM task_logs {where_clause}", values
            ).fetchone()["cnt"]
            rows = conn.execute(
                f"""
                SELECT id, task_id, level, source, message, created_at
                FROM task_logs
                {where_clause}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                [*values, page_size, offset],
            ).fetchall()

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [dict(row) for row in rows],
        }

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
        if row is None:
            return {}
        task = dict(row)
        task["input_payload"] = json.loads(task.get("input_payload") or "{}")
        task["output_payload"] = json.loads(task.get("output_payload") or "{}")
        task["cancel_requested"] = bool(task.get("cancel_requested", 0))
        return task
