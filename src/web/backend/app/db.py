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

                CREATE TABLE IF NOT EXISTS qa_conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    kg_path TEXT NOT NULL,
                    auto_title INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS qa_messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    params_snapshot TEXT NOT NULL DEFAULT '{}',
                    qa_response_snapshot TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES qa_conversations(id)
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_status_created
                ON tasks(status, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_task_logs_task_created
                ON task_logs(task_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_qa_conversations_updated
                ON qa_conversations(updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_qa_messages_conversation_created
                ON qa_messages(conversation_id, created_at ASC);
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

    def list_tasks_by_filters(
        self,
        *,
        task_type: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = []
        values: list[Any] = []
        if task_type:
            filters.append("task_type = ?")
            values.append(task_type)
        if status:
            filters.append("status = ?")
            values.append(status)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM tasks
                {where_clause}
                ORDER BY created_at DESC
                """,
                values,
            ).fetchall()

        return [self._row_to_task(row) for row in rows]

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

    def create_qa_conversation(
        self,
        *,
        conversation_id: str,
        kg_path: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        normalized_title = (title or "").strip()
        auto_title = 0 if normalized_title else 1

        with self._write_lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO qa_conversations(id, title, kg_path, auto_title, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, normalized_title, kg_path, auto_title, now, now),
            )
            conn.commit()
        return {
            "id": conversation_id,
            "title": normalized_title,
            "kg_path": kg_path,
            "auto_title": bool(auto_title),
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "last_message_preview": None,
        }

    def list_qa_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.kg_path,
                    c.auto_title,
                    c.created_at,
                    c.updated_at,
                    (
                        SELECT COUNT(1)
                        FROM qa_messages m
                        WHERE m.conversation_id = c.id
                    ) AS message_count,
                    (
                        SELECT m.content
                        FROM qa_messages m
                        WHERE m.conversation_id = c.id
                        ORDER BY m.created_at DESC
                        LIMIT 1
                    ) AS last_message_preview
                FROM qa_conversations c
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return [self._row_to_conversation(row) for row in rows]

    def get_qa_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.kg_path,
                    c.auto_title,
                    c.created_at,
                    c.updated_at,
                    (
                        SELECT COUNT(1)
                        FROM qa_messages m
                        WHERE m.conversation_id = c.id
                    ) AS message_count,
                    (
                        SELECT m.content
                        FROM qa_messages m
                        WHERE m.conversation_id = c.id
                        ORDER BY m.created_at DESC
                        LIMIT 1
                    ) AS last_message_preview
                FROM qa_conversations c
                WHERE c.id = ?
                """,
                (conversation_id,),
            ).fetchone()
        return self._row_to_conversation(row) if row else None

    def list_qa_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    role,
                    content,
                    params_snapshot,
                    qa_response_snapshot,
                    created_at
                FROM qa_messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def delete_qa_conversation(self, conversation_id: str) -> bool:
        with self._write_lock, self._connect() as conn:
            row = conn.execute("SELECT id FROM qa_conversations WHERE id = ?", (conversation_id,)).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM qa_messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM qa_conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return True

    def create_qa_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        role: str,
        content: str,
        params_snapshot: dict[str, Any] | None = None,
        qa_response_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        now = utc_now_iso()
        params_payload = params_snapshot or {}
        response_payload = qa_response_snapshot or {}

        with self._write_lock, self._connect() as conn:
            conversation_row = conn.execute(
                "SELECT id, title, auto_title FROM qa_conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation_row is None:
                return None

            conn.execute(
                """
                INSERT INTO qa_messages(
                    id, conversation_id, role, content, params_snapshot, qa_response_snapshot, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    json.dumps(params_payload, ensure_ascii=False, default=str),
                    json.dumps(response_payload, ensure_ascii=False, default=str),
                    now,
                ),
            )

            should_update_title = (
                role == "user"
                and bool(conversation_row["auto_title"])
                and not str(conversation_row["title"] or "").strip()
            )
            if should_update_title:
                conn.execute(
                    "UPDATE qa_conversations SET title = ? WHERE id = ?",
                    (self._build_conversation_title(content), conversation_id),
                )

            conn.execute(
                "UPDATE qa_conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            conn.commit()

        return {
            "id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "params_snapshot": params_payload,
            "qa_response_snapshot": response_payload,
            "created_at": now,
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

    @staticmethod
    def _row_to_conversation(row: sqlite3.Row) -> dict[str, Any]:
        if row is None:
            return {}
        conversation = dict(row)
        conversation["auto_title"] = bool(conversation.get("auto_title", 0))
        conversation["title"] = str(conversation.get("title") or "").strip()
        conversation["message_count"] = int(conversation.get("message_count") or 0)
        preview = conversation.get("last_message_preview")
        conversation["last_message_preview"] = str(preview).strip() if preview else None
        return conversation

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
        if row is None:
            return {}
        message = dict(row)
        message["params_snapshot"] = json.loads(message.get("params_snapshot") or "{}")
        message["qa_response_snapshot"] = json.loads(message.get("qa_response_snapshot") or "{}")
        return message

    @staticmethod
    def _build_conversation_title(content: str, max_len: int = 24) -> str:
        normalized = " ".join(content.split()).strip()
        if len(normalized) <= max_len:
            return normalized
        return f"{normalized[:max_len].rstrip()}..."
