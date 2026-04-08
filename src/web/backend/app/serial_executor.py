from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from app.db import Database, TERMINAL_STATES, utc_now_iso
from app.services.kg_service import KGBuildService, TaskCanceledError


@dataclass
class QueueStats:
    running_task_id: str | None
    queued: int


class SerialTaskExecutor:
    """Single-worker queue that executes heavyweight tasks serially."""

    def __init__(self, db: Database, poll_interval_sec: float = 0.2) -> None:
        self.db = db
        self.poll_interval_sec = poll_interval_sec
        self._kg_service = KGBuildService()

        self._queue: deque[str] = deque()
        self._cancel_requested: set[str] = set()

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._running_task_id: str | None = None
        self._shutdown = False

        self._recover_unfinished_tasks()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="rakg-web-serial-worker")
        self._worker.start()

    def submit_kg_task(self, payload: dict[str, Any]) -> str:
        task_id = uuid.uuid4().hex
        self.db.create_task(task_id=task_id, task_type="kg_build", input_payload=payload)
        self.db.log("INFO", "queue", "Task enqueued", task_id=task_id)

        with self._cond:
            self._queue.append(task_id)
            self._cond.notify()
        return task_id

    def cancel_task(self, task_id: str) -> tuple[bool, str]:
        task = self.db.get_task(task_id)
        if not task:
            return False, "not_found"

        status = task["status"]
        if status in TERMINAL_STATES:
            return False, status

        with self._cond:
            self.db.update_task_state(task_id, cancel_requested=True, message="cancel requested")
            self._cancel_requested.add(task_id)

            if status == "queued":
                # queued 任务可以直接从内存队列移除。
                self._queue = deque(item for item in self._queue if item != task_id)
                self.db.update_task_state(
                    task_id,
                    status="canceled",
                    progress=task.get("progress", 0.0),
                    finished_at=utc_now_iso(),
                    message="canceled before execution",
                )
                self.db.log("INFO", "queue", "Task canceled before execution", task_id=task_id)
                return True, "canceled"

        self.db.log("INFO", "queue", "Cancellation marked for running task", task_id=task_id)
        return True, "cancel_requested"

    def get_queue_stats(self) -> QueueStats:
        with self._lock:
            return QueueStats(
                running_task_id=self._running_task_id,
                queued=len(self._queue),
            )

    def shutdown(self) -> None:
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()
        self._worker.join(timeout=2)

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._shutdown:
                    self._cond.wait(timeout=self.poll_interval_sec)
                if self._shutdown:
                    return
                task_id = self._queue.popleft()
                self._running_task_id = task_id

            try:
                self._run_task(task_id)
            finally:
                with self._lock:
                    self._running_task_id = None
                time.sleep(self.poll_interval_sec)

    def _recover_unfinished_tasks(self) -> None:
        """
        Recover queued tasks from previous process lifetime.
        Running tasks are marked failed because in-process execution state is lost.
        """
        queued = self.db.get_tasks_by_status(["queued"])
        running = self.db.get_tasks_by_status(["running"])

        for task in running:
            task_id = task["task_id"]
            self.db.update_task_state(
                task_id,
                status="failed",
                message="worker interrupted by process restart",
                finished_at=utc_now_iso(),
                error_message="task interrupted by process restart",
            )
            self.db.log("ERROR", "queue", "Task marked failed during recovery", task_id=task_id)

        for task in queued:
            task_id = task["task_id"]
            if task.get("cancel_requested"):
                self.db.update_task_state(
                    task_id,
                    status="canceled",
                    finished_at=utc_now_iso(),
                    message="canceled during recovery",
                )
                continue
            self._queue.append(task_id)

        if queued:
            self.db.log("INFO", "queue", f"Recovered queued tasks: {len(self._queue)}")

    def _run_task(self, task_id: str) -> None:
        task = self.db.get_task(task_id)
        if not task:
            return
        if task["status"] in TERMINAL_STATES:
            return

        payload = task.get("input_payload", {})
        if task.get("cancel_requested"):
            self._finalize_canceled(task_id, message="canceled before start")
            return

        self.db.update_task_state(
            task_id,
            status="running",
            started_at=utc_now_iso(),
            progress=max(0.01, task.get("progress", 0.0)),
            message="running",
        )
        self.db.log("INFO", "worker", "Task started", task_id=task_id)

        def on_progress(progress: float, message: str) -> None:
            self.db.update_task_state(task_id, progress=progress, message=message)

        def on_log(level: str, message: str) -> None:
            self.db.log(level, "kg_builder", message, task_id=task_id)

        def is_cancel_requested() -> bool:
            return task_id in self._cancel_requested

        try:
            result = self._kg_service.run(
                task_id=task_id,
                payload=payload,
                on_progress=on_progress,
                on_log=on_log,
                is_cancel_requested=is_cancel_requested,
            )
            if is_cancel_requested():
                self._finalize_canceled(task_id, message="canceled")
                return

            self.db.update_task_state(
                task_id,
                status="succeeded",
                progress=1.0,
                message="succeeded",
                finished_at=utc_now_iso(),
                output_payload=result.output_payload,
            )
            self.db.log("INFO", "worker", "Task succeeded", task_id=task_id)
        except TaskCanceledError as exc:
            self._finalize_canceled(task_id, message=str(exc))
        except Exception as exc:  # noqa: BLE001
            self.db.update_task_state(
                task_id,
                status="failed",
                progress=1.0,
                message="failed",
                finished_at=utc_now_iso(),
                error_message=str(exc),
            )
            self.db.log("ERROR", "worker", f"Task failed: {exc}", task_id=task_id)
        finally:
            self._cancel_requested.discard(task_id)

    def _finalize_canceled(self, task_id: str, message: str) -> None:
        self.db.update_task_state(
            task_id,
            status="canceled",
            finished_at=utc_now_iso(),
            message=message,
        )
        self.db.log("INFO", "worker", f"Task canceled: {message}", task_id=task_id)
