"""Unified synchronous executor for single and batched LLM calls."""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable

import src.config as config_module
from src.logger import get_logger

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"
LOG_FILE_ENV_KEY = "RAKG_LOGGER_FILE"
DEFAULT_LOGGER_FILE = "Agent.log"

logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file=os.getenv(LOG_FILE_ENV_KEY, DEFAULT_LOGGER_FILE),
)


DEFAULT_MAX_WORKERS = 4


class _NoopProgressReporter(AbstractContextManager):
    def update(self, _count: int = 1) -> None:
        return None

    def close(self) -> None:
        return None

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self.close()
        return False


class _TextProgressReporter(_NoopProgressReporter):
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(int(total), 0)
        self.completed = 0
        self._last_reported = 0
        self._step = max(1, self.total // 10) if self.total else 1
        if self.total:
            sys.stderr.write(f"{self.label}: 0/{self.total}\n")
            sys.stderr.flush()

    def update(self, count: int = 1) -> None:
        if self.total <= 0:
            return
        self.completed = min(self.total, self.completed + max(int(count), 0))
        should_report = (
            self.completed == self.total
            or self.completed == 1
            or (self.completed - self._last_reported) >= self._step
        )
        if should_report:
            self._last_reported = self.completed
            sys.stderr.write(f"{self.label}: {self.completed}/{self.total}\n")
            sys.stderr.flush()


def _create_progress_reporter(label: str | None, total: int, enabled: bool):
    if not enabled or not label or total <= 0:
        return _NoopProgressReporter()

    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        tqdm = None

    if tqdm is not None:
        return tqdm(total=total, desc=label, dynamic_ncols=True, leave=False)
    return _TextProgressReporter(label, total)


@dataclass(frozen=True)
class LLMTask:
    """Description of a single LLM task."""

    kind: str
    payload: Any
    chain_factory: Callable[[], Any] | None = None
    invoke_fn: Callable[[Any], Any] | None = None
    parser: Callable[[Any], Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chain_factory is None and self.invoke_fn is None:
            raise ValueError("LLMTask requires either chain_factory or invoke_fn.")


@dataclass
class LLMTaskError:
    """Structured error returned for failed LLM tasks."""

    kind: str
    metadata: dict[str, Any]
    error: Exception
    traceback_text: str
    duration_seconds: float

    @property
    def message(self) -> str:
        return str(self.error)

    def reraise(self) -> None:
        raise self.error


class LLMExecutor:
    """Execute independent LLM calls sequentially or in a thread pool."""

    @staticmethod
    def _normalize_parallel_enabled(enable_parallel: bool | None) -> bool:
        if enable_parallel is None:
            return bool(getattr(config_module, "LLM_PARALLEL_ENABLED", False))
        return bool(enable_parallel)

    @staticmethod
    def _normalize_max_workers(max_workers: int | None) -> int:
        configured = getattr(config_module, "LLM_PARALLEL_MAX_WORKERS", DEFAULT_MAX_WORKERS)
        candidate = configured if max_workers is None else max_workers
        try:
            normalized = int(candidate)
        except (TypeError, ValueError):
            return DEFAULT_MAX_WORKERS
        if normalized < 1:
            return DEFAULT_MAX_WORKERS
        return normalized

    def _execute_task(self, task: LLMTask) -> Any | LLMTaskError:
        started_at = time.perf_counter()
        try:
            if task.invoke_fn is not None:
                raw_result = task.invoke_fn(task.payload)
            else:
                chain = task.chain_factory()
                raw_result = chain.invoke(task.payload)
            if task.parser is not None:
                return task.parser(raw_result)
            return raw_result
        except Exception as exc:
            return LLMTaskError(
                kind=task.kind,
                metadata=dict(task.metadata or {}),
                error=exc,
                traceback_text=traceback.format_exc(),
                duration_seconds=max(time.perf_counter() - started_at, 0.0),
            )

    def invoke(self, task: LLMTask) -> Any | LLMTaskError:
        """Execute one task and return its result or a structured error."""
        return self._execute_task(task)

    def invoke_batch(
        self,
        tasks: list[LLMTask],
        *,
        enable_parallel: bool | None = None,
        max_workers: int | None = None,
        preserve_order: bool = True,
        progress_label: str | None = None,
        progress_total: int | None = None,
        progress_enabled: bool = False,
        progress_callback: Callable[[int, int, str | None], None] | None = None,
    ) -> list[Any | LLMTaskError]:
        """Execute a batch of independent tasks with stable result alignment."""
        if not tasks:
            return []

        started_at = time.perf_counter()
        requested_parallel = self._normalize_parallel_enabled(enable_parallel)
        normalized_max_workers = self._normalize_max_workers(max_workers)
        actual_max_workers = min(normalized_max_workers, len(tasks))
        should_parallelize = requested_parallel and len(tasks) > 1 and actual_max_workers > 1
        task_kind = ",".join(sorted({str(task.kind or "unknown") for task in tasks}))
        if progress_total is None:
            normalized_progress_total = len(tasks)
        else:
            try:
                normalized_progress_total = max(int(progress_total), 0)
            except (TypeError, ValueError):
                normalized_progress_total = len(tasks)
        completed_count = 0

        def record_progress() -> None:
            nonlocal completed_count
            completed_count += 1
            if progress_callback is not None:
                progress_callback(completed_count, normalized_progress_total, progress_label)
            progress_reporter.update(1)

        if preserve_order:
            results: list[Any | LLMTaskError] = [None] * len(tasks)
        else:
            results = []

        with _create_progress_reporter(
            label=progress_label,
            total=normalized_progress_total,
            enabled=progress_enabled,
        ) as progress_reporter:
            if should_parallelize:
                with ThreadPoolExecutor(max_workers=actual_max_workers) as pool:
                    future_map = {
                        pool.submit(self._execute_task, task): idx for idx, task in enumerate(tasks)
                    }
                    for future in as_completed(future_map):
                        idx = future_map[future]
                        result = future.result()
                        if preserve_order:
                            results[idx] = result
                        else:
                            results.append(result)
                        record_progress()
            else:
                for idx, task in enumerate(tasks):
                    result = self._execute_task(task)
                    if preserve_order:
                        results[idx] = result
                    else:
                        results.append(result)
                    record_progress()

        success_count = sum(1 for item in results if not isinstance(item, LLMTaskError))
        failure_count = len(results) - success_count
        logger.info(
            "LLM batch finished: label=%s kind=%s tasks=%s parallel=%s max_workers=%s success=%s failure=%s elapsed=%.3fs",
            progress_label or task_kind,
            task_kind,
            len(tasks),
            should_parallelize,
            actual_max_workers,
            success_count,
            failure_count,
            max(time.perf_counter() - started_at, 0.0),
        )
        return results
