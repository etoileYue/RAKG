"""Unified synchronous executor for single and batched LLM calls."""

from __future__ import annotations

import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        if preserve_order:
            results: list[Any | LLMTaskError] = [None] * len(tasks)
        else:
            results = []

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
        else:
            for idx, task in enumerate(tasks):
                result = self._execute_task(task)
                if preserve_order:
                    results[idx] = result
                else:
                    results.append(result)

        success_count = sum(1 for item in results if not isinstance(item, LLMTaskError))
        failure_count = len(results) - success_count
        logger.info(
            "LLM batch finished: kind=%s tasks=%s parallel=%s max_workers=%s success=%s failure=%s elapsed=%.3fs",
            task_kind,
            len(tasks),
            should_parallelize,
            actual_max_workers,
            success_count,
            failure_count,
            max(time.perf_counter() - started_at, 0.0),
        )
        return results
