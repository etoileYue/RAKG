import threading
import time
import unittest
from unittest import mock

import src.config as config_module
from src.llm_executor import DEFAULT_MAX_WORKERS
from src.llm_executor import LLMExecutor
from src.llm_executor import LLMTask
from src.llm_executor import LLMTaskError


class _SleepChain:
    def __init__(self, value, delay):
        self.value = value
        self.delay = delay

    def invoke(self, _payload):
        time.sleep(self.delay)
        return self.value


class LLMExecutorTests(unittest.TestCase):
    def test_invoke_batch_sequential_mode_preserves_order(self):
        executor = LLMExecutor()
        recorder = []

        def make_task(index):
            return LLMTask(
                kind="ner",
                payload={"index": index},
                invoke_fn=lambda payload, idx=index: recorder.append(idx) or payload["index"] + 10,
            )

        with mock.patch.object(config_module, "LLM_PARALLEL_ENABLED", False):
            results = executor.invoke_batch([make_task(0), make_task(1), make_task(2)])

        self.assertEqual(results, [10, 11, 12])
        self.assertEqual(recorder, [0, 1, 2])

    def test_invoke_batch_parallel_mode_preserves_input_order(self):
        executor = LLMExecutor()
        threads_seen = set()

        tasks = [
            LLMTask(
                kind="relation",
                payload=None,
                chain_factory=lambda delay=delay, value=value: _SleepChain(value, delay),
                parser=lambda result: threads_seen.add(threading.current_thread().name) or result,
            )
            for delay, value in [(0.08, "first"), (0.01, "second"), (0.03, "third")]
        ]

        with mock.patch.object(config_module, "LLM_PARALLEL_ENABLED", True):
            with mock.patch.object(config_module, "LLM_PARALLEL_MAX_WORKERS", 3):
                results = executor.invoke_batch(tasks)

        self.assertEqual(results, ["first", "second", "third"])
        self.assertGreaterEqual(len(threads_seen), 2)

    def test_invalid_max_workers_falls_back_to_default(self):
        executor = LLMExecutor()
        self.assertEqual(executor._normalize_max_workers(0), DEFAULT_MAX_WORKERS)
        self.assertEqual(executor._normalize_max_workers(-3), DEFAULT_MAX_WORKERS)
        self.assertEqual(executor._normalize_max_workers("bad"), DEFAULT_MAX_WORKERS)

    def test_invoke_batch_reports_progress_in_sequential_mode(self):
        executor = LLMExecutor()
        progress_events = []

        tasks = [
            LLMTask(kind="ner", payload={"index": idx}, invoke_fn=lambda payload: payload["index"])
            for idx in range(3)
        ]

        with mock.patch.object(config_module, "LLM_PARALLEL_ENABLED", False):
            results = executor.invoke_batch(
                tasks,
                progress_label="SIM first-pass",
                progress_total=3,
                progress_enabled=False,
                progress_callback=lambda completed, total, label: progress_events.append(
                    (completed, total, label)
                ),
            )

        self.assertEqual(results, [0, 1, 2])
        self.assertEqual(
            progress_events,
            [
                (1, 3, "SIM first-pass"),
                (2, 3, "SIM first-pass"),
                (3, 3, "SIM first-pass"),
            ],
        )

    def test_invoke_batch_reports_progress_in_parallel_mode(self):
        executor = LLMExecutor()
        progress_events = []
        tasks = [
            LLMTask(
                kind="relation",
                payload={"index": idx},
                invoke_fn=lambda payload, delay=delay: time.sleep(delay) or payload["index"],
            )
            for idx, delay in [(0, 0.06), (1, 0.01), (2, 0.03)]
        ]

        with mock.patch.object(config_module, "LLM_PARALLEL_ENABLED", True):
            with mock.patch.object(config_module, "LLM_PARALLEL_MAX_WORKERS", 3):
                results = executor.invoke_batch(
                    tasks,
                    progress_label="REL",
                    progress_total=3,
                    progress_enabled=False,
                    progress_callback=lambda completed, total, label: progress_events.append(
                        (completed, total, label)
                    ),
                )

        self.assertEqual(results, [0, 1, 2])
        self.assertEqual(progress_events[-1], (3, 3, "REL"))
        self.assertEqual(sorted(completed for completed, _, _ in progress_events), [1, 2, 3])

    def test_task_error_contains_metadata(self):
        executor = LLMExecutor()
        result = executor.invoke(
            LLMTask(
                kind="similarity",
                payload=None,
                invoke_fn=lambda _payload: (_ for _ in ()).throw(ValueError("boom")),
                metadata={"pair": ("e1", "e2")},
            )
        )

        self.assertIsInstance(result, LLMTaskError)
        self.assertEqual(result.kind, "similarity")
        self.assertEqual(result.metadata["pair"], ("e1", "e2"))
        self.assertEqual(result.message, "boom")


if __name__ == "__main__":
    unittest.main()
