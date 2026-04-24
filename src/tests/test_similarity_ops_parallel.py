import time
import unittest
from unittest import mock

import src.config as config_module
from src.pipeline.similarity_ops import PipelineSimilarityOpsMixin


class _DummyResponse:
    def __init__(self, content):
        self.content = content


class _DynamicChain:
    def __init__(self, response_map, delay_map, recorder):
        self.response_map = response_map
        self.delay_map = delay_map
        self.recorder = recorder

    def invoke(self, payload):
        pair = (
            payload["entity1"]["name"],
            payload["entity2"]["name"],
        )
        self.recorder["invoke_count"] = self.recorder.get("invoke_count", 0) + 1
        time.sleep(self.delay_map.get(pair, 0.0))
        return _DummyResponse(self.response_map[pair])


class _DynamicPrompt:
    def __init__(self, response_map, delay_map, recorder):
        self.response_map = response_map
        self.delay_map = delay_map
        self.recorder = recorder

    def __or__(self, _model):
        return _DynamicChain(self.response_map, self.delay_map, self.recorder)


class _DummySimilarityPipeline(PipelineSimilarityOpsMixin):
    def __init__(self):
        self.similarity_model = object()
        self.disambiguation_config = self.set_disambiguation_config()
        self.reset_disambiguation_runtime_state()


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def invoke_batch(self, tasks, **kwargs):
        self.calls.append(kwargs)
        return [task.invoke_fn(task.payload) for task in tasks]


class SimilarityOpsParallelTests(unittest.TestCase):
    def test_parallel_first_pass_preserves_positive_order(self):
        pipeline = _DummySimilarityPipeline()
        recorder = {}
        entities = {
            "e1": {"name": "Alpha", "type": "Person", "description": "d1"},
            "e2": {"name": "Beta", "type": "Person", "description": "d2"},
            "e3": {"name": "Gamma", "type": "Person", "description": "d3"},
            "e4": {"name": "Delta", "type": "Person", "description": "d4"},
        }
        response_map = {
            ("Alpha", "Beta"): '{"result": true}',
            ("Gamma", "Delta"): '{"result": true}',
        }
        delay_map = {
            ("Alpha", "Beta"): 0.08,
            ("Gamma", "Delta"): 0.01,
        }

        with mock.patch.object(config_module, "LLM_PARALLEL_ENABLED", True):
            with mock.patch.object(config_module, "LLM_PARALLEL_MAX_WORKERS", 4):
                with mock.patch(
                    "src.pipeline.similarity_ops.ChatPromptTemplate.from_template",
                    return_value=_DynamicPrompt(response_map, delay_map, recorder),
                ):
                    positives, gray_queue, resolved, metrics = pipeline._run_two_pass_similarity_disambiguation(
                        candidates=[("e1", "e2", 0.91), ("e3", "e4", 0.83)],
                        left_entities=entities,
                        right_entities=entities,
                        scope_label="entity",
                    )

        self.assertEqual(positives, [("e1", "e2", 0.91), ("e3", "e4", 0.83)])
        self.assertEqual(gray_queue, [])
        self.assertEqual(resolved, 0)
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertEqual(metrics["llm_calls_saved"], 0)
        self.assertEqual(recorder["invoke_count"], 2)

    def test_similarity_progress_metadata_uses_pass_specific_task_counts(self):
        entities = {
            "e1": {"name": "Alpha", "type": "Person", "description": "d1"},
            "e2": {"name": "Beta", "type": "Person", "description": "d2"},
            "e3": {"name": "Gamma", "type": "Person", "description": "d3"},
        }
        response_map = {
            ("Alpha", "Beta"): '{"result": true}',
            ("Alpha", "Gamma"): '{"result": true}',
        }

        with mock.patch(
            "src.pipeline.similarity_ops.ChatPromptTemplate.from_template",
            return_value=_DynamicPrompt(response_map, {}, {}),
        ):
            first_pass_pipeline = _DummySimilarityPipeline()
            first_pass_pipeline._llm_executor = _RecordingExecutor()
            first_pass_pipeline._run_similarity_batch(
                pairs=[("e1", "e2", 0.91), ("e1", "e3", 0.82)],
                left_entities=entities,
                right_entities=entities,
                scope_label="entity",
                pass_label="first-pass",
            )

            second_pass_pipeline = _DummySimilarityPipeline()
            second_pass_pipeline._llm_executor = _RecordingExecutor()
            second_pass_pipeline._run_similarity_batch(
                pairs=[("e1", "e2", 0.91)],
                left_entities=entities,
                right_entities=entities,
                scope_label="entity",
                pass_label="second-pass",
            )

        self.assertEqual(
            first_pass_pipeline._llm_executor.calls,
            [
                {
                    "progress_label": "SIM first-pass",
                    "progress_total": 2,
                    "progress_enabled": True,
                }
            ],
        )
        self.assertEqual(
            second_pass_pipeline._llm_executor.calls,
            [
                {
                    "progress_label": "SIM second-pass",
                    "progress_total": 1,
                    "progress_enabled": True,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
