import unittest
from unittest import mock

import src.config as config_module
from src.pipeline.similarity_ops import PipelineSimilarityOpsMixin


class DummyResponse:
    def __init__(self, content):
        self.content = content


class FakeChain:
    def __init__(self, response_text, recorder):
        self.response_text = response_text
        self.recorder = recorder

    def invoke(self, payload):
        self.recorder["invoke_count"] = self.recorder.get("invoke_count", 0) + 1
        self.recorder["payload"] = payload
        return DummyResponse(self.response_text)


class FakePrompt:
    def __init__(self, response_text, recorder):
        self.response_text = response_text
        self.recorder = recorder

    def __or__(self, _model):
        return FakeChain(self.response_text, self.recorder)


class DummySimilarityPipeline(PipelineSimilarityOpsMixin):
    def __init__(self):
        self.similarity_model = object()
        self.disambiguation_config = self.set_disambiguation_config()
        self.reset_disambiguation_runtime_state()


class SimilarityDisambiguationOptimizationTests(unittest.TestCase):
    def test_optimize_candidates_applies_type_gate_topk_and_direct_merge(self):
        pipeline = DummySimilarityPipeline()
        entities = {
            "e1": {"name": "Alice", "type": "Person", "description": "d1"},
            "e2": {"name": "Paris", "type": "Location", "description": "d2"},
            "e3": {"name": " Alice ", "type": "人物", "description": "d3"},
            "e4": {"name": "Bob", "type": "Person", "description": "d4"},
        }
        candidates = [
            ("e1", "e2", 0.96),
            ("e1", "e3", 0.95),
            ("e1", "e4", 0.94),
        ]

        with mock.patch.object(config_module, "DISAMBIGUATION_PER_ENTITY_TOP_K", 1):
            config = pipeline.set_disambiguation_config()
            optimized = pipeline._optimize_similarity_candidates(
                candidates=candidates,
                left_entities=entities,
                right_entities=entities,
                same_side_compare=True,
                disambiguation_config=config,
            )

        self.assertEqual(optimized["raw_candidates"], 3)
        self.assertEqual(optimized["after_type_gate"], 2)
        self.assertEqual(optimized["after_topk"], 1)
        self.assertEqual(optimized["direct_merged"], 1)
        self.assertEqual(optimized["direct_pairs"], [("e1", "e3", 0.95)])
        self.assertEqual(optimized["llm_candidates"], [])

    def test_similarity_llm_single_truncates_description_payload(self):
        pipeline = DummySimilarityPipeline()
        recorder = {}

        with mock.patch.object(config_module, "DISAMBIGUATION_DESCRIPTION_MAX_CHARS", 8):
            pipeline.set_disambiguation_config()
            with mock.patch(
                "src.pipeline.similarity_ops.ChatPromptTemplate.from_template",
                return_value=FakePrompt('{"result": true}', recorder),
            ):
                result = pipeline.similarity_llm_single(
                    {
                        "name": "EntityA",
                        "type": "Person",
                        "description": "123456789012345",
                    },
                    {
                        "name": "EntityB",
                        "type": "Person",
                        "description": "abcdefghijk",
                    },
                )

        self.assertTrue(result["result"])
        self.assertEqual(recorder["payload"]["entity1"]["description"], "12345678")
        self.assertEqual(recorder["payload"]["entity2"]["description"], "abcdefgh")

    def test_two_pass_similarity_uses_pair_cache(self):
        pipeline = DummySimilarityPipeline()
        recorder = {}
        entities = {
            "e1": {"name": "A", "type": "Person", "description": "desc"},
            "e2": {"name": "B", "type": "Person", "description": "desc"},
        }

        with mock.patch(
            "src.pipeline.similarity_ops.ChatPromptTemplate.from_template",
            return_value=FakePrompt("not_json", recorder),
        ):
            positives, gray_queue, resolved, metrics = pipeline._run_two_pass_similarity_disambiguation(
                candidates=[("e1", "e2", 0.9)],
                left_entities=entities,
                right_entities=entities,
                scope_label="entity",
            )

        self.assertEqual(positives, [])
        self.assertEqual(len(gray_queue), 1)
        self.assertEqual(resolved, 0)
        self.assertEqual(metrics["llm_calls"], 1)
        self.assertEqual(metrics["llm_calls_saved"], 1)
        self.assertEqual(recorder.get("invoke_count"), 1)

    def test_invalid_global_disambiguation_config_falls_back(self):
        pipeline = DummySimilarityPipeline()

        with mock.patch.object(config_module, "DISAMBIGUATION_SIMILARITY_THRESHOLD", 1.5):
            with mock.patch.object(config_module, "DISAMBIGUATION_PER_ENTITY_TOP_K", 0):
                with mock.patch.object(config_module, "DISAMBIGUATION_DESCRIPTION_MAX_CHARS", -10):
                    resolved = pipeline.set_disambiguation_config()

        self.assertEqual(resolved["similarity_threshold"], 0.60)
        self.assertEqual(resolved["per_entity_top_k"], 8)
        self.assertEqual(resolved["description_max_chars"], 160)


if __name__ == "__main__":
    unittest.main()
