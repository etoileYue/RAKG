import time
import unittest
from unittest import mock

import src.config as config_module
from src.pipeline.io_ops import PipelineIOOpsMixin
from src.pipeline.relation_ops import PipelineRelationOpsMixin


class _DummyResponse:
    def __init__(self, content):
        self.content = content


class _DynamicChain:
    def __init__(self, response_map, delay_map):
        self.response_map = response_map
        self.delay_map = delay_map

    def invoke(self, payload):
        key = payload.get("text") or payload.get("target_entity")
        time.sleep(self.delay_map.get(key, 0.0))
        return _DummyResponse(self.response_map[key])


class _DynamicPrompt:
    def __init__(self, response_map, delay_map):
        self.response_map = response_map
        self.delay_map = delay_map

    def __or__(self, _model):
        return _DynamicChain(self.response_map, self.delay_map)


class _DummyRelationPipeline(PipelineIOOpsMixin, PipelineRelationOpsMixin):
    def __init__(self):
        self.model = object()
        self.appended = []

    def _append_jsonl(self, output_file, data):
        self.appended.append((output_file, data))

    def _build_relation_context(self, **kwargs):
        entity = kwargs["entity"]
        name = entity["name"]
        chunk_id = entity["chunkid"][0]
        return {
            "candidate_chunk_ids": [chunk_id],
            "candidate_chunks": {chunk_id: f"{name} context"},
            "evidence_blocks": [],
            "chunk_text": f"{name} context",
        }


class _RecordingExecutor:
    def __init__(self):
        self.calls = []

    def invoke_batch(self, tasks, **kwargs):
        self.calls.append(kwargs)
        return [task.invoke_fn(task.payload) for task in tasks]


class RelationOpsParallelTests(unittest.TestCase):
    def _run_extract(self, parallel_enabled):
        pipeline = _DummyRelationPipeline()
        text_list = ["Alice leads Acme.", "She founded it."]
        sent_to_id = {
            "Alice leads Acme.": "c1",
            "She founded it.": "c2",
        }
        response_map = {
            "Alice leads Acme.": '{"entity1": {"name": "Alice", "type": "Person", "description": "Leader"}}',
            "She founded it.": '{"State": false}',
            "Alice leads Acme. She founded it.": '{"entity1": {"name": "Acme", "type": "Org", "description": "Company"}}',
        }
        delay_map = {
            "Alice leads Acme.": 0.08,
            "She founded it.": 0.01,
            "Alice leads Acme. She founded it.": 0.02,
        }

        with mock.patch.object(config_module, "LLM_PARALLEL_ENABLED", parallel_enabled):
            with mock.patch.object(config_module, "LLM_PARALLEL_MAX_WORKERS", 4):
                with mock.patch(
                    "src.pipeline.relation_ops.ChatPromptTemplate.from_template",
                    return_value=_DynamicPrompt(response_map, delay_map),
                ):
                    result = pipeline.extract_from_text_multiply(
                        text_list,
                        sent_to_id,
                        output_file="/tmp/ner_parallel.jsonl",
                    )
        return pipeline, result

    def test_extract_from_text_multiply_matches_between_parallel_modes(self):
        sequential_pipeline, sequential_result = self._run_extract(False)
        parallel_pipeline, parallel_result = self._run_extract(True)

        self.assertEqual(parallel_result, sequential_result)
        self.assertEqual(list(parallel_result.keys()), ["entity1", "entity2"])
        self.assertEqual(parallel_result["entity1"]["name"], "Alice")
        self.assertEqual(parallel_result["entity1"]["chunkid"], ["c1"])
        self.assertEqual(parallel_result["entity2"]["name"], "Acme")
        self.assertEqual(parallel_result["entity2"]["chunkid"], ["c2", "c1"])

        self.assertEqual(
            [item[1]["text"] for item in parallel_pipeline.appended],
            ["Alice leads Acme.", "She founded it.", "Alice leads Acme. She founded it."],
        )
        self.assertEqual(parallel_pipeline.appended, sequential_pipeline.appended)

    def test_get_target_kg_all_matches_between_parallel_modes(self):
        entity_dic = {
            "entity1": {
                "name": "Alice",
                "type": "Person",
                "description": "Founder",
                "chunkid": ["c1"],
            },
            "entity2": {
                "name": "Acme",
                "type": "Org",
                "description": "Company",
                "chunkid": ["c2"],
            },
        }
        response_map = {
            "Alice context": '{"central_entity": {"name": "Alice", "type": "Person", "description": "Founder", "attributes": [], "relationships": []}}',
            "Acme context": '{"central_entity": {"name": "Acme", "type": "Org", "description": "Company", "attributes": [], "relationships": []}}',
        }
        delay_map = {
            "Alice context": 0.08,
            "Acme context": 0.01,
        }

        def run(parallel_enabled):
            pipeline = _DummyRelationPipeline()
            with mock.patch.object(config_module, "LLM_PARALLEL_ENABLED", parallel_enabled):
                with mock.patch.object(config_module, "LLM_PARALLEL_MAX_WORKERS", 4):
                    with mock.patch(
                        "src.pipeline.relation_ops.ChatPromptTemplate.from_template",
                        return_value=_DynamicPrompt(response_map, delay_map),
                    ):
                        result = pipeline.get_target_kg_all(
                            entity_dic=entity_dic,
                            id_to_sentence={"c1": "Alice context", "c2": "Acme context"},
                            sentences=["Alice context", "Acme context"],
                            sentence_to_id={"Alice context": "c1", "Acme context": "c2"},
                            vectors=[[1.0, 0.0], [0.0, 1.0]],
                            output_file="/tmp/relation_parallel.jsonl",
                        )
            return pipeline, result

        sequential_pipeline, sequential_result = run(False)
        parallel_pipeline, parallel_result = run(True)

        self.assertEqual(parallel_result, sequential_result)
        self.assertEqual(list(parallel_result.keys()), ["entity1", "entity2"])
        self.assertEqual(
            [item[1]["entity"]["name"] for item in parallel_pipeline.appended],
            ["Alice", "Acme"],
        )
        self.assertEqual(parallel_pipeline.appended, sequential_pipeline.appended)

    def test_get_target_kg_all_passes_rel_progress_metadata(self):
        pipeline = _DummyRelationPipeline()
        pipeline._llm_executor = _RecordingExecutor()
        entity_dic = {
            "entity1": {
                "name": "Alice",
                "type": "Person",
                "description": "Founder",
                "chunkid": ["c1"],
            },
            "entity2": {
                "name": "Acme",
                "type": "Org",
                "description": "Company",
                "chunkid": ["c2"],
            },
        }
        response_map = {
            "Alice context": '{"central_entity": {"name": "Alice", "type": "Person", "description": "Founder", "attributes": [], "relationships": []}}',
            "Acme context": '{"central_entity": {"name": "Acme", "type": "Org", "description": "Company", "attributes": [], "relationships": []}}',
        }

        with mock.patch(
            "src.pipeline.relation_ops.ChatPromptTemplate.from_template",
            return_value=_DynamicPrompt(response_map, {}),
        ):
            result = pipeline.get_target_kg_all(
                entity_dic=entity_dic,
                id_to_sentence={"c1": "Alice context", "c2": "Acme context"},
                sentences=["Alice context", "Acme context"],
                sentence_to_id={"Alice context": "c1", "Acme context": "c2"},
                vectors=[[1.0, 0.0], [0.0, 1.0]],
                output_file="/tmp/relation_parallel.jsonl",
            )

        self.assertEqual(list(result.keys()), ["entity1", "entity2"])
        self.assertEqual(
            pipeline._llm_executor.calls,
            [
                {
                    "progress_label": "REL",
                    "progress_total": 2,
                    "progress_enabled": True,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
