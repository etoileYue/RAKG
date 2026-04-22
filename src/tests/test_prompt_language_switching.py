import unittest
from unittest import mock

import src.config as config_module
import src.prompt as prompt_module
from src.pipeline.qa_answer_ops import PipelineQAAnswerOpsMixin
from src.pipeline.qa_match_ops import PipelineQAMatchOpsMixin
from src.pipeline.relation_ops import PipelineRelationOpsMixin
from src.pipeline.similarity_ops import PipelineSimilarityOpsMixin


class DummyResponse:
    def __init__(self, content):
        self.content = content


class FakeChain:
    def __init__(self, template, response_text, recorder):
        self.template = template
        self.response_text = response_text
        self.recorder = recorder

    def invoke(self, payload):
        self.recorder["payload"] = payload
        return DummyResponse(self.response_text)


class FakePrompt:
    def __init__(self, template, response_text, recorder):
        self.template = template
        self.response_text = response_text
        self.recorder = recorder

    def __or__(self, model):
        self.recorder["template"] = self.template
        self.recorder["model"] = model
        return FakeChain(self.template, self.response_text, self.recorder)


class DummyRelationPipeline(PipelineRelationOpsMixin):
    def __init__(self):
        self.model = object()
        self.appended = []

    @staticmethod
    def _dedupe_preserve_order(items):
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _append_jsonl(self, output_file, data):
        self.appended.append((output_file, data))

    def _build_relation_context(self, **kwargs):
        return {
            "candidate_chunk_ids": ["c1"],
            "candidate_chunks": {"c1": "Alice founded Acme."},
            "evidence_blocks": [
                {
                    "chunk_id": "c1",
                    "source": "entity_chunk",
                    "similarity": 1.0,
                    "rank": 1,
                    "text": "Alice founded Acme.",
                }
            ],
            "chunk_text": "[chunk_id=c1] Alice founded Acme.",
        }


class DummySimilarityPipeline(PipelineSimilarityOpsMixin):
    def __init__(self):
        self.similarity_model = object()


class DummyQAMatchPipeline(PipelineQAMatchOpsMixin):
    def __init__(self):
        self.model = object()


class DummyQAAnswerPipeline(PipelineQAAnswerOpsMixin):
    def __init__(self):
        self.model = object()

    def build_qa_retrieval_context(self, **kwargs):
        return {
            "context_text": "source=chunk:c1 Alice founded Acme.",
            "graph_paths": ["Alice -> founded -> Acme"],
            "evidence_items": [
                {"source": "chunk:c1", "text": "Alice founded Acme."},
            ],
        }


def make_template_factory(response_text, recorder):
    return lambda template: FakePrompt(template, response_text, recorder)


class PromptSelectorTests(unittest.TestCase):
    def test_get_prompt_returns_chinese_template_when_language_is_zh(self):
        with mock.patch.object(config_module, "PROMPT_LANGUAGE", "zh"):
            self.assertEqual(
                prompt_module.get_prompt("text2entity"),
                prompt_module.text2entity_zh,
            )

    def test_get_prompt_returns_english_template_when_language_is_en(self):
        with mock.patch.object(config_module, "PROMPT_LANGUAGE", "en"):
            self.assertEqual(
                prompt_module.get_prompt("qa_answer"),
                prompt_module.kg_qa_answer_prompt_en,
            )

    def test_get_prompt_raises_on_invalid_language(self):
        with mock.patch.object(config_module, "PROMPT_LANGUAGE", "fr"):
            with self.assertRaisesRegex(ValueError, "Unsupported PROMPT_LANGUAGE"):
                prompt_module.get_prompt("text2entity")

    def test_get_prompt_raises_on_unknown_prompt_key(self):
        with self.assertRaisesRegex(ValueError, "Unknown prompt key"):
            prompt_module.get_prompt("not_a_real_prompt", prompt_language="zh")


class RuntimePromptSwitchingTests(unittest.TestCase):
    def test_relation_ops_uses_selected_text2entity_prompt(self):
        pipeline = DummyRelationPipeline()
        recorder = {}
        with mock.patch.object(config_module, "PROMPT_LANGUAGE", "zh"):
            with mock.patch(
                "src.pipeline.relation_ops.ChatPromptTemplate.from_template",
                side_effect=make_template_factory('{"entity1": {"name": "Alice", "type": "Person", "description": "A person"}}', recorder),
            ):
                result = pipeline.extract_from_text_single("Alice founded Acme.", "/tmp/ignore.jsonl")

        self.assertEqual(recorder["template"], prompt_module.text2entity_zh)
        self.assertIn("entity1", result)

    def test_relation_ops_uses_selected_entity_centric_kg_prompt(self):
        pipeline = DummyRelationPipeline()
        recorder = {}
        entity_dic = {
            "entity1": {
                "name": "Alice",
                "type": "Person",
                "description": "Founder",
                "chunkid": ["c1"],
            }
        }
        with mock.patch.object(config_module, "PROMPT_LANGUAGE", "en"):
            with mock.patch(
                "src.pipeline.relation_ops.ChatPromptTemplate.from_template",
                side_effect=make_template_factory(
                    '{"central_entity": {"name": "Alice", "type": "Person", "description": "Founder", "attributes": [], "relationships": []}}',
                    recorder,
                ),
            ):
                result = pipeline.get_target_kg_single(
                    entity_dic=entity_dic,
                    entity_id="entity1",
                    id_to_sentence={"c1": "Alice founded Acme."},
                    sentences=["Alice founded Acme."],
                    sentence_to_id={"Alice founded Acme.": "c1"},
                    vectors=[[1.0, 0.0]],
                    output_file="/tmp/ignore.jsonl",
                    related_kg={"entity": "Acme"},
                )

        self.assertEqual(recorder["template"], prompt_module.extract_entiry_centric_kg_en_v2)
        self.assertIn("central_entity", result)

    def test_similarity_ops_uses_selected_entity_similarity_prompt(self):
        pipeline = DummySimilarityPipeline()
        recorder = {}
        with mock.patch.object(config_module, "PROMPT_LANGUAGE", "zh"):
            with mock.patch(
                "src.pipeline.similarity_ops.ChatPromptTemplate.from_template",
                side_effect=make_template_factory('{"result": true}', recorder),
            ):
                result = pipeline.similarity_llm_single(
                    {"name": "河南商报", "type": "媒体机构", "description": "媒体"},
                    {"name": "顶端新闻·河南商报", "type": "组织名", "description": "媒体"},
                )

        self.assertEqual(recorder["template"], prompt_module.judge_sim_entity_zh)
        self.assertTrue(result["result"])

    def test_qa_match_ops_uses_selected_question_entity_extract_prompt(self):
        pipeline = DummyQAMatchPipeline()
        recorder = {}
        with mock.patch.object(config_module, "PROMPT_LANGUAGE", "zh"):
            with mock.patch(
                "src.pipeline.qa_match_ops.ChatPromptTemplate.from_template",
                side_effect=make_template_factory('{"entities": ["爱因斯坦"], "keywords": ["相对论"]}', recorder),
            ):
                result = pipeline.extract_question_entities("爱因斯坦提出了什么理论？")

        self.assertEqual(recorder["template"], prompt_module.question_entity_extract_prompt_zh)
        self.assertEqual(result, ["爱因斯坦", "相对论"])

    def test_qa_answer_ops_uses_selected_qa_answer_prompt(self):
        pipeline = DummyQAAnswerPipeline()
        recorder = {}
        with mock.patch.object(config_module, "PROMPT_LANGUAGE", "en"):
            with mock.patch(
                "src.pipeline.qa_answer_ops.ChatPromptTemplate.from_template",
                side_effect=make_template_factory(
                    '{"answer": "Alice founded Acme.", "evidence_sources": [{"source": "chunk:c1", "quote": "Alice founded Acme."}], "graph_paths": ["Alice -> founded -> Acme"]}',
                    recorder,
                ),
            ):
                result = pipeline.answer_question_with_kg("Who founded Acme?")

        self.assertEqual(recorder["template"], prompt_module.kg_qa_answer_prompt_en)
        self.assertEqual(result["answer"], "Alice founded Acme.")


if __name__ == "__main__":
    unittest.main()
