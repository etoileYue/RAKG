import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dataset import generate_multifieldqa_zh_qa as genqa


class DummyResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return DummyResponse(
            json.dumps(
                {
                    "qa_pairs": [
                        {"question": "谁是作者？", "answer": "张三", "evidence": "作者是张三"},
                        {"question": "外部信息？", "answer": "李四", "evidence": "原文没有这句"},
                        {"question": "", "answer": "北京", "evidence": "地点在北京"},
                        {"question": "谁是作者？", "answer": "张三", "evidence": "作者是张三"},
                    ]
                },
                ensure_ascii=False,
            )
        )


class FakeProvider:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


class MultiFieldQAZHGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "test.jsonl"
        row = {
            "_id": "s1",
            "input": "原始问题？",
            "context": "作者是张三。地点在北京。",
            "answers": ["张三"],
            "length": 12,
            "dataset": "multifieldqa_zh",
            "language": "zh",
        }
        self.input_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        self.output_path = self.root / "expanded_qa.jsonl"
        self.summary_path = self.root / "summary.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_response_keeps_unmatched_answer_and_evidence(self):
        pairs = genqa.parse_qa_generation_response(FakeModel().invoke("prompt"))
        self.assertEqual(len(pairs), 4)
        valid_pairs = [
            pair
            for pair in pairs
            if genqa.validate_generated_pair(pair, "作者是张三。地点在北京。") is not None
        ]
        self.assertEqual(len(valid_pairs), 3)
        self.assertIn({"question": "外部信息？", "answer": "李四", "evidence": "原文没有这句"}, valid_pairs)

    def test_main_merges_original_and_does_not_duplicate_on_rerun(self):
        model = FakeModel()
        with mock.patch.object(genqa, "LLMProvider", return_value=FakeProvider(model)):
            summary = genqa.main(
                [
                    "--input-path",
                    str(self.input_path),
                    "--output-path",
                    str(self.output_path),
                    "--summary-path",
                    str(self.summary_path),
                    "--qa-count",
                    "2",
                    "--rate-limit-initial-wait",
                    "0",
                ]
            )
            second_summary = genqa.main(
                [
                    "--input-path",
                    str(self.input_path),
                    "--output-path",
                    str(self.output_path),
                    "--summary-path",
                    str(self.summary_path),
                    "--qa-count",
                    "2",
                    "--rate-limit-initial-wait",
                    "0",
                ]
            )

        self.assertEqual(summary["generated_count"], 2)
        self.assertEqual(summary["unmatched_answer_count"], 1)
        self.assertEqual(summary["unmatched_evidence_count"], 1)
        self.assertEqual(summary["invalid_count"], 0)
        self.assertEqual(second_summary["generated_count"], 0)
        records = genqa.read_jsonl(self.output_path)
        self.assertEqual(len(records), 1)
        group = records[0]
        self.assertEqual(group["source_sample_index"], 0)
        self.assertNotIn("source_sample_id", group)
        self.assertEqual([pair["qa_id"] for pair in group["qa_pairs"]], ["0", "1", "2"])
        original = group["qa_pairs"][0]
        self.assertEqual(original["qa_source"], "original")
        self.assertEqual(original["answers"], ["张三"])
        self.assertEqual(group["qa_pairs"][1]["qa_source"], "generated")
        self.assertEqual(group["qa_pairs"][2]["answers"], ["李四"])
        self.assertEqual(group["qa_pairs"][2]["evidence"], "原文没有这句")
        self.assertEqual(len(model.prompts), 1)

    def test_rerun_converts_legacy_flat_records_to_grouped_jsonl(self):
        genqa.write_jsonl(
            self.output_path,
            [
                {
                    "qa_id": "s1#gen001",
                    "source_sample_id": "s1",
                    "source_sample_index": 0,
                    "question": "旧生成问题1",
                    "answers": ["北京"],
                    "qa_source": "generated",
                    "evidence": "地点在北京",
                },
                {
                    "qa_id": "s1#gen002",
                    "source_sample_id": "s1",
                    "source_sample_index": 0,
                    "question": "旧生成问题2",
                    "answers": ["张三"],
                    "qa_source": "generated",
                    "evidence": "作者是张三",
                },
            ],
        )

        model = FakeModel()
        with mock.patch.object(genqa, "LLMProvider", return_value=FakeProvider(model)):
            summary = genqa.main(
                [
                    "--input-path",
                    str(self.input_path),
                    "--output-path",
                    str(self.output_path),
                    "--summary-path",
                    str(self.summary_path),
                    "--qa-count",
                    "2",
                    "--rate-limit-initial-wait",
                    "0",
                ]
            )

        self.assertEqual(summary["generated_count"], 0)
        self.assertEqual(len(model.prompts), 0)
        records = genqa.read_jsonl(self.output_path)
        self.assertEqual(len(records), 1)
        self.assertNotIn("source_sample_id", records[0])
        self.assertEqual([pair["qa_id"] for pair in records[0]["qa_pairs"]], ["0", "1", "2"])
        self.assertEqual(records[0]["qa_pairs"][0]["qa_source"], "original")


if __name__ == "__main__":
    unittest.main()
