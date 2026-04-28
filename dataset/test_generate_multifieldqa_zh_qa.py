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
                        {"question": "地点在哪里？", "answer": "北京", "evidence": "地点在北京"},
                        {"question": "无效证据？", "answer": "张三", "evidence": "原文没有这句"},
                        {"question": "无效答案？", "answer": "上海", "evidence": "地点在北京"},
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

    def test_parse_response_and_filter_invalid_evidence(self):
        pairs = genqa.parse_qa_generation_response(FakeModel().invoke("prompt"))
        self.assertEqual(len(pairs), 4)
        valid_pairs = [
            pair
            for pair in pairs
            if genqa.validate_generated_pair(pair, "作者是张三。地点在北京。") is not None
        ]
        self.assertEqual(len(valid_pairs), 2)

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
        self.assertEqual(second_summary["generated_count"], 0)
        records = genqa.read_jsonl(self.output_path)
        self.assertEqual(len(records), 3)
        self.assertEqual([record["qa_id"] for record in records], ["s1#gen001", "s1#gen002", "s1#original"])
        original = [record for record in records if record["qa_source"] == "original"][0]
        self.assertEqual(original["answers"], ["张三"])
        self.assertEqual(len(model.prompts), 1)


if __name__ == "__main__":
    unittest.main()
