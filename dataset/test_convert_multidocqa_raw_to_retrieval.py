import json
import tempfile
import unittest
from pathlib import Path

from dataset.convert_multidocqa_raw_to_retrieval import ConversionStats
from dataset.convert_multidocqa_raw_to_retrieval import convert_record
from dataset.convert_multidocqa_raw_to_retrieval import resolve_field_mapping
from dataset.convert_multidocqa_raw_to_retrieval import write_jsonl


class ConvertMultiDocQARawToRetrievalTests(unittest.TestCase):
    def test_resolve_field_mapping_uses_common_aliases(self):
        columns = ["sample_id", "question", "relevant_doc", "irrelevant_docs", "answer"]
        mapping = resolve_field_mapping(columns)
        self.assertEqual(
            mapping,
            {
                "id": "sample_id",
                "query": "question",
                "positive": "relevant_doc",
                "negatives": "irrelevant_docs",
                "answer": "answer",
            },
        )

    def test_convert_record_dedupes_and_truncates_negatives(self):
        record = {
            "sample_id": "42",
            "question": "什么是 RAKG？",
            "relevant_doc": "RAKG 是一种图谱构建框架。",
            "irrelevant_docs": [
                "文档A",
                "文档A",
                "RAKG 是一种图谱构建框架。",
                "文档B",
                "文档C",
            ],
            "answer": "一个用于知识图谱构建的方法。",
        }
        mapping = resolve_field_mapping(record.keys())
        stats = ConversionStats()

        sample = convert_record(
            record=record,
            mapping=mapping,
            split="train",
            row_index=7,
            max_negatives=2,
            dataset_name="yuyijiong/Multi-Doc-QA-Chinese",
            data_dir="raw",
            stats=stats,
        )

        self.assertEqual(sample["id"], "42")
        self.assertEqual(sample["positives"], ["RAKG 是一种图谱构建框架。"])
        self.assertEqual(sample["negatives"], ["文档A", "文档B"])
        self.assertEqual(sample["meta"]["negative_count"], 2)
        self.assertEqual(stats.written_rows, 1)
        self.assertEqual(stats.duplicate_negative_rows, 1)
        self.assertEqual(stats.positive_removed_from_negatives, 1)
        self.assertEqual(stats.truncated_negative_rows, 1)

    def test_convert_record_filters_invalid_samples(self):
        mapping = resolve_field_mapping(
            ["question", "relevant_doc", "irrelevant_docs", "answer"]
        )

        stats_missing_positive = ConversionStats()
        sample = convert_record(
            record={
                "question": "问题",
                "relevant_doc": "   ",
                "irrelevant_docs": ["负例"],
                "answer": "答案",
            },
            mapping=mapping,
            split="train",
            row_index=1,
            max_negatives=20,
            dataset_name="yuyijiong/Multi-Doc-QA-Chinese",
            data_dir="raw",
            stats=stats_missing_positive,
        )
        self.assertIsNone(sample)
        self.assertEqual(stats_missing_positive.filtered_empty_positive, 1)

        stats_missing_negative = ConversionStats()
        sample = convert_record(
            record={
                "question": "问题",
                "relevant_doc": "正例",
                "irrelevant_docs": ["正例"],
                "answer": "答案",
            },
            mapping=mapping,
            split="train",
            row_index=2,
            max_negatives=20,
            dataset_name="yuyijiong/Multi-Doc-QA-Chinese",
            data_dir="raw",
            stats=stats_missing_negative,
        )
        self.assertIsNone(sample)
        self.assertEqual(stats_missing_negative.filtered_empty_negative, 1)

    def test_write_jsonl_persists_rows(self):
        rows = [
            {
                "id": "train-000001",
                "query": "问题",
                "positives": ["正例"],
                "negatives": ["负例1", "负例2"],
                "answer": "答案",
                "meta": {"source_dataset": "x", "subset": "raw", "split": "train", "negative_count": 2},
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sample.jsonl"
            count = write_jsonl(output_path, rows)
            self.assertEqual(count, 1)
            loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(loaded, rows)


if __name__ == "__main__":
    unittest.main()
