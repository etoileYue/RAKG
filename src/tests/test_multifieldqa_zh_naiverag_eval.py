import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.eval.multifieldqa_zh import evaluate_multifieldqa_zh as mfq
from src.eval.multifieldqa_zh import evaluate_multifieldqa_zh_naiverag as naive_mfq


class FakeNaiveRAGAgent:
    build_fail_ids = set()
    answer_fail_stems = set()
    process_calls = []
    answer_calls = []

    def process(self, topic_data, idx, total_topics, rag_output_dir):
        self.process_calls.append(
            {
                "topic": topic_data["topic"],
                "idx": idx,
                "total_topics": total_topics,
                "rag_output_dir": rag_output_dir,
            }
        )
        if topic_data["topic"] in self.build_fail_ids:
            raise RuntimeError("build failed")

        output_path = Path(rag_output_dir) / f"{idx}.json"
        output_path.write_text(
            json.dumps(
                {
                    "index": idx,
                    "topic": topic_data["topic"],
                    "sentences": [topic_data["content"]],
                    "vectors": [[1.0]],
                    "sentence_to_id": {topic_data["content"]: "s0"},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"output_path": str(output_path)}

    def answer_question(self, question, index_input, top_k=5):
        self.answer_calls.append({"question": question, "index_input": index_input, "top_k": top_k})
        sample_name = Path(index_input).stem
        if sample_name in self.answer_fail_stems:
            raise RuntimeError("answer failed")
        return {
            "answer": f"预测答案-{sample_name}",
            "formatted_answer": f"答案：预测答案-{sample_name}",
            "retrieval": {
                "top_k": top_k,
                "items": [
                    {
                        "rank": 1,
                        "sentence": f"证据-{sample_name}",
                        "sentence_id": "s0",
                        "score": 0.9,
                    }
                ],
                "context_text": f"证据-{sample_name}",
            },
            "evidence_sources": [{"source": "s0", "quote": f"证据-{sample_name}"}],
            "llm_output_raw": '{"answer":"ok"}',
        }


class NaiveRAGMultiFieldQAZHEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        FakeNaiveRAGAgent.build_fail_ids = set()
        FakeNaiveRAGAgent.answer_fail_stems = set()
        FakeNaiveRAGAgent.process_calls = []
        FakeNaiveRAGAgent.answer_calls = []

        self.dataset_path = self.root / "test.jsonl"
        rows = [
            {
                "_id": "s1",
                "input": "问题1",
                "context": "上下文1",
                "answers": ["预测答案-0", "其他答案"],
                "length": 100,
                "dataset": "multifieldqa_zh",
                "language": "zh",
            },
            {
                "_id": "s2",
                "input": "问题2",
                "context": "上下文2",
                "answers": ["错误答案"],
                "length": 200,
                "dataset": "multifieldqa_zh",
                "language": "zh",
            },
        ]
        with self.dataset_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_stage_writes_indexes_manifest_and_summary(self):
        output_root = self.root / "out_build"
        with mock.patch.object(naive_mfq, "NaiveRAGAgent", FakeNaiveRAGAgent):
            summary = naive_mfq.main(
                [
                    "build",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                ]
            )

        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["error_count"], 0)
        self.assertTrue((output_root / "index" / "0.json").exists())
        self.assertTrue((output_root / "summary" / "build_summary.json").exists())

        manifest_records = naive_mfq.load_jsonl(output_root / "summary" / "build_manifest.jsonl")
        self.assertEqual(len(manifest_records), 2)
        self.assertEqual(manifest_records[0]["sample_id"], "s1")
        self.assertEqual(manifest_records[0]["status"], "success")
        self.assertTrue(manifest_records[0]["index_path"].endswith("/index/0.json"))
        self.assertEqual(FakeNaiveRAGAgent.process_calls[0]["idx"], 0)

    def test_answer_stage_writes_predictions_and_skips_completed(self):
        output_root = self.root / "out_answer"
        with mock.patch.object(naive_mfq, "NaiveRAGAgent", FakeNaiveRAGAgent):
            naive_mfq.main(
                [
                    "build",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                    "--limit",
                    "1",
                ]
            )
            summary = naive_mfq.main(
                [
                    "answer",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                    "--limit",
                    "1",
                    "--top-k",
                    "3",
                ]
            )
            second_summary = naive_mfq.main(
                [
                    "answer",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                    "--limit",
                    "1",
                    "--top-k",
                    "3",
                ]
            )

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(second_summary["skipped_count"], 1)
        predictions = naive_mfq.load_jsonl(output_root / "result" / "predictions.jsonl")
        self.assertEqual(len(predictions), 1)
        record = predictions[0]
        self.assertEqual(record["sample_id"], "s1")
        self.assertEqual(record["pred_answer"], "预测答案-0")
        self.assertEqual(record["retrieval"]["top_k"], 3)
        self.assertIn("items", record["retrieval"])
        self.assertEqual(record["evidence_sources"][0]["source"], "s0")
        self.assertEqual(len(FakeNaiveRAGAgent.answer_calls), 1)

    def test_expanded_qa_reuses_same_index_and_skips_by_qa_id(self):
        output_root = self.root / "out_expanded"
        qa_dataset_path = self.root / "expanded_qa.jsonl"
        naive_mfq.write_jsonl(
            qa_dataset_path,
            [
                {
                    "qa_id": "s1#gen001",
                    "source_sample_id": "s1",
                    "source_sample_index": 0,
                    "question": "生成问题1",
                    "answers": ["预测答案-0"],
                    "qa_source": "generated",
                    "evidence": "上下文1",
                },
                {
                    "qa_id": "s1#gen002",
                    "source_sample_id": "s1",
                    "source_sample_index": 0,
                    "question": "生成问题2",
                    "answers": ["预测答案-0"],
                    "qa_source": "generated",
                    "evidence": "上下文1",
                },
            ],
        )

        with mock.patch.object(naive_mfq, "NaiveRAGAgent", FakeNaiveRAGAgent):
            naive_mfq.main(
                [
                    "build",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                    "--limit",
                    "1",
                ]
            )
            summary = naive_mfq.main(
                [
                    "answer",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--qa-dataset-path",
                    str(qa_dataset_path),
                    "--output-root",
                    str(output_root),
                ]
            )
            second_summary = naive_mfq.main(
                [
                    "answer",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--qa-dataset-path",
                    str(qa_dataset_path),
                    "--output-root",
                    str(output_root),
                ]
            )

        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(second_summary["skipped_count"], 2)
        self.assertEqual(len(FakeNaiveRAGAgent.answer_calls), 2)
        self.assertEqual({Path(call["index_input"]).name for call in FakeNaiveRAGAgent.answer_calls}, {"0.json"})
        predictions = naive_mfq.load_jsonl(output_root / "result" / "predictions.jsonl")
        self.assertEqual([record["qa_id"] for record in predictions], ["s1#gen001", "s1#gen002"])
        self.assertEqual({record["source_sample_id"] for record in predictions}, {"s1"})

    def test_score_stage_uses_max_reference_f1_and_can_skip_llm_judge(self):
        output_root = self.root / "out_score"
        output_root.mkdir(parents=True, exist_ok=True)
        naive_mfq.write_jsonl(
            output_root / "result" / "predictions.jsonl",
            [
                {
                    "sample_id": "s1",
                    "sample_index": 0,
                    "question": "问题1",
                    "answers": ["预测答案-0", "其他答案"],
                    "index_path": "/tmp/0.json",
                    "pred_answer": "预测答案-0",
                    "formatted_answer": "答案：预测答案-0",
                    "retrieval": {"context_text": "证据-0", "items": []},
                    "evidence_sources": [],
                    "llm_output_raw": "",
                    "status": "success",
                },
                {
                    "sample_id": "s2",
                    "sample_index": 1,
                    "question": "问题2",
                    "answers": ["错误答案"],
                    "index_path": "/tmp/1.json",
                    "pred_answer": "完全不同",
                    "formatted_answer": "答案：完全不同",
                    "retrieval": {"context_text": "", "items": []},
                    "evidence_sources": [],
                    "llm_output_raw": "",
                    "status": "success",
                },
            ],
        )

        fake_jieba = mock.Mock()
        fake_jieba.cut.side_effect = lambda text, cut_all=False: list(text)

        with mock.patch.object(mfq, "_get_jieba", return_value=fake_jieba):
            summary = naive_mfq.main(
                [
                    "score",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                    "--skip-llm-judge",
                ]
            )

        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["avg_f1"], 0.5)
        self.assertEqual(summary["answer_judge_accuracy"], None)
        self.assertEqual(summary["retrieval_judge_accuracy"], None)

        scored = naive_mfq.load_jsonl(output_root / "result" / "scored_results.jsonl")
        scored_by_id = naive_mfq.index_records_by_sample_id(scored)
        self.assertEqual(scored_by_id["s1"]["official_f1"], 1.0)
        self.assertNotIn("answer_judge", scored_by_id["s1"])
        self.assertEqual(scored_by_id["s1"]["index_path"], "/tmp/0.json")

    def test_all_stage_runs_build_answer_and_score(self):
        output_root = self.root / "out_all"
        fake_jieba = mock.Mock()
        fake_jieba.cut.side_effect = lambda text, cut_all=False: list(text)

        with mock.patch.object(naive_mfq, "NaiveRAGAgent", FakeNaiveRAGAgent):
            with mock.patch.object(mfq, "_get_jieba", return_value=fake_jieba):
                summary = naive_mfq.main(
                    [
                        "all",
                        "--dataset-path",
                        str(self.dataset_path),
                        "--output-root",
                        str(output_root),
                        "--limit",
                        "1",
                        "--skip-llm-judge",
                    ]
                )

        self.assertEqual(summary["phase"], "all")
        self.assertEqual(summary["build"]["success_count"], 1)
        self.assertEqual(summary["answer"]["success_count"], 1)
        self.assertEqual(summary["score"]["success_count"], 1)
        self.assertTrue((output_root / "summary" / "build_summary.json").exists())
        self.assertTrue((output_root / "summary" / "answer_summary.json").exists())
        self.assertTrue((output_root / "summary" / "score_summary.json").exists())
        self.assertTrue((output_root / "result" / "scored_results.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
