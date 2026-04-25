import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from src.eval.multifieldqa_zh import evaluate_multifieldqa_zh as mfq


class DummyResponse:
    def __init__(self, content):
        self.content = content


class FakeJudgeModel:
    def __init__(self, answer_result=1, retrieval_result=1):
        self.answer_result = answer_result
        self.retrieval_result = retrieval_result
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if "检索上下文" in prompt:
            return DummyResponse(json.dumps({"result": self.retrieval_result}, ensure_ascii=False))
        if "完全不同" in prompt:
            return DummyResponse(json.dumps({"result": 0}, ensure_ascii=False))
        return DummyResponse(json.dumps({"result": self.answer_result}, ensure_ascii=False))


class FakeProvider:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


class FakeAgent:
    CHECKPOINT_FILE_NAME = "checkpoint_state.json"
    build_fail_ids = set()
    answer_fail_ids = set()
    prepare_calls = []
    process_calls = []

    def __init__(self):
        self.current_graph_path = None

    def _prepare_checkpoint_state(
        self,
        *,
        output_dir,
        topics,
        topic_indices=None,
        ner_output_dir,
        rel_output_dir,
        sim_output_dir,
        graph_output_dir,
        force_rebuild=False,
    ):
        checkpoint_path = Path(output_dir) / self.CHECKPOINT_FILE_NAME
        checkpoint_loaded = checkpoint_path.exists() and not force_rebuild
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_loaded:
            checkpoint_state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        else:
            checkpoint_state = {"version": 1, "input_fingerprint": "fake", "topics": {}}

        resolved_indices = topic_indices or list(range(1, len(topics) + 1))
        for idx, topic in zip(resolved_indices, topics):
            checkpoint_state["topics"].setdefault(
                str(idx),
                {
                    "topic_name": topic["topic"],
                    "stages": {"ner": "pending", "sim": "pending", "rel": "pending"},
                },
            )
        checkpoint_path.write_text(
            json.dumps(checkpoint_state, ensure_ascii=False),
            encoding="utf-8",
        )
        self.prepare_calls.append(
            {
                "output_dir": output_dir,
                "topic_indices": list(resolved_indices),
                "force_rebuild": force_rebuild,
                "checkpoint_loaded": checkpoint_loaded,
            }
        )
        return checkpoint_state, str(checkpoint_path), checkpoint_loaded

    def process(
        self,
        topic_data,
        idx,
        total_topics,
        ner_output_dir,
        rel_output_dir,
        sim_output_dir,
        graph_output_dir,
        **kwargs,
    ):
        self.process_calls.append(
            {"topic": topic_data["topic"], "idx": idx, "kwargs": kwargs}
        )
        if topic_data["topic"] in self.build_fail_ids:
            raise RuntimeError("build failed")
        graph_path = Path(graph_output_dir) / f"{idx}.json"
        graph_path.write_text(
            json.dumps(
                {
                    "entities": [{"name": topic_data["topic"], "type": "Doc", "description": "ctx"}],
                    "relations": [],
                    "chunk_map": {"c1": topic_data["content"]},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"output_path": str(graph_path)}

    def initialize_qa_graph_index(self, knowledge_graph, cache_key=None, force_rebuild=False):
        self.current_graph_path = knowledge_graph
        return cache_key

    def answer_question_with_kg(self, question, cache_key=None, **kwargs):
        graph_path = str(self.current_graph_path)
        if "fail-answer" in graph_path:
            raise RuntimeError("answer failed")
        sample_name = Path(graph_path).stem
        return {
            "answer": f"预测答案-{sample_name}",
            "formatted_answer": f"答案：预测答案-{sample_name}",
            "graph_paths": [f"path-{sample_name}"],
            "retrieval": {
                "context_text": f"证据-{sample_name}",
                "evidence_items": [{"source": "chunk:c1", "text": f"证据-{sample_name}"}],
                "graph_paths": [f"path-{sample_name}"],
                "matched_nodes": [{"name": "n1"}],
                "seed_nodes": ["n1"],
            },
            "llm_output_raw": '{"answer":"ok"}',
        }

    def clear_qa_graph_index(self, cache_key=None):
        self.current_graph_path = None


class MultiFieldQAZHEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        FakeAgent.build_fail_ids = set()
        FakeAgent.answer_fail_ids = set()
        FakeAgent.prepare_calls = []
        FakeAgent.process_calls = []
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

    def test_resolve_output_paths_groups_summary_and_result_files(self):
        output_root = self.root / "out_paths"
        paths = mfq.resolve_output_paths(output_root)

        self.assertEqual(paths["build_manifest_path"], output_root / "summary" / "build_manifest.jsonl")
        self.assertEqual(paths["build_summary_path"], output_root / "summary" / "build_summary.json")
        self.assertEqual(paths["answer_summary_path"], output_root / "summary" / "answer_summary.json")
        self.assertEqual(paths["score_summary_path"], output_root / "summary" / "score_summary.json")
        self.assertEqual(paths["predictions_path"], output_root / "result" / "predictions.jsonl")
        self.assertEqual(paths["scored_results_path"], output_root / "result" / "scored_results.jsonl")

    def test_build_stage_records_success_and_error_and_skips_completed(self):
        output_root = self.root / "out"
        FakeAgent.build_fail_ids = {"s2"}

        with mock.patch.object(mfq, "NER_Agent", FakeAgent):
            summary = mfq.main(
                [
                    "build",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["error_count"], 1)

            manifest_records = mfq.load_jsonl(output_root / "summary" / "build_manifest.jsonl")
            record_by_id = mfq.index_records_by_sample_id(manifest_records)
            self.assertEqual(record_by_id["s1"]["status"], "success")
            self.assertEqual(record_by_id["s2"]["status"], "error")
            self.assertTrue((output_root / "build_cache" / "checkpoint_state.json").exists())
            self.assertEqual(FakeAgent.prepare_calls[0]["topic_indices"], [0, 1])
            self.assertEqual(
                FakeAgent.process_calls[0]["kwargs"]["checkpoint_path"],
                str(output_root / "build_cache" / "checkpoint_state.json"),
            )
            self.assertFalse(FakeAgent.process_calls[0]["kwargs"]["auto_resume"])

            summary_second = mfq.main(
                [
                    "build",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                ]
            )
            self.assertEqual(summary_second["skipped_count"], 1)
            self.assertEqual(summary_second["error_count"], 1)
            self.assertTrue(summary_second["checkpoint_loaded"])
            self.assertTrue(summary_second["auto_resume"])

    def test_build_stage_backfills_manifest_from_existing_graph(self):
        output_root = self.root / "out_existing_graph"
        graph_dir = output_root / "graphs"
        graph_dir.mkdir(parents=True)
        (graph_dir / "0.json").write_text(
            json.dumps({"entities": [], "relations": []}, ensure_ascii=False),
            encoding="utf-8",
        )

        with mock.patch.object(mfq, "NER_Agent", FakeAgent):
            summary = mfq.main(
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

        self.assertEqual(summary["backfilled_manifest_count"], 1)
        self.assertEqual(summary["skipped_count"], 1)
        self.assertEqual(FakeAgent.process_calls, [])
        manifest_records = mfq.load_jsonl(output_root / "summary" / "build_manifest.jsonl")
        self.assertEqual(manifest_records[0]["sample_id"], "s1")
        self.assertTrue(manifest_records[0]["recovered_from_existing_graph"])

    def test_build_stage_auto_resumes_when_legacy_stage_cache_exists(self):
        output_root = self.root / "out_legacy_stage_cache"
        ner_dir = output_root / "build_cache" / "ner_data"
        ner_dir.mkdir(parents=True)
        (ner_dir / "output_text_ner_0.jsonl").write_text(
            json.dumps({"text": "上下文1", "entities": {}}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        with mock.patch.object(mfq, "NER_Agent", FakeAgent):
            summary = mfq.main(
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

        self.assertFalse(summary["checkpoint_loaded"])
        self.assertTrue(summary["auto_resume"])
        self.assertEqual(summary["existing_stage_cache_count"], 1)
        self.assertEqual(summary["existing_stage_caches"][0]["stages"], ["ner"])
        self.assertTrue(FakeAgent.process_calls[0]["kwargs"]["auto_resume"])

    def test_answer_stage_writes_prediction_fields(self):
        output_root = self.root / "out_answer"
        with mock.patch.object(mfq, "NER_Agent", FakeAgent):
            mfq.main(
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
            summary = mfq.main(
                [
                    "answer",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                    "--limit",
                    "1",
                ]
            )

        self.assertEqual(summary["success_count"], 1)
        predictions = mfq.load_jsonl(output_root / "result" / "predictions.jsonl")
        self.assertEqual(len(predictions), 1)
        record = predictions[0]
        self.assertEqual(record["sample_id"], "s1")
        self.assertIn("pred_answer", record)
        self.assertIn("formatted_answer", record)
        self.assertIn("retrieval", record)
        self.assertIn("context_text", record["retrieval"])
        self.assertEqual(record["status"], "success")

    def test_score_stage_uses_max_reference_f1_and_optional_judges(self):
        output_root = self.root / "out_score"
        ensure = output_root.mkdir(parents=True, exist_ok=True)
        self.assertIsNone(ensure)

        predictions = [
            {
                "sample_id": "s1",
                "sample_index": 0,
                "question": "问题1",
                "answers": ["预测答案-0", "其他答案"],
                "pred_answer": "预测答案-0",
                "formatted_answer": "答案：预测答案-0",
                "graph_path": "/tmp/0.json",
                "graph_paths": ["path-0"],
                "retrieval": {"context_text": "证据-0", "evidence_items": [{"source": "chunk:c1", "text": "证据-0"}]},
                "status": "success",
            },
            {
                "sample_id": "s2",
                "sample_index": 1,
                "question": "问题2",
                "answers": ["错误答案"],
                "pred_answer": "完全不同",
                "formatted_answer": "答案：完全不同",
                "graph_path": "/tmp/1.json",
                "graph_paths": ["path-1"],
                "retrieval": {"context_text": "", "evidence_items": []},
                "status": "success",
            },
        ]
        mfq.write_jsonl(output_root / "result" / "predictions.jsonl", predictions)

        fake_jieba = mock.Mock()
        fake_jieba.cut.side_effect = lambda text, cut_all=False: list(text)
        judge_model = FakeJudgeModel(answer_result=1, retrieval_result=0)

        with mock.patch.object(mfq, "_get_jieba", return_value=fake_jieba):
            with mock.patch.object(mfq, "LLMProvider", return_value=FakeProvider(judge_model)):
                summary = mfq.main(
                    [
                        "score",
                        "--dataset-path",
                        str(self.dataset_path),
                        "--output-root",
                        str(output_root),
                    ]
                )

        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["avg_f1"], 0.5)
        self.assertEqual(summary["answer_judge_accuracy"], 0.5)
        self.assertEqual(summary["retrieval_judge_accuracy"], 0.0)

        scored = mfq.load_jsonl(output_root / "result" / "scored_results.jsonl")
        scored_by_id = mfq.index_records_by_sample_id(scored)
        self.assertEqual(scored_by_id["s1"]["official_f1"], 1.0)
        self.assertEqual(scored_by_id["s1"]["answer_judge"], 1)
        self.assertEqual(scored_by_id["s1"]["retrieval_judge"], 0)
        self.assertEqual(scored_by_id["s2"]["retrieval_judge"], 0)

    def test_score_stage_can_skip_llm_judges(self):
        output_root = self.root / "out_score_skip"
        output_root.mkdir(parents=True, exist_ok=True)
        mfq.write_jsonl(
            output_root / "result" / "predictions.jsonl",
            [
                {
                    "sample_id": "s1",
                    "sample_index": 0,
                    "question": "问题1",
                    "answers": ["预测答案-0"],
                    "pred_answer": "预测答案-0",
                    "formatted_answer": "答案：预测答案-0",
                    "graph_path": "/tmp/0.json",
                    "graph_paths": ["path-0"],
                    "retrieval": {"context_text": "证据-0", "evidence_items": []},
                    "status": "success",
                }
            ],
        )

        fake_jieba = mock.Mock()
        fake_jieba.cut.side_effect = lambda text, cut_all=False: list(text)

        with mock.patch.object(mfq, "_get_jieba", return_value=fake_jieba):
            summary = mfq.main(
                [
                    "score",
                    "--dataset-path",
                    str(self.dataset_path),
                    "--output-root",
                    str(output_root),
                    "--skip-llm-judge",
                ]
            )

        self.assertEqual(summary["answer_judge_accuracy"], None)
        self.assertEqual(summary["retrieval_judge_accuracy"], None)
        scored = mfq.load_jsonl(output_root / "result" / "scored_results.jsonl")
        self.assertNotIn("answer_judge", scored[0])

    def test_cli_prints_summary_json(self):
        output_root = self.root / "out_cli"
        with mock.patch.object(mfq, "NER_Agent", FakeAgent):
            with mock.patch("sys.stdout", new_callable=StringIO) as stdout:
                exit_code = mfq.cli(
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

        self.assertEqual(exit_code, 0)
        printed = json.loads(stdout.getvalue())
        self.assertEqual(printed["phase"], "build")
        self.assertEqual(printed["success_count"], 1)


if __name__ == "__main__":
    unittest.main()
