import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from src.kgAgent import NER_Agent
from src.web.backend.app.schemas import KGBuildRequest

_BACKEND_ROOT = Path(__file__).resolve().parents[1] / "web" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.kg_service import KGBuildService  # noqa: E402


class _DummyLLMProvider:
    def get_llm(self):
        return object()

    def get_similarity_model(self):
        return object()

    def get_embedding_model(self):
        return object()


class DisambiguationConfigInterfaceTests(unittest.TestCase):
    def test_kg_build_request_disambiguation_config_validation(self):
        req = KGBuildRequest(
            input_type="text",
            text="hello",
            disambiguation_config={
                "similarity_threshold": 0.55,
                "per_entity_top_k": 5,
                "description_max_chars": 120,
            },
        )
        self.assertIsNotNone(req.disambiguation_config)
        self.assertEqual(req.disambiguation_config.per_entity_top_k, 5)
        self.assertAlmostEqual(req.disambiguation_config.similarity_threshold, 0.55)

        with self.assertRaises(ValidationError):
            KGBuildRequest(
                input_type="text",
                text="hello",
                disambiguation_config={"similarity_threshold": 1.2},
            )

    def test_process_all_topics_passes_resolved_disambiguation_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "topics.json"
            output_dir = tmp_path / "out"
            input_path.write_text(
                json.dumps([{"topic": "t1", "content": "sample"}], ensure_ascii=False),
                encoding="utf-8",
            )

            with mock.patch("src.kgAgent.LLMProvider", return_value=_DummyLLMProvider()):
                agent = NER_Agent()

            captured = {}

            def fake_process(**kwargs):
                captured["disambiguation_config"] = kwargs.get("disambiguation_config")
                return {
                    "index": 1,
                    "topic": "t1",
                    "output_path": str(output_dir / "RAKG_graph_re" / "1.json"),
                    "knowledge_graph": {},
                    "current_doc_kg": {},
                    "alias_resolution": {},
                }

            agent.process = fake_process  # type: ignore[method-assign]

            custom_config = {
                "similarity_threshold": 0.72,
                "per_entity_top_k": 3,
                "description_max_chars": 90,
            }
            agent.process_all_topics(
                json_path=str(input_path),
                output_dir=str(output_dir),
                disambiguation_config=custom_config,
            )

            passed = captured.get("disambiguation_config")
            self.assertIsInstance(passed, dict)
            self.assertEqual(passed["similarity_threshold"], 0.72)
            self.assertEqual(passed["per_entity_top_k"], 3)
            self.assertEqual(passed["description_max_chars"], 90)
            self.assertTrue(passed["type_gate_enabled"])
            self.assertTrue(passed["direct_merge_enabled"])

    def test_kg_service_run_forwards_disambiguation_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = tmp_path / "kg_out"
            captured = {}

            class DummyAgent:
                def process_all_topics(self, **kwargs):
                    captured["kwargs"] = kwargs
                    return {
                        "total_topics": 1,
                        "processed_topics": 1,
                        "failed_topics_count": 0,
                        "failed_topics": [],
                        "graph_paths": [str(output_dir / "RAKG_graph_re" / "1.json")],
                    }

            payload = {
                "input_type": "text",
                "text": "hello world",
                "topic": "web_topic",
                "output_dir": str(output_dir),
                "force_rebuild": False,
                "disambiguation_config": {
                    "similarity_threshold": 0.66,
                    "per_entity_top_k": 4,
                },
            }

            with mock.patch("app.services.kg_service.NER_Agent", return_value=DummyAgent()):
                service = KGBuildService()
                service.run(
                    task_id="task_test",
                    payload=payload,
                    on_progress=lambda *_args, **_kwargs: None,
                    on_log=lambda *_args, **_kwargs: None,
                    is_cancel_requested=lambda: False,
                )

            passed = captured["kwargs"]["disambiguation_config"]
            self.assertEqual(passed, payload["disambiguation_config"])


if __name__ == "__main__":
    unittest.main()
