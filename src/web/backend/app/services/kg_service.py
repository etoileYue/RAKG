from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.kgAgent import NER_Agent

from app.config import DEFAULT_TASK_OUTPUT_ROOT, REPO_ROOT


class TaskCanceledError(RuntimeError):
    pass


@dataclass
class KGBuildResult:
    output_payload: dict[str, Any]


class KGBuildService:
    """Wraps existing NER_Agent pipeline and exposes progress callbacks."""

    def __init__(self) -> None:
        DEFAULT_TASK_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        task_id: str,
        payload: dict[str, Any],
        on_progress: Callable[[float, str], None],
        on_log: Callable[[str, str], None],
        is_cancel_requested: Callable[[], bool],
    ) -> KGBuildResult:
        output_dir = self._resolve_output_dir(task_id, payload.get("output_dir"))
        output_dir.mkdir(parents=True, exist_ok=True)

        input_json_path = self._prepare_input_json(output_dir, payload)
        topics = self._load_topics(input_json_path)
        total_topics = len(topics)

        ner_output_dir = output_dir / "ner_data"
        rel_output_dir = output_dir / "rel_data"
        sim_output_dir = output_dir / "sim_data"
        graph_output_dir = output_dir / "RAKG_graph_re"
        for directory in [ner_output_dir, rel_output_dir, sim_output_dir, graph_output_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        agent = NER_Agent()

        processed_topics = 0
        failed_topics: list[dict[str, Any]] = []
        produced_graph_paths: list[str] = []

        for idx, topic_data in enumerate(topics, start=1):
            if is_cancel_requested():
                raise TaskCanceledError("Task canceled by user")

            topic_name = topic_data.get("topic", f"topic_{idx}")
            on_progress((idx - 1) / max(total_topics, 1), f"processing {idx}/{total_topics}: {topic_name}")
            on_log("INFO", f"Start processing topic {idx}/{total_topics}: {topic_name}")

            try:
                result = agent.process(
                    topic_data=topic_data,
                    idx=idx,
                    total_topics=total_topics,
                    ner_output_dir=str(ner_output_dir),
                    rel_output_dir=str(rel_output_dir),
                    sim_output_dir=str(sim_output_dir),
                    graph_output_dir=str(graph_output_dir),
                    skip_ner=False,
                    skip_sim=False,
                    skip_rel=False,
                    existing_kg=None,
                )
                processed_topics += 1
                produced_graph_paths.append(result["output_path"])
                on_log("INFO", f"Topic {idx} finished. graph={result['output_path']}")
            except Exception as exc:  # noqa: BLE001
                failed_topics.append(
                    {
                        "index": idx,
                        "topic": topic_name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                on_log("ERROR", f"Topic {idx} failed: {exc}")

        summary = {
            "total_topics": total_topics,
            "processed_topics": processed_topics,
            "failed_topics_count": len(failed_topics),
            "failed_topics": failed_topics,
        }

        summary_path = output_dir / "process_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        # V1 提供图谱目录与最后一个图谱路径，便于前端直接加载。
        output_payload = {
            "output_dir": str(output_dir),
            "source_json_path": str(input_json_path),
            "summary_path": str(summary_path),
            "graph_dir": str(graph_output_dir),
            "graph_paths": produced_graph_paths,
            "latest_graph_path": produced_graph_paths[-1] if produced_graph_paths else "",
            "summary": summary,
        }
        on_progress(1.0, "completed")
        return KGBuildResult(output_payload=output_payload)

    def _resolve_output_dir(self, task_id: str, output_dir: str | None) -> Path:
        if output_dir and output_dir.strip():
            candidate = Path(output_dir.strip())
            if not candidate.is_absolute():
                candidate = (REPO_ROOT / candidate).resolve()
            return candidate
        return (DEFAULT_TASK_OUTPUT_ROOT / "tasks" / task_id).resolve()

    def _prepare_input_json(self, output_dir: Path, payload: dict[str, Any]) -> Path:
        input_type = payload.get("input_type")
        if input_type == "json_path":
            raw_path = str(payload.get("json_path", "")).strip()
            if not raw_path:
                raise ValueError("json_path is required when input_type=json_path")
            json_path = Path(raw_path)
            if not json_path.is_absolute():
                json_path = (REPO_ROOT / json_path).resolve()
            if not json_path.exists():
                raise FileNotFoundError(f"json_path does not exist: {json_path}")
            return json_path

        if input_type == "text":
            text = str(payload.get("text") or "").strip()
            topic = str(payload.get("topic") or "web_topic").strip() or "web_topic"
            if not text:
                raise ValueError("text is required when input_type=text")
            path = output_dir / "input_topics.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump([{"topic": topic, "content": text}], handle, ensure_ascii=False, indent=2)
            return path

        raise ValueError(f"Unsupported input_type: {input_type}")

    @staticmethod
    def _load_topics(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not data:
            raise ValueError("Input JSON must be a non-empty topic list")
        return data
