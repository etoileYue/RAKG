from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
from typing import Any

from src.kgAgent import NER_Agent

from app.config import REPO_ROOT


class QAService:
    """Synchronous QA service with in-process graph index cache reuse."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._agent: NER_Agent | None = None

    def ask(
        self,
        *,
        kg_path: str,
        question: str,
        max_hop: int,
        seed_top_k: int,
        max_context_items: int,
    ) -> dict[str, Any]:
        normalized_path = self._resolve_kg_path(kg_path)
        cache_key = self._build_cache_key(normalized_path)

        with self._lock:
            if self._agent is None:
                self._agent = NER_Agent()

            agent = self._agent
            agent.initialize_qa_graph_index(str(normalized_path), cache_key=cache_key)
            result = agent.answer_question_with_kg(
                question=question,
                cache_key=cache_key,
                max_hop=max_hop,
                seed_top_k=seed_top_k,
                max_context_items=max_context_items,
            )

        return {
            "question": result.get("question", question),
            "formatted_answer": result.get("formatted_answer", ""),
            "answer": result.get("answer", ""),
            "retrieved_context": result.get("retrieval", {}),
            "evidence_sources": result.get("evidence_sources", []),
            "graph_paths": result.get("graph_paths", []),
            "intermediate": {
                "llm_output_raw": result.get("llm_output_raw", ""),
            },
        }

    @staticmethod
    def _resolve_kg_path(raw_path: str) -> Path:
        path = Path(raw_path.strip())
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"KG path does not exist: {path}")
        return path

    @staticmethod
    def _build_cache_key(path: Path) -> str:
        digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()
        return f"web_qa:{digest}:{int(path.stat().st_mtime)}"
