"""QA question entity extraction and graph node matching."""

import re
import traceback

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline.shared import logger
from src.prompt import get_prompt
from src.utils import dedupe_preserve_order
from src.utils import parse_json_like_response
from src.utils import safe_embed_documents


class PipelineQAMatchOpsMixin:
    """Question understanding and node matching helpers for QA."""

    def extract_question_entities(self, question, max_entities=8):
        prompt = ChatPromptTemplate.from_template(get_prompt("qa_question_entity_extract"))
        chain = prompt | self.model

        entities = []
        keywords = []
        try:
            result = chain.invoke({"question": question})
            parsed = parse_json_like_response(result)
            if isinstance(parsed, dict):
                entities = parsed.get("entities", [])
                keywords = parsed.get("keywords", [])
        except Exception:
            logger.warning("Failed to extract question entities by LLM: %s", traceback.format_exc())

        merged = []
        for item in (entities or []) + (keywords or []):
            if isinstance(item, str) and item.strip():
                merged.append(item.strip())

        if not merged:
            merged = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_\-]{1,}", question)

        return dedupe_preserve_order(merged)[:max_entities]

    def match_question_entities_to_graph(
        self,
        question,
        question_entities,
        graph_data=None,
        graph_index=None,
        top_k=5,
        similarity_threshold=0.35,
    ):
        if graph_index is not None:
            entity_lookup = graph_index.get("entity_lookup", {})
            node_names = graph_index.get("node_names", [])
            node_vectors = graph_index.get("node_vectors")
        else:
            if graph_data is None:
                raise ValueError("Either graph_data or graph_index must be provided.")
            entity_lookup, _, _, _ = self._build_graph_indices(graph_data)
            node_names = list(entity_lookup.keys())
            node_texts = [self._entity_to_retrieval_text(entity_lookup[name]) for name in node_names]
            node_vectors = None
            try:
                node_vectors = np.array(safe_embed_documents(self.embeddings, node_texts))
            except Exception:
                logger.warning("Failed to compute entity embeddings: %s", traceback.format_exc())

        if not entity_lookup or not node_names:
            return []

        score_map = {}
        reason_map = {}
        candidates = question_entities[:] if question_entities else [question]

        for candidate in candidates:
            if not candidate or not isinstance(candidate, str):
                continue
            q = candidate.strip()
            if not q:
                continue
            q_lower = q.lower()

            for name in node_names:
                string_score = 0.0
                entity_item = entity_lookup.get(name, {})
                alias_candidates = [name]
                aliases = entity_item.get("aliases", [])
                if isinstance(aliases, list):
                    alias_candidates.extend([str(alias) for alias in aliases if str(alias).strip()])

                for alias_name in alias_candidates:
                    alias_lower = alias_name.lower()
                    alias_score = 0.0
                    if q_lower == alias_lower:
                        alias_score = 1.0 if alias_name == name else 0.95
                    elif q_lower in alias_lower or alias_lower in q_lower:
                        alias_score = 0.90 if alias_name == name else 0.85
                    string_score = max(string_score, alias_score)

                if string_score > score_map.get(name, -1):
                    score_map[name] = string_score
                    reason_map[name] = f"string_match:{q}"

            if node_vectors is not None:
                try:
                    query_vec = self.embeddings.embed_query(q)
                    sims = cosine_similarity([query_vec], node_vectors)[0]
                    for idx, sim in enumerate(sims):
                        sim = float(sim)
                        if sim < similarity_threshold:
                            continue
                        name = node_names[idx]
                        if sim > score_map.get(name, -1):
                            score_map[name] = sim
                            reason_map[name] = f"semantic_similarity:{q}"
                except Exception:
                    logger.warning(
                        "Failed to compute query similarity for %s: %s",
                        q,
                        traceback.format_exc(),
                    )

        if not score_map and node_vectors is not None:
            try:
                query_vec = self.embeddings.embed_query(question)
                sims = cosine_similarity([query_vec], node_vectors)[0]
                top_indices = np.argsort(sims)[::-1][:top_k]
                for idx in top_indices:
                    name = node_names[idx]
                    score_map[name] = float(sims[idx])
                    reason_map[name] = "semantic_fallback:question"
            except Exception:
                logger.warning("Failed to run fallback semantic retrieval: %s", traceback.format_exc())

        ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)[:top_k]
        matched = []
        for name, score in ranked:
            entity = entity_lookup.get(name, {})
            matched.append(
                {
                    "name": name,
                    "score": float(score),
                    "reason": reason_map.get(name, "unknown"),
                    "type": entity.get("type", ""),
                    "description": entity.get("description", ""),
                }
            )
        return matched
