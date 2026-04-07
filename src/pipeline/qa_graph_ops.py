"""QA graph normalization, indexing, and cache management."""

import hashlib
import json
import os
import traceback

import numpy as np

from src.pipeline.shared import logger
from src.utils import safe_embed_documents


class PipelineQAGraphOpsMixin:
    """Knowledge graph indexing helpers for QA retrieval."""

    def _normalize_knowledge_graph(self, knowledge_graph):
        data = knowledge_graph
        if isinstance(data, str):
            if os.path.exists(data):
                with open(data, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(data)

        if not isinstance(data, dict):
            raise ValueError(
                "knowledge_graph must be a dict, JSON string, or a valid JSON file path."
            )

        entities = data.get("entities", [])
        relations = data.get("relations", [])
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError("knowledge_graph must contain list fields: entities and relations.")
        return {"entities": entities, "relations": relations}

    def _build_graph_indices(self, graph_data):
        entity_lookup = {}
        for entity in graph_data.get("entities", []):
            if not isinstance(entity, dict):
                continue
            name = entity.get("name")
            if not name:
                continue
            entity_lookup[name] = entity

        adjacency_out = {}
        adjacency_in = {}
        normalized_relations = []

        for rel in graph_data.get("relations", []):
            source = None
            relation = None
            target = None
            rel_description = ""

            if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                source, relation, target = rel[0], rel[1], rel[2]
                if len(rel) >= 4:
                    rel_description = rel[3] or ""
            elif isinstance(rel, dict):
                source = rel.get("source")
                relation = rel.get("relation")
                target = rel.get("target")
                rel_description = rel.get("description", "") or rel.get("rel_description", "")

            if not source or not relation or not target:
                continue

            rel_item = {
                "source": source,
                "relation": relation,
                "target": target,
                "description": rel_description,
            }
            normalized_relations.append(rel_item)

            adjacency_out.setdefault(source, []).append(rel_item)
            adjacency_in.setdefault(target, []).append(rel_item)

            if source not in entity_lookup:
                entity_lookup[source] = {
                    "name": source,
                    "type": "Unknown",
                    "description": "",
                    "attributes": {},
                }
            if target not in entity_lookup:
                entity_lookup[target] = {
                    "name": target,
                    "type": "Unknown",
                    "description": "",
                    "attributes": {},
                }

        return entity_lookup, normalized_relations, adjacency_out, adjacency_in

    def _compute_graph_cache_key(self, knowledge_graph):
        if isinstance(knowledge_graph, str) and os.path.exists(knowledge_graph):
            abs_path = os.path.abspath(knowledge_graph)
            return f"path:{abs_path}"

        normalized_graph = self._normalize_knowledge_graph(knowledge_graph)
        graph_str = json.dumps(normalized_graph, ensure_ascii=False, sort_keys=True)
        digest = hashlib.md5(graph_str.encode("utf-8")).hexdigest()
        return f"graph:{digest}"

    def initialize_qa_graph_index(self, knowledge_graph, cache_key=None, force_rebuild=False):
        normalized_graph = self._normalize_knowledge_graph(knowledge_graph)
        cache_key = cache_key or self._compute_graph_cache_key(normalized_graph)

        if (not force_rebuild) and cache_key in self._qa_graph_index_cache:
            self._qa_default_graph_cache_key = cache_key
            return cache_key

        entity_lookup, normalized_relations, adjacency_out, adjacency_in = self._build_graph_indices(
            normalized_graph
        )
        node_names = list(entity_lookup.keys())
        node_texts = [self._entity_to_retrieval_text(entity_lookup[name]) for name in node_names]

        node_vectors = None
        if node_texts:
            try:
                node_vectors = np.array(safe_embed_documents(self.embeddings, node_texts))
            except Exception:
                logger.warning(
                    "Failed to precompute node embeddings for graph index: %s",
                    traceback.format_exc(),
                )

        self._qa_graph_index_cache[cache_key] = {
            "cache_key": cache_key,
            "graph_data": normalized_graph,
            "entity_lookup": entity_lookup,
            "relations": normalized_relations,
            "adjacency_out": adjacency_out,
            "adjacency_in": adjacency_in,
            "node_names": node_names,
            "node_texts": node_texts,
            "node_vectors": node_vectors,
        }
        self._qa_default_graph_cache_key = cache_key
        return cache_key

    def _get_qa_graph_index(self, knowledge_graph=None, cache_key=None, auto_initialize=True):
        if cache_key and cache_key in self._qa_graph_index_cache:
            return self._qa_graph_index_cache[cache_key]

        if knowledge_graph is not None:
            resolved_key = cache_key or self._compute_graph_cache_key(knowledge_graph)
            if resolved_key not in self._qa_graph_index_cache:
                if not auto_initialize:
                    raise ValueError(f"QA graph index not found for cache_key={resolved_key}")
                self.initialize_qa_graph_index(knowledge_graph, cache_key=resolved_key)
            return self._qa_graph_index_cache[resolved_key]

        if cache_key:
            raise ValueError(f"QA graph index not found for cache_key={cache_key}")

        if (
            self._qa_default_graph_cache_key
            and self._qa_default_graph_cache_key in self._qa_graph_index_cache
        ):
            return self._qa_graph_index_cache[self._qa_default_graph_cache_key]

        raise ValueError(
            "No QA graph index available. Please call initialize_qa_graph_index(...) first, "
            "or pass knowledge_graph to answer_question_with_kg/build_qa_retrieval_context."
        )

    def clear_qa_graph_index(self, cache_key=None):
        if cache_key is None:
            self._qa_graph_index_cache.clear()
            self._qa_default_graph_cache_key = None
            return

        self._qa_graph_index_cache.pop(cache_key, None)
        if self._qa_default_graph_cache_key == cache_key:
            self._qa_default_graph_cache_key = None

    def _entity_to_retrieval_text(self, entity):
        attrs = entity.get("attributes", {})
        attr_text = ""
        if isinstance(attrs, dict):
            attr_text = "; ".join([f"{k}:{v}" for k, v in attrs.items()])
        elif isinstance(attrs, list):
            attr_text = "; ".join([str(item) for item in attrs])

        aliases = entity.get("aliases", [])
        alias_text = ""
        if isinstance(aliases, list):
            alias_text = "; ".join([str(item) for item in aliases if str(item).strip()])

        text_parts = [
            entity.get("name", ""),
            entity.get("type", ""),
            entity.get("description", ""),
            attr_text,
            alias_text,
        ]
        return " ".join([str(part) for part in text_parts if part]).strip()
