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

    def _normalize_chunk_ids(self, chunk_ids):
        if chunk_ids is None:
            return []
        if isinstance(chunk_ids, list):
            values = chunk_ids
        elif isinstance(chunk_ids, (set, tuple)):
            values = list(chunk_ids)
        else:
            text = str(chunk_ids).strip()
            if not text:
                return []
            values = text.split(";;;") if ";;;" in text else [text]
        normalized = []
        for value in values:
            chunk_id = str(value).strip()
            if chunk_id:
                normalized.append(chunk_id)
        seen = set()
        deduped = []
        for value in normalized:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _normalize_qa_provenance(self, provenance, legacy_chunk_ids=None):
        if not isinstance(provenance, dict):
            provenance = {}
        chunk_ids = self._normalize_chunk_ids(provenance.get("chunk_ids", []))
        chunk_ids = self._normalize_chunk_ids(chunk_ids + self._normalize_chunk_ids(legacy_chunk_ids))
        normalized = {"chunk_ids": chunk_ids}
        strategy = str(provenance.get("strategy", "")).strip()
        if strategy:
            normalized["strategy"] = strategy
        return normalized

    def _normalize_provenance(self, provenance, legacy_chunk_ids=None):
        return self._normalize_qa_provenance(provenance, legacy_chunk_ids=legacy_chunk_ids)

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
        chunk_map = data.get("chunk_map", {})
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError("knowledge_graph must contain list fields: entities and relations.")
        if not isinstance(chunk_map, dict):
            chunk_map = {}
        normalized_chunk_map = {}
        for key, value in chunk_map.items():
            chunk_id = str(key).strip()
            sentence = str(value).strip() if value is not None else ""
            if chunk_id and sentence and chunk_id not in normalized_chunk_map:
                normalized_chunk_map[chunk_id] = sentence
        return {"entities": entities, "relations": relations, "chunk_map": normalized_chunk_map}

    def _build_graph_indices(self, graph_data):
        entity_lookup = {}
        for entity in graph_data.get("entities", []):
            if not isinstance(entity, dict):
                continue
            name = entity.get("name")
            if not name:
                continue
            normalized_entity = dict(entity)
            normalized_entity["provenance"] = self._normalize_qa_provenance(
                entity.get("provenance", {}),
                legacy_chunk_ids=entity.get("chunkid", []),
            )
            entity_lookup[name] = normalized_entity

        adjacency_out = {}
        adjacency_in = {}
        normalized_relations = []

        for rel in graph_data.get("relations", []):
            source = None
            relation = None
            target = None
            rel_description = ""
            rel_provenance = {}

            if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                source, relation, target = rel[0], rel[1], rel[2]
                if len(rel) >= 4:
                    rel_description = rel[3] or ""
                if len(rel) >= 5 and isinstance(rel[4], dict):
                    rel_provenance = rel[4]
            elif isinstance(rel, dict):
                source = rel.get("source")
                relation = rel.get("relation")
                target = rel.get("target")
                rel_description = rel.get("description", "") or rel.get("rel_description", "")
                rel_provenance = rel.get("provenance", {})

            if not source or not relation or not target:
                continue

            rel_item = {
                "source": source,
                "relation": relation,
                "target": target,
                "description": rel_description,
                "provenance": self._normalize_qa_provenance(rel_provenance),
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
                    "aliases": [],
                    "provenance": {"chunk_ids": []},
                }
            if target not in entity_lookup:
                entity_lookup[target] = {
                    "name": target,
                    "type": "Unknown",
                    "description": "",
                    "attributes": {},
                    "aliases": [],
                    "provenance": {"chunk_ids": []},
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
            "chunk_map": normalized_graph.get("chunk_map", {}),
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
