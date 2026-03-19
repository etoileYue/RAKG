from collections import deque
import hashlib
import json
import logging
import os
import re
import traceback

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import get_logger
from src.prompt import kg_qa_answer_prompt_cn
from src.prompt import question_entity_extract_prompt_cn
from src.utils import dedupe_preserve_order
from src.utils import parse_json_like_response

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file="Agent.log",
)


class KnowledgeGraphQA:
    def _parse_json_like_response(self, response):
        return parse_json_like_response(response)

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
                node_vectors = np.array(self.embeddings.embed_documents(node_texts))
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

    def extract_question_entities(self, question, max_entities=8):
        prompt = ChatPromptTemplate.from_template(question_entity_extract_prompt_cn)
        chain = prompt | self.model

        entities = []
        keywords = []
        try:
            result = chain.invoke({"question": question})
            parsed = self._parse_json_like_response(result)
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
                node_vectors = np.array(self.embeddings.embed_documents(node_texts))
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

    def expand_graph_neighbors(
        self,
        graph_data=None,
        seed_nodes=None,
        graph_index=None,
        max_hop=1,
        max_neighbors_per_node=8,
        max_paths=40,
    ):
        if graph_index is not None:
            entity_lookup = graph_index.get("entity_lookup", {})
            adjacency_out = graph_index.get("adjacency_out", {})
            adjacency_in = graph_index.get("adjacency_in", {})
        else:
            if graph_data is None:
                raise ValueError("Either graph_data or graph_index must be provided.")
            normalized_graph = self._normalize_knowledge_graph(graph_data)
            entity_lookup, _, adjacency_out, adjacency_in = self._build_graph_indices(normalized_graph)

        seed_nodes = seed_nodes or []
        hop_limit = 1 if max_hop <= 1 else 2

        entity_evidence = []
        relation_evidence = []
        graph_paths = []
        visited_path = set()
        visited_entity_source = set()
        visited_relation_source = set()

        def add_entity_evidence(node_name):
            entity = entity_lookup.get(node_name, {"name": node_name})
            description = entity.get("description", "")
            if isinstance(description, str) and description.strip():
                source = f"entity:{node_name}.description"
                if source not in visited_entity_source:
                    entity_evidence.append(
                        {
                            "source": source,
                            "text": description.strip(),
                            "entity": node_name,
                        }
                    )
                    visited_entity_source.add(source)

            attrs = entity.get("attributes", {})
            if isinstance(attrs, dict):
                for key, value in list(attrs.items())[:5]:
                    source = f"entity:{node_name}.attribute:{key}"
                    if source in visited_entity_source:
                        continue
                    text = f"{node_name} 的属性 {key}: {value}"
                    entity_evidence.append({"source": source, "text": str(text), "entity": node_name})
                    visited_entity_source.add(source)

        for seed in seed_nodes:
            if seed not in entity_lookup:
                continue
            add_entity_evidence(seed)

            queue = deque([(seed, 0, seed)])
            visited_depth = {seed: 0}

            while queue and len(graph_paths) < max_paths:
                current_node, current_depth, current_path = queue.popleft()
                if current_depth >= hop_limit:
                    continue

                outgoing = adjacency_out.get(current_node, [])[:max_neighbors_per_node]
                incoming = adjacency_in.get(current_node, [])[:max_neighbors_per_node]

                for rel in outgoing:
                    source = rel["source"]
                    relation = rel["relation"]
                    target = rel["target"]
                    rel_desc = rel.get("description", "")
                    next_node = target
                    new_path = f"{current_path} --[{relation}]-> {next_node}"

                    if new_path not in visited_path:
                        graph_paths.append(new_path)
                        visited_path.add(new_path)

                    rel_source = f"relation:{source}--[{relation}]-->{target}"
                    if rel_source not in visited_relation_source:
                        relation_evidence.append(
                            {
                                "source": rel_source,
                                "text": rel_desc.strip()
                                if isinstance(rel_desc, str) and rel_desc.strip()
                                else f"{source} --[{relation}]-> {target}",
                                "path": new_path,
                                "hop": current_depth + 1,
                            }
                        )
                        visited_relation_source.add(rel_source)

                    add_entity_evidence(next_node)
                    next_depth = current_depth + 1
                    if next_depth <= hop_limit and next_depth < visited_depth.get(next_node, 999):
                        visited_depth[next_node] = next_depth
                        queue.append((next_node, next_depth, new_path))

                for rel in incoming:
                    source = rel["source"]
                    relation = rel["relation"]
                    target = rel["target"]
                    rel_desc = rel.get("description", "")
                    next_node = source
                    new_path = f"{current_path} <-[{relation}]-- {next_node}"

                    if new_path not in visited_path:
                        graph_paths.append(new_path)
                        visited_path.add(new_path)

                    rel_source = f"relation:{source}--[{relation}]-->{target}"
                    if rel_source not in visited_relation_source:
                        relation_evidence.append(
                            {
                                "source": rel_source,
                                "text": rel_desc.strip()
                                if isinstance(rel_desc, str) and rel_desc.strip()
                                else f"{source} --[{relation}]-> {target}",
                                "path": new_path,
                                "hop": current_depth + 1,
                            }
                        )
                        visited_relation_source.add(rel_source)

                    add_entity_evidence(next_node)
                    next_depth = current_depth + 1
                    if next_depth <= hop_limit and next_depth < visited_depth.get(next_node, 999):
                        visited_depth[next_node] = next_depth
                        queue.append((next_node, next_depth, new_path))

        return {
            "entity_evidence": entity_evidence,
            "relation_evidence": relation_evidence,
            "graph_paths": graph_paths[:max_paths],
        }

    def build_qa_retrieval_context(
        self,
        question,
        knowledge_graph=None,
        max_hop=2,
        seed_top_k=5,
        max_context_items=30,
        cache_key=None,
        graph_index=None,
    ):
        if graph_index is None:
            graph_index = self._get_qa_graph_index(
                knowledge_graph=knowledge_graph,
                cache_key=cache_key,
                auto_initialize=True,
            )

        question_entities = self.extract_question_entities(question)
        matched_nodes = self.match_question_entities_to_graph(
            question=question,
            question_entities=question_entities,
            graph_index=graph_index,
            top_k=seed_top_k,
        )
        seed_nodes = [item["name"] for item in matched_nodes]

        expanded = self.expand_graph_neighbors(
            graph_index=graph_index,
            seed_nodes=seed_nodes,
            max_hop=max_hop,
        )

        evidence_items = []
        for item in expanded["entity_evidence"]:
            evidence_items.append({"source": item["source"], "text": item["text"], "path": ""})
        for item in expanded["relation_evidence"]:
            evidence_items.append(
                {
                    "source": item["source"],
                    "text": item["text"],
                    "path": item.get("path", ""),
                }
            )

        evidence_items = evidence_items[:max_context_items]
        context_lines = []
        for idx, item in enumerate(evidence_items, start=1):
            line = f"{idx}. [source={item['source']}] {item['text']}"
            if item.get("path"):
                line += f" | path={item['path']}"
            context_lines.append(line)
        context_text = "\n".join(context_lines)

        return {
            "question": question,
            "cache_key": graph_index.get("cache_key"),
            "question_entities": question_entities,
            "matched_nodes": matched_nodes,
            "seed_nodes": seed_nodes,
            "graph_paths": expanded["graph_paths"],
            "evidence_items": evidence_items,
            "context_text": context_text,
        }

    def _normalize_evidence_output(self, evidence_sources, fallback_evidence):
        if not isinstance(evidence_sources, list):
            evidence_sources = []

        normalized = []
        for item in evidence_sources:
            if isinstance(item, dict):
                source = str(item.get("source", "")).strip()
                quote = str(item.get("quote", "")).strip()
                if source or quote:
                    normalized.append({"source": source, "quote": quote})
            elif isinstance(item, str) and item.strip():
                normalized.append({"source": "", "quote": item.strip()})

        if normalized:
            return normalized

        fallback = []
        for item in fallback_evidence[:3]:
            fallback.append({"source": item.get("source", ""), "quote": item.get("text", "")})
        return fallback

    def _format_qa_output(self, answer, evidence_sources, graph_paths):
        answer_text = (
            answer
            if isinstance(answer, str) and answer.strip()
            else "根据现有图谱证据不足以得出确定结论"
        )
        lines = [f"答案：{answer_text}", "证据来源："]
        if evidence_sources:
            for idx, item in enumerate(evidence_sources, start=1):
                source = item.get("source", "")
                quote = item.get("quote", "")
                lines.append(f"{idx}. [{source}] {quote}")
        else:
            lines.append("1. 无可用证据")

        lines.append("图谱路径：")
        if graph_paths:
            for idx, path in enumerate(graph_paths, start=1):
                lines.append(f"{idx}. {path}")
        else:
            lines.append("1. 无可用路径")
        return "\n".join(lines)

    def answer_question_with_kg(
        self,
        question,
        knowledge_graph=None,
        max_hop=2,
        seed_top_k=5,
        max_context_items=30,
        cache_key=None,
        graph_index=None,
    ):
        retrieval = self.build_qa_retrieval_context(
            question=question,
            knowledge_graph=knowledge_graph,
            max_hop=max_hop,
            seed_top_k=seed_top_k,
            max_context_items=max_context_items,
            cache_key=cache_key,
            graph_index=graph_index,
        )

        graph_paths_text = "\n".join([f"- {path}" for path in retrieval["graph_paths"][:10]])
        prompt = ChatPromptTemplate.from_template(kg_qa_answer_prompt_cn)
        chain = prompt | self.model
        llm_result = chain.invoke(
            {
                "question": question,
                "context": retrieval["context_text"],
                "graph_paths": graph_paths_text,
            }
        )

        parsed = self._parse_json_like_response(llm_result)
        if isinstance(parsed, dict):
            answer = parsed.get("answer", "")
            evidence_sources = self._normalize_evidence_output(
                parsed.get("evidence_sources", []),
                retrieval["evidence_items"],
            )
            graph_paths = parsed.get("graph_paths", [])
            if not isinstance(graph_paths, list):
                graph_paths = []
        else:
            answer = llm_result.content if hasattr(llm_result, "content") else str(llm_result)
            evidence_sources = self._normalize_evidence_output([], retrieval["evidence_items"])
            graph_paths = retrieval["graph_paths"][:3]

        if not graph_paths:
            graph_paths = retrieval["graph_paths"][:3]

        formatted_answer = self._format_qa_output(answer, evidence_sources, graph_paths)
        return {
            "question": question,
            "retrieval": retrieval,
            "llm_output_raw": llm_result.content if hasattr(llm_result, "content") else str(llm_result),
            "answer": answer,
            "evidence_sources": evidence_sources,
            "graph_paths": graph_paths,
            "formatted_answer": formatted_answer,
        }
