from copy import deepcopy
from itertools import combinations
import json
import logging
import os
import traceback

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import get_logger
from src.prompt import extract_entiry_centric_kg_en_v2
from src.prompt import judge_sim_entity_en
from src.prompt import text2entity_en
from src.utils import add_chunkid_to_entities
from src.utils import append_jsonl
from src.utils import dedupe_preserve_order
from src.utils import ensure_parent_dir
from src.utils import parse_similarity_response
from src.utils import renumber_entities
from src.utils import retry

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file="Agent.log",
)

debug_logger = get_logger(
    name=f"{os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME)}.debug",
    level=logging.DEBUG,
    log_file="Debug.log",
)


class NERPipeline:
    def _ensure_parent_dir(self, output_file):
        ensure_parent_dir(output_file)

    def _append_jsonl(self, output_file, data):
        append_jsonl(output_file, data)

    def _dedupe_preserve_order(self, items):
        return dedupe_preserve_order(items)

    def _is_boundary_candidate(self, similarity_score, threshold, gray_margin):
        return similarity_score <= (threshold + gray_margin)

    def add_chunkid(self, ner_result, chunkid):
        return add_chunkid_to_entities(ner_result, chunkid)

    def rewrite(self, ner_result, entity_num):
        return renumber_entities(ner_result, entity_num)

    def _split_semicolon_values(self, value)->list:
        """拆分由;;;分隔的字符串，返回list作为拆分结果"""
        if value is None:
            return []
        if isinstance(value, list):
            raw_items = value
        else:
            raw_items = str(value).split(";;;")

        normalized = []
        for item in raw_items:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return self._dedupe_preserve_order(normalized)

    def _merge_semicolon_text(self, left, right):
        merged = self._dedupe_preserve_order(
            self._split_semicolon_values(left) + self._split_semicolon_values(right)
        )
        return ";;;".join(merged)

    def _normalize_aliases(self, aliases, canonical_name):
        """返回别名列表，若存在以正式名相同的别名则去除"""
        if aliases is None:
            aliases = []
        if not isinstance(aliases, list):
            aliases = self._split_semicolon_values(aliases)

        merged = []
        for alias in aliases:
            if alias is None:
                continue
            text = str(alias).strip()
            if not text:
                continue
            if canonical_name and text == canonical_name:
                continue
            merged.append(text)
        return self._dedupe_preserve_order(merged)

    def _normalize_relation_text(self, relation_text):
        if relation_text is None:
            return ""
        relation = str(relation_text).strip().lower()
        if not relation:
            return ""
        return " ".join(relation.split())

    def _normalize_graph_input(self, graph_data):
        """返回图谱内容，{"entities": graph_data.entities, "relations": graph_data.relations}"""
        if graph_data is None:
            return {"entities": [], "relations": []}

        data = graph_data
        if isinstance(data, str):
            if os.path.exists(data):
                with open(data, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(data)

        if not isinstance(data, dict):
            raise ValueError("graph_data must be a dict, JSON string, or JSON file path.")

        entities = data.get("entities", [])
        relations = data.get("relations", [])
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError("graph_data must contain list fields: entities and relations.")
        return {"entities": entities, "relations": relations}

    def ensure_entity_aliases(self, entity_dic):
        """添加aliases字段"""
        normalized = {}
        for entity_id, entity in (entity_dic or {}).items():
            if not isinstance(entity, dict):
                continue

            canonical_name = str(entity.get("name", "")).strip()
            if not canonical_name:
                continue

            item = deepcopy(entity)
            item["name"] = canonical_name
            item["aliases"] = self._normalize_aliases(item.get("aliases", []), canonical_name)
            normalized[entity_id] = item
        return normalized

    def _collapse_entities_by_name(self, entity_dic):
        """按标准名称合并实体并合并它们的描述/块ID/别名"""
        grouped = {}
        for _, entity in (entity_dic or {}).items():
            if not isinstance(entity, dict):
                continue

            name = str(entity.get("name", "")).strip()
            if not name:
                continue

            if name not in grouped:
                grouped[name] = {
                    "name": name,
                    "type": entity.get("type", "Unknown"),
                    "description": entity.get("description", ""),
                    "chunkid": entity.get("chunkid", ""),
                    "aliases": self._normalize_aliases(entity.get("aliases", []), name),
                }
                continue

            current = grouped[name]
            if (
                (not current.get("type") or str(current.get("type")).strip().lower() == "unknown")
                and entity.get("type")
            ):
                current["type"] = entity.get("type")

            current["description"] = self._merge_semicolon_text(
                current.get("description", ""), entity.get("description", "")
            )
            current["chunkid"] = self._merge_semicolon_text(
                current.get("chunkid", ""), entity.get("chunkid", "")
            )
            current["aliases"] = self._normalize_aliases(
                current.get("aliases", []) + entity.get("aliases", []),
                name,
            )

        collapsed = {}
        for idx, (_, entity) in enumerate(grouped.items(), start=1):
            collapsed[f"entity{idx}"] = entity
        return collapsed

    def extract_from_text_single(self, text_single, output_file):
        prompt = ChatPromptTemplate.from_template(text2entity_en)
        chain = prompt | self.model
        result = chain.invoke({"text": text_single})
        debug_logger.debug("-extract_from_text_single-")
        debug_logger.debug(f"text_single={text_single}, result={result}")

        if hasattr(result, "content"):
            result_json = json.loads(result.content)
        else:
            result_json = json.loads(result)

        combined_data = {"text": text_single, "entities": result_json}
        self._append_jsonl(output_file, combined_data)
        return result_json

    def extract_from_text_multiply(self, text_list, sent_to_id, output_file):
        ner_result_for_all = {}
        entity_num = 1
        for text in text_list:
            ner_result = self.extract_from_text_single(text, output_file)
            if "State" in ner_result:
                continue

            ner_result_num = len(ner_result)
            ner_result = self.rewrite(ner_result, entity_num)

            chunkid = sent_to_id.get(text)
            if chunkid is None:
                logger.warning("Sentence not found in sentence_to_id mapping, skipping chunk.")
                continue

            entity_num += ner_result_num
            ner_result_with_chunkid = self.add_chunkid(ner_result, chunkid)
            ner_result_for_all.update(ner_result_with_chunkid)
        return ner_result_for_all

    def similarity_candidates(self, entities, threshold=0.60):
        def get_embedding_vector(text):
            result = self.embeddings.embed_documents([text])
            if isinstance(result, list) and isinstance(result[0], list):
                return result[0]
            return result

        entity_texts = {k: f"{v['name']} {v['type']}" for k, v in entities.items()}
        vectors = {k: get_embedding_vector(text) for k, text in entity_texts.items()}

        keys = list(vectors.keys())
        sim_matrix = np.zeros((len(keys), len(keys)))

        for i, j in combinations(range(len(keys)), 2):
            sim = float(cosine_similarity([vectors[keys[i]]], [vectors[keys[j]]])[0][0])
            sim_matrix[i][j] = sim

        candidates = [
            (keys[i], keys[j], float(sim_matrix[i][j]))
            for i, j in zip(*np.where(sim_matrix > threshold))
        ]
        return candidates

    @retry()
    def similarity_llm_single(self, entity1, entity2):
        prompt = ChatPromptTemplate.from_template(judge_sim_entity_en)
        chain = prompt | self.similarity_model
        result = chain.invoke({"entity1": str(entity1), "entity2": str(entity2)})
        debug_logger.debug("-similarity_llm_single-")
        debug_logger.debug(f"entity1={entity1}, entity2={entity2}, result={result}")
        return parse_similarity_response(result)

    def _run_two_pass_similarity_disambiguation(
        self,
        candidates,
        left_entities,
        right_entities=None,
        threshold=0.60,
        gray_margin=0.05,
        scope_label="entity",
    ):
        if right_entities is None:
            right_entities = left_entities

        positives = []
        gray_queue = []
        for left_id, right_id, score in candidates:
            entity1 = left_entities.get(left_id)
            entity2 = right_entities.get(right_id)
            try:
                result = self.similarity_llm_single(entity1, entity2)
                is_boundary = self._is_boundary_candidate(score, threshold, gray_margin)
                needs_review = result.get("needs_review", False) or is_boundary

                if needs_review:
                    gray_queue.append(
                        {
                            "pair": (left_id, right_id),
                            "similarity_score": score,
                            "reason": result.get("reason", "unknown"),
                            "parse_status": result.get("parse_status", "unknown"),
                            "first_pass_result": result,
                        }
                    )
                    continue

                if result.get("result", False):
                    positives.append((left_id, right_id, score))
            except Exception:
                logger.error(
                    "Error processing %s pair (%s, %s): %s",
                    scope_label,
                    left_id,
                    right_id,
                    traceback.format_exc(),
                )

        resolved_by_second_pass = 0
        for item in gray_queue:
            left_id, right_id = item["pair"]
            entity1 = left_entities.get(left_id)
            entity2 = right_entities.get(right_id)
            try:
                second_pass_result = self.similarity_llm_single(entity1, entity2)
                if second_pass_result.get("result", False):
                    positives.append((left_id, right_id, item["similarity_score"]))
                    resolved_by_second_pass += 1
                item["second_pass_result"] = second_pass_result
            except Exception:
                logger.error(
                    "Error in second-pass %s disambiguation for pair (%s, %s): %s",
                    scope_label,
                    left_id,
                    right_id,
                    traceback.format_exc(),
                )

        return positives, gray_queue, resolved_by_second_pass

    def similarity_result(self, entities, threshold=0.60, gray_margin=0.05):
        candidates = self.similarity_candidates(entities, threshold=threshold)
        positives, gray_queue, resolved_by_second_pass = self._run_two_pass_similarity_disambiguation(
            candidates=candidates,
            left_entities=entities,
            threshold=threshold,
            gray_margin=gray_margin,
            scope_label="entity",
        )
        candidates_result = [(left_id, right_id) for left_id, right_id, _ in positives]

        self.last_disambiguation_gray_queue = gray_queue
        logger.info(
            "Disambiguation summary: candidates=%s, gray_queue=%s, "
            "resolved_by_second_pass=%s, merged_pairs=%s",
            len(candidates),
            len(gray_queue),
            resolved_by_second_pass,
            len(candidates_result),
        )
        return candidates_result

    def similarity_candidates_cross(self, left_entities, right_entities, threshold=0.60):
        """生成相似度高于阈值的跨图合并候选项。"""
        if not left_entities or not right_entities:
            return []

        left_keys = list(left_entities.keys())
        right_keys = list(right_entities.keys())

        left_texts = [f"{left_entities[k].get('name', '')} {left_entities[k].get('type', '')}" for k in left_keys]
        right_texts = [
            f"{right_entities[k].get('name', '')} {right_entities[k].get('type', '')}" for k in right_keys
        ]

        left_vectors = np.array(self.embeddings.embed_documents(left_texts))
        right_vectors = np.array(self.embeddings.embed_documents(right_texts))
        sim_matrix = cosine_similarity(left_vectors, right_vectors)

        candidates = []
        matched = np.argwhere(sim_matrix > threshold)
        for left_idx, right_idx in matched:
            candidates.append(
                (
                    left_keys[int(left_idx)],
                    right_keys[int(right_idx)],
                    float(sim_matrix[int(left_idx)][int(right_idx)]),
                )
            )
        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    def cross_similarity_result(self, left_entities, right_entities, threshold=0.60, gray_margin=0.05):
        # Run LLM verification for cross-graph candidates and keep best match per new entity.
        candidates = self.similarity_candidates_cross(
            left_entities=left_entities,
            right_entities=right_entities,
            threshold=threshold,
        )

        positives, gray_queue, _ = self._run_two_pass_similarity_disambiguation(
            candidates=candidates,
            left_entities=left_entities,
            right_entities=right_entities,
            threshold=threshold,
            gray_margin=gray_margin,
            scope_label="cross-graph entity",
        )

        best_match = {}
        for left_id, right_id, score in positives:
            current = best_match.get(left_id)
            if current is None or score > current["score"]:
                best_match[left_id] = {"target_id": right_id, "score": score}

        logger.info(
            "Cross-graph entity alignment summary: candidates=%s, gray_queue=%s, matched=%s",
            len(candidates),
            len(gray_queue),
            len(best_match),
        )
        return best_match

    def _build_existing_entity_lookup(self, existing_graph):
        # Build temporary ID-indexed lookup from existing graph entities.
        graph = self._normalize_graph_input(existing_graph)
        existing_entities = {}
        for idx, entity in enumerate(graph.get("entities", []), start=1):
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            existing_entities[f"existing_{idx}"] = {
                "name": name,
                "type": entity.get("type", "Unknown"),
                "description": entity.get("description", ""),
                "aliases": self._normalize_aliases(entity.get("aliases", []), name),
            }
        return existing_entities

    def align_entities_to_existing_graph(
        self,
        new_entities,
        existing_graph,
        threshold=0.60,
        gray_margin=0.05,
    ):
        # Align newly extracted entities to existing graph entities using cross-graph disambiguation.
        new_entities = self.ensure_entity_aliases(new_entities)
        if not new_entities:
            return {}, {}

        existing_entities = self._build_existing_entity_lookup(existing_graph)
        if not existing_entities:
            return self._collapse_entities_by_name(new_entities), {}

        matched_map = self.cross_similarity_result(
            left_entities=new_entities,
            right_entities=existing_entities,
            threshold=threshold,
            gray_margin=gray_margin,
        )

        aligned_entities = {}
        alias_resolution = {}
        for entity_id, entity in new_entities.items():
            match = matched_map.get(entity_id)
            if not match:
                aligned_entities[entity_id] = entity
                alias_resolution[entity.get("name", "")] = entity.get("name", "")
                continue

            existing_entity = existing_entities.get(match["target_id"], {})
            canonical_name = existing_entity.get("name") or entity.get("name")
            aligned = deepcopy(entity)
            aligned["name"] = canonical_name

            existing_type = str(existing_entity.get("type", "")).strip()
            if existing_type and existing_type.lower() != "unknown":
                aligned["type"] = existing_type

            aligned["description"] = self._merge_semicolon_text(
                existing_entity.get("description", ""),
                aligned.get("description", ""),
            )
            aligned["aliases"] = self._normalize_aliases(
                []
                + existing_entity.get("aliases", [])
                + [entity.get("name", "")]
                + entity.get("aliases", []),
                canonical_name,
            )

            aligned_entities[entity_id] = aligned
            alias_resolution[entity.get("name", "")] = canonical_name
            for alias in aligned["aliases"]:
                alias_resolution[alias] = canonical_name

        collapsed = self._collapse_entities_by_name(aligned_entities)
        return collapsed, alias_resolution

    def entity_Disambiguation(self, entity_dic, sim_entity_list):
        entity_dic = self.ensure_entity_aliases(entity_dic)
        parent = {}

        # 使用并查集将两两一组的相似实体合并
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        for entity in entity_dic:
            parent[entity] = entity

        for pair in sim_entity_list:
            a, b = pair
            if a in entity_dic and b in entity_dic:
                union(a, b)

        groups = {}
        for entity in entity_dic:
            root = find(entity)
            if root not in groups:
                groups[root] = []
            groups[root].append(entity)

        for group in groups.values():
            if len(group) == 1:
                continue

            main_entity = group[0]
            descriptions = []
            chunkids = []
            aliases = []

            for entity in group:
                descriptions.extend(self._split_semicolon_values(entity_dic[entity].get("description", "")))
                chunkids.extend(self._split_semicolon_values(entity_dic[entity].get("chunkid", "")))
                aliases.append(entity_dic[entity].get("name", ""))
                aliases.extend(entity_dic[entity].get("aliases", []))
                if entity != main_entity:
                    del entity_dic[entity]

            dedup_descriptions = self._dedupe_preserve_order(descriptions)
            dedup_chunkids = self._dedupe_preserve_order(chunkids)
            entity_dic[main_entity]["description"] = ";;;".join(dedup_descriptions)
            entity_dic[main_entity]["chunkid"] = ";;;".join(dedup_chunkids)
            entity_dic[main_entity]["aliases"] = self._normalize_aliases(
                aliases,
                entity_dic[main_entity].get("name", ""),
            )
        return self._collapse_entities_by_name(entity_dic)

    def get_sentences_for_entity(self, entity_dic, entity_id, id_to_sentence):
        if entity_id not in entity_dic:
            raise ValueError(f"Entity '{entity_id}' not found in entity_dic.")

        chunkids = entity_dic[entity_id].get("chunkid", "")
        if not chunkids:
            return []

        chunkid_list = chunkids.split(";;;")
        chunkid_list = [cid.strip() for cid in chunkid_list if cid.strip()]

        sentences = []
        for chunkid in chunkid_list:
            if chunkid in id_to_sentence:
                sentences.append(id_to_sentence[chunkid])
            else:
                logger.warning("Chunk ID '%s' not found in id_to_sentence.", chunkid)
        return sentences

    def get_retriever_context(self, query, sentences, sentence_to_id, vectors, top_k=5):
        if not sentences or not vectors:
            return []

        query_vector = self.embeddings.embed_query(query)
        sentence_vectors = np.array(vectors)

        try:
            similarities = cosine_similarity([query_vector], sentence_vectors)[0]
        except Exception:
            logger.error("Failed to calculate sentence similarity: %s", traceback.format_exc())
            return []

        top_indices = np.argsort(similarities)[::-1][:top_k]
        retriever_context = []
        for idx in top_indices:
            sentence = sentences[idx]
            similarity = similarities[idx]
            sentence_id = sentence_to_id[sentence]
            retriever_context.append((sentence, similarity, sentence_id))

        return retriever_context

    def get_target_kg_single(
        self,
        entity_dic,
        entity_id,
        id_to_sentence,
        sentences,
        sentence_to_id,
        vectors,
        output_file,
        related_kg=None,
    ):
        chunk_text_list = self.get_sentences_for_entity(entity_dic, entity_id, id_to_sentence)
        query = entity_dic[entity_id].get("name", "")
        context = self.get_retriever_context(query, sentences, sentence_to_id, vectors, top_k=5)
        retrieved_sentences = [item[0] for item in context]
        unique_sentences = self._dedupe_preserve_order(chunk_text_list + retrieved_sentences)
        chunk_text = ", ".join(unique_sentences)
        related_kg_payload = "none"
        if related_kg:
            related_kg_payload = json.dumps(related_kg, ensure_ascii=False)

        prompt = ChatPromptTemplate.from_template(extract_entiry_centric_kg_en_v2)
        chain = prompt | self.model
        result = chain.invoke(
            {
                "text": chunk_text,
                "target_entity": entity_dic[entity_id].get("name"),
                "related_kg": related_kg_payload,
            }
        )

        debug_logger.debug("-get_target_kg_single-")
        debug_logger.debug(
            "text=%s, target_entity=%s, related_kg=%s, result=%s",
            chunk_text,
            entity_dic[entity_id].get("name"),
            related_kg_payload,
            result,
        )

        if hasattr(result, "content"):
            result_json = json.loads(result.content)
        else:
            result_json = json.loads(result)

        combined_data = {"chunk_text": chunk_text, "entity": entity_dic[entity_id], "kg": result_json}
        self._append_jsonl(output_file, combined_data)
        return result_json

    def get_target_kg_all(
        self,
        entity_dic,
        id_to_sentence,
        sentences,
        sentence_to_id,
        vectors,
        output_file,
        related_kg_map=None,
    ):
        results = {}
        related_kg_map = related_kg_map or {}
        for entity_id in entity_dic:
            result = self.get_target_kg_single(
                entity_dic,
                entity_id,
                id_to_sentence,
                sentences,
                sentence_to_id,
                vectors,
                output_file,
                related_kg=related_kg_map.get(entity_id),
            )
            results[entity_id] = result
        return results

    def convert_knowledge_graph(self, input_data):
        output = {"entities": [], "relations": []}
        entity_registry = {}

        for entity_key in input_data:
            node_data = input_data.get(entity_key, {})
            central_entity = node_data.get("central_entity")
            if not isinstance(central_entity, dict):
                logger.warning(
                    "Skip malformed kg node for %s: missing central_entity dict.",
                    entity_key,
                )
                continue

            entity_name = central_entity.get("name")
            entity_type = central_entity.get("type", "Unknown")
            if not entity_name:
                logger.warning("Skip malformed central_entity for %s: missing name.", entity_key)
                continue

            if entity_name not in entity_registry:
                entity = {
                    "name": entity_name,
                    "type": entity_type,
                    "description": central_entity.get("description", ""),
                    "attributes": {},
                    "aliases": self._normalize_aliases(central_entity.get("aliases", []), entity_name),
                }
                if "attributes" in central_entity:
                    for attr in central_entity["attributes"]:
                        entity["attributes"][attr["key"]] = attr["value"]
                entity_registry[entity_name] = entity
            else:
                existing_entity = entity_registry[entity_name]
                existing_entity["description"] = self._merge_semicolon_text(
                    existing_entity.get("description", ""),
                    central_entity.get("description", ""),
                )
                existing_entity["aliases"] = self._normalize_aliases(
                    existing_entity.get("aliases", []) + central_entity.get("aliases", []),
                    entity_name,
                )
                if "attributes" in central_entity:
                    for attr in central_entity["attributes"]:
                        key = attr.get("key")
                        value = attr.get("value")
                        if key is None or value is None:
                            continue
                        existing_entity["attributes"].setdefault(key, value)

        for entity_key in input_data:
            node_data = input_data.get(entity_key, {})
            central_entity = node_data.get("central_entity")
            if not isinstance(central_entity, dict):
                continue

            source_name = central_entity.get("name")
            if not source_name:
                continue

            if "relationships" not in central_entity:
                continue

            for rel in central_entity["relationships"]:
                if not isinstance(rel, dict):
                    logger.warning(
                        "Skip malformed relation for %s: relation is not a dict.",
                        entity_key,
                    )
                    continue
                if "target_name" not in rel or "target_type" not in rel or "relation" not in rel:
                    logger.warning(
                        "Skip malformed relation for %s: missing required fields.",
                        entity_key,
                    )
                    continue

                target_names = (
                    rel["target_name"]
                    if isinstance(rel["target_name"], list)
                    else [rel["target_name"]]
                )
                target_type = rel["target_type"]

                for target_name in target_names:
                    if target_name not in entity_registry:
                        entity_registry[target_name] = {
                            "name": target_name,
                            "type": target_type,
                            "description": rel.get("target_description", ""),
                            "attributes": {},
                            "aliases": [],
                        }

                    relation_description = rel.get("relation_description", "")
                    output["relations"].append(
                        [
                            source_name,
                            rel["relation"],
                            target_name,
                            relation_description,
                        ]
                    )

        output["entities"] = list(entity_registry.values())
        return output

    def _iter_normalized_relations(self, graph_data):
        # Iterate all relations in normalized dict shape: source/relation/target/description.
        for rel in graph_data.get("relations", []):
            source = None
            relation = None
            target = None
            description = ""

            if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                source, relation, target = rel[0], rel[1], rel[2]
                if len(rel) >= 4:
                    description = rel[3] or ""
            elif isinstance(rel, dict):
                source = rel.get("source")
                relation = rel.get("relation")
                target = rel.get("target")
                description = rel.get("description", "") or rel.get("rel_description", "")

            if not source or not relation or not target:
                continue

            yield {
                "source": str(source).strip(),
                "relation": str(relation).strip(),
                "target": str(target).strip(),
                "description": str(description or "").strip(),
            }

    def _build_alias_to_canonical(self, entity_registry):
        # Build alias-to-canonical map for entity name resolution.
        alias_to_canonical = {}
        for canonical_name, entity in entity_registry.items():
            alias_to_canonical[canonical_name] = canonical_name
            aliases = self._normalize_aliases(entity.get("aliases", []), canonical_name)
            for alias in aliases:
                alias_to_canonical[str(alias)] = canonical_name
        return alias_to_canonical

    def _resolve_name_by_alias(self, raw_name, alias_to_canonical):
        # Resolve raw source/target text to canonical entity name when possible.
        if raw_name is None:
            return None
        name = str(raw_name).strip()
        if not name:
            return None
        return alias_to_canonical.get(name, name)

    def _merge_entity_records(self, base_entity, incoming_entity):
        # Merge two entity records into one canonical entity representation.
        canonical_name = base_entity.get("name") or incoming_entity.get("name")
        merged = {
            "name": canonical_name,
            "type": base_entity.get("type", "Unknown"),
            "description": self._merge_semicolon_text(
                base_entity.get("description", ""),
                incoming_entity.get("description", ""),
            ),
            "attributes": {},
            "aliases": [],
        }

        merged_attrs = {}
        for attrs in [base_entity.get("attributes", {}), incoming_entity.get("attributes", {})]:
            if isinstance(attrs, dict):
                for key, value in attrs.items():
                    merged_attrs.setdefault(key, value)
        merged["attributes"] = merged_attrs

        base_type = str(base_entity.get("type", "")).strip()
        incoming_type = str(incoming_entity.get("type", "")).strip()
        if (not base_type or base_type.lower() == "unknown") and incoming_type:
            merged["type"] = incoming_type

        merged["aliases"] = self._normalize_aliases(
            []
            + base_entity.get("aliases", [])
            + incoming_entity.get("aliases", [])
            + [base_entity.get("name", ""), incoming_entity.get("name", "")],
            canonical_name,
        )
        return merged

    def build_related_kg_context(self, graph_data, entity_name, max_relations=20):
        # Build 1-hop related KG context for a target entity to feed relation extraction prompt.
        graph = self._normalize_graph_input(graph_data)
        if not entity_name:
            return None

        entity_registry = {}
        for entity in graph.get("entities", []):
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            entity_registry[name] = {
                "name": name,
                "type": entity.get("type", "Unknown"),
                "description": entity.get("description", ""),
                "attributes": entity.get("attributes", {}) if isinstance(entity.get("attributes"), dict) else {},
                "aliases": self._normalize_aliases(entity.get("aliases", []), name),
            }

        alias_to_canonical = self._build_alias_to_canonical(entity_registry)
        canonical = self._resolve_name_by_alias(entity_name, alias_to_canonical)
        if not canonical or canonical not in entity_registry:
            return None

        related_relations = []
        for rel in self._iter_normalized_relations(graph):
            source = self._resolve_name_by_alias(rel["source"], alias_to_canonical)
            target = self._resolve_name_by_alias(rel["target"], alias_to_canonical)
            if source == canonical or target == canonical:
                related_relations.append(
                    {
                        "source": source,
                        "relation": rel["relation"],
                        "target": target,
                        "description": rel.get("description", ""),
                    }
                )
            if len(related_relations) >= max_relations:
                break

        return {
            "central_entity": entity_registry[canonical],
            "related_relations": related_relations,
        }

    def merge_knowledge_graphs(self, base_graph, new_graph):
        # Merge two graphs and normalize aliases before deduping entities/relations.
        base_graph = self._normalize_graph_input(base_graph)
        new_graph = self._normalize_graph_input(new_graph)

        entity_registry = {}
        for entity in base_graph.get("entities", []):
            if not isinstance(entity, dict):
                continue
            canonical_name = str(entity.get("name", "")).strip()
            if not canonical_name:
                continue
            entity_registry[canonical_name] = {
                "name": canonical_name,
                "type": entity.get("type", "Unknown"),
                "description": entity.get("description", ""),
                "attributes": entity.get("attributes", {}) if isinstance(entity.get("attributes"), dict) else {},
                "aliases": self._normalize_aliases(entity.get("aliases", []), canonical_name),
            }

        def merge_entity_into_registry(entity_item):
            if not isinstance(entity_item, dict):
                return
            incoming_name = str(entity_item.get("name", "")).strip()
            if not incoming_name:
                return

            alias_to_canonical_local = self._build_alias_to_canonical(entity_registry)
            resolved_name = self._resolve_name_by_alias(incoming_name, alias_to_canonical_local)

            if resolved_name in entity_registry:
                target_name = resolved_name
            else:
                target_name = None
                for alias in self._normalize_aliases(entity_item.get("aliases", []), incoming_name):
                    resolved_alias = self._resolve_name_by_alias(alias, alias_to_canonical_local)
                    if resolved_alias in entity_registry:
                        target_name = resolved_alias
                        break
                if not target_name:
                    target_name = incoming_name

            incoming_normalized = {
                "name": target_name,
                "type": entity_item.get("type", "Unknown"),
                "description": entity_item.get("description", ""),
                "attributes": entity_item.get("attributes", {})
                if isinstance(entity_item.get("attributes"), dict)
                else {},
                "aliases": self._normalize_aliases(entity_item.get("aliases", []), target_name),
            }

            existing = entity_registry.get(target_name)
            if existing is None:
                entity_registry[target_name] = incoming_normalized
            else:
                entity_registry[target_name] = self._merge_entity_records(existing, incoming_normalized)

        for entity in new_graph.get("entities", []):
            merge_entity_into_registry(entity)

        alias_to_canonical = self._build_alias_to_canonical(entity_registry)
        relation_registry = {}

        def merge_relation_into_registry(rel_item):
            source = self._resolve_name_by_alias(rel_item.get("source"), alias_to_canonical)
            target = self._resolve_name_by_alias(rel_item.get("target"), alias_to_canonical)
            relation = str(rel_item.get("relation", "")).strip()
            if not source or not target or not relation:
                return

            if source not in entity_registry:
                entity_registry[source] = {
                    "name": source,
                    "type": "Unknown",
                    "description": "",
                    "attributes": {},
                    "aliases": [],
                }
            if target not in entity_registry:
                entity_registry[target] = {
                    "name": target,
                    "type": "Unknown",
                    "description": "",
                    "attributes": {},
                    "aliases": [],
                }

            rel_key = (source, self._normalize_relation_text(relation), target)
            description = str(rel_item.get("description", "")).strip()
            if rel_key not in relation_registry:
                relation_registry[rel_key] = {
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "description": description,
                }
                return

            relation_registry[rel_key]["description"] = self._merge_semicolon_text(
                relation_registry[rel_key].get("description", ""),
                description,
            )

        for relation in self._iter_normalized_relations(base_graph):
            merge_relation_into_registry(relation)
        for relation in self._iter_normalized_relations(new_graph):
            merge_relation_into_registry(relation)

        merged_graph = {"entities": list(entity_registry.values()), "relations": []}
        for rel_data in relation_registry.values():
            merged_graph["relations"].append(
                [
                    rel_data["source"],
                    rel_data["relation"],
                    rel_data["target"],
                    rel_data.get("description", ""),
                ]
            )
        return merged_graph
