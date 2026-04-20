"""图谱规范化、对齐与合并能力。"""

from copy import deepcopy
import json
import os
import re
from src.pipeline.shared import logger

class PipelineGraphOpsMixin:
    """图谱规范化、对齐与合并方法集合。"""

    def _normalize_relation_text(self, relation_text)->str:
        """功能说明：_normalize_relation_text。"""
        if relation_text is None:
            return ""
        relation = str(relation_text).strip().lower()
        if not relation:
            return ""
        return " ".join(relation.split())

    def _normalize_chunk_ids(self, chunk_ids)->list:
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
        return self._dedupe_preserve_order(normalized)

    def _normalize_chunk_map(self, chunk_map)->dict:
        if not isinstance(chunk_map, dict):
            return {}

        normalized = {}
        for key, value in chunk_map.items():
            chunk_id = str(key).strip()
            sentence = str(value).strip() if value is not None else ""
            if chunk_id and sentence and chunk_id not in normalized:
                normalized[chunk_id] = sentence
        return normalized

    def _normalize_provenance(self, provenance)->dict:
        if not isinstance(provenance, dict):
            provenance = {}
        normalized = {
            "chunk_ids": self._normalize_chunk_ids(provenance.get("chunk_ids", [])),
        }
        strategy = str(provenance.get("strategy", "")).strip()
        if strategy:
            normalized["strategy"] = strategy
        confidence = provenance.get("confidence", None)
        if isinstance(confidence, (int, float)):
            normalized["confidence"] = float(confidence)
        elif isinstance(confidence, str):
            text = confidence.strip()
            if text:
                try:
                    normalized["confidence"] = float(text)
                except ValueError:
                    pass
        return normalized

    def _merge_provenance(self, left, right)->dict:
        left_norm = self._normalize_provenance(left)
        right_norm = self._normalize_provenance(right)
        merged = {
            "chunk_ids": self._dedupe_preserve_order(
                left_norm.get("chunk_ids", []) + right_norm.get("chunk_ids", [])
            )
        }
        strategy = str(right_norm.get("strategy") or left_norm.get("strategy") or "").strip()
        if strategy:
            merged["strategy"] = strategy
        if "confidence" in right_norm:
            merged["confidence"] = right_norm["confidence"]
        elif "confidence" in left_norm:
            merged["confidence"] = left_norm["confidence"]
        return merged

    def _extract_text_tokens(self, text, max_tokens=16)->list:
        if text is None:
            return []
        matches = re.findall(r"[\u4e00-\u9fff]{1,}|[A-Za-z0-9]{2,}", str(text).lower())
        tokens = self._dedupe_preserve_order([item.strip() for item in matches if item.strip()])
        return tokens[:max_tokens]

    def _infer_relation_provenance(
        self,
        source_name,
        target_name,
        relation_text,
        relation_description,
        candidate_chunks,
        fallback_chunk_ids,
    )->dict:
        candidate_chunks = self._normalize_chunk_map(candidate_chunks)
        fallback_chunk_ids = self._normalize_chunk_ids(fallback_chunk_ids)
        if not candidate_chunks:
            if fallback_chunk_ids:
                return {"chunk_ids": fallback_chunk_ids, "strategy": "fallback_entity_chunks"}
            return {"chunk_ids": [], "strategy": "missing_candidates"}

        source_lower = str(source_name or "").strip().lower()
        target_lower = str(target_name or "").strip().lower()
        rel_tokens = self._dedupe_preserve_order(
            self._extract_text_tokens(relation_text, max_tokens=8)
            + self._extract_text_tokens(relation_description, max_tokens=20)
        )

        matched = []
        soft_matched = []
        for chunk_id, sentence in candidate_chunks.items():
            sent_lower = str(sentence).lower()
            source_hit = bool(source_lower) and source_lower in sent_lower
            target_hit = bool(target_lower) and target_lower in sent_lower
            token_hits = 0
            for token in rel_tokens:
                if token and token in sent_lower:
                    token_hits += 1

            if source_hit and target_hit:
                matched.append(chunk_id)
                continue
            if (source_hit or target_hit) and token_hits >= 1:
                soft_matched.append(chunk_id)
                continue
            if token_hits >= 2:
                soft_matched.append(chunk_id)

        selected = self._dedupe_preserve_order(matched if matched else soft_matched)[:5]
        if selected:
            return {"chunk_ids": selected, "strategy": "heuristic_sentence_match"}
        if fallback_chunk_ids:
            return {"chunk_ids": fallback_chunk_ids, "strategy": "fallback_entity_chunks"}
        return {"chunk_ids": [], "strategy": "missing_candidates"}

    def _normalize_graph_input(self, graph_data)->dict:
        """
        载入graph_data,如果是dict则返回,是文件路径则加载
        return {"entities": entities, "relations": relations}
        """
        if graph_data is None:
            return {"entities": [], "relations": [], "chunk_map": {}}

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
        chunk_map = data.get("chunk_map", {})
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError("graph_data must contain list fields: entities and relations.")
        return {
            "entities": entities,
            "relations": relations,
            "chunk_map": self._normalize_chunk_map(chunk_map),
        }

    def _build_existing_entity_lookup(self, existing_graph)->dict:
        """把已有图数据中的实体entities整理成一个“标准化 + 可索引”的查找字典"""
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
        """将新实体对齐到已有图谱中的实体"""
        # new_entities = self.ensure_entity_aliases(new_entities)
        if not new_entities:
            return {}, {}

        existing_entities = self._build_existing_entity_lookup(existing_graph)
        # if not existing_entities:
        #     return self._collapse_entities_by_name(new_entities), {}

        matched_map = self.cross_similarity_result(
            left_entities=new_entities,
            right_entities=existing_entities,
            threshold=threshold,
            gray_margin=gray_margin,
        )

        aligned_entities = {}
        alias_resolution = {} #别名解析表，后续如果关系里提到的是别名，也能统一替换成标准名
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

    def convert_knowledge_graph(self, input_data, chunk_map=None):
        """转换为entities + relations组织的知识图谱"""
        output = {
            "entities": [],
            "relations": [],
            "chunk_map": self._normalize_chunk_map(chunk_map),
        }
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

            node_provenance = node_data.get("_provenance", {})
            if not isinstance(node_provenance, dict):
                node_provenance = {}

            node_candidate_chunks = self._normalize_chunk_map(node_provenance.get("candidate_chunks", {}))
            for chunk_id, sentence in node_candidate_chunks.items():
                output["chunk_map"].setdefault(chunk_id, sentence)

            entity_name = central_entity.get("name")
            entity_type = central_entity.get("type", "Unknown")
            if not entity_name:
                logger.warning("Skip malformed central_entity for %s: missing name.", entity_key)
                continue

            central_provenance = self._normalize_provenance(central_entity.get("provenance", {}))
            node_entity_chunk_ids = self._normalize_chunk_ids(node_provenance.get("entity_chunk_ids", []))
            entity_provenance = self._merge_provenance(
                central_provenance,
                {"chunk_ids": node_entity_chunk_ids},
            )

            if entity_name not in entity_registry:
                entity = {
                    "name": entity_name,
                    "type": entity_type,
                    "description": central_entity.get("description", ""),
                    "attributes": {},
                    "aliases": self._normalize_aliases(central_entity.get("aliases", []), entity_name),
                    "provenance": entity_provenance,
                }
                if "attributes" in central_entity:
                    for attr_index, attr in enumerate(central_entity["attributes"]):
                        try:
                            entity["attributes"][attr["key"]] = attr["value"]
                        except Exception as exc:
                            raise ValueError(
                                "Malformed central_entity.attributes item while converting knowledge graph: "
                                f"entity_key={entity_key}, entity_name={entity_name}, "
                                f"attr_index={attr_index}, attr={repr(attr)}"
                            ) from exc
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
                existing_entity["provenance"] = self._merge_provenance(
                    existing_entity.get("provenance", {}),
                    entity_provenance,
                )
                if "attributes" in central_entity:
                    for attr_index, attr in enumerate(central_entity["attributes"]):
                        try:
                            key = attr.get("key")
                            value = attr.get("value")
                            if key is None or value is None:
                                continue
                            existing_entity["attributes"].setdefault(key, value)
                        except Exception as exc:
                            raise ValueError(
                                "Malformed central_entity.attributes item while merging knowledge graph entity: "
                                f"entity_key={entity_key}, entity_name={entity_name}, "
                                f"attr_index={attr_index}, attr={repr(attr)}"
                            ) from exc

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

            node_provenance = node_data.get("_provenance", {})
            if not isinstance(node_provenance, dict):
                node_provenance = {}
            candidate_chunk_ids = self._normalize_chunk_ids(node_provenance.get("candidate_chunk_ids", []))
            candidate_chunks = self._normalize_chunk_map(node_provenance.get("candidate_chunks", {}))
            relation_candidates = {}
            for chunk_id in candidate_chunk_ids:
                sentence = candidate_chunks.get(chunk_id) or output["chunk_map"].get(chunk_id)
                if isinstance(sentence, str) and sentence.strip():
                    relation_candidates[chunk_id] = sentence.strip()
            if not relation_candidates:
                relation_candidates = candidate_chunks
            for chunk_id, sentence in relation_candidates.items():
                output["chunk_map"].setdefault(chunk_id, sentence)

            source_entity = entity_registry.get(source_name, {})
            fallback_source_chunks = self._normalize_chunk_ids(
                source_entity.get("provenance", {}).get("chunk_ids", [])
            )

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
                relation_description = rel.get("relation_description", "")
                relation_name = rel["relation"]
                relation_provenance_raw = rel.get("provenance", {})
                relation_provenance_raw = (
                    relation_provenance_raw if isinstance(relation_provenance_raw, dict) else {}
                )

                for target_name in target_names:
                    llm_relation_provenance = self._normalize_provenance(relation_provenance_raw)
                    if llm_relation_provenance.get("chunk_ids"):
                        relation_provenance = dict(llm_relation_provenance)
                        relation_provenance.setdefault("strategy", "llm_relation_provenance")
                    else:
                        relation_provenance = self._infer_relation_provenance(
                            source_name=source_name,
                            target_name=target_name,
                            relation_text=relation_name,
                            relation_description=relation_description,
                            candidate_chunks=relation_candidates,
                            fallback_chunk_ids=fallback_source_chunks,
                        )
                        if "confidence" in llm_relation_provenance:
                            relation_provenance["confidence"] = llm_relation_provenance["confidence"]

                    if target_name not in entity_registry:
                        entity_registry[target_name] = {
                            "name": target_name,
                            "type": target_type,
                            "description": rel.get("target_description", ""),
                            "attributes": {},
                            "aliases": [],
                            "provenance": {"chunk_ids": relation_provenance.get("chunk_ids", [])},
                        }
                    else:
                        target_entity = entity_registry[target_name]
                        target_entity["description"] = self._merge_semicolon_text(
                            target_entity.get("description", ""),
                            rel.get("target_description", ""),
                        )
                        target_entity["provenance"] = self._merge_provenance(
                            target_entity.get("provenance", {}),
                            {"chunk_ids": relation_provenance.get("chunk_ids", [])},
                        )

                    output["relations"].append(
                        {
                            "source": source_name,
                            "relation": relation_name,
                            "target": target_name,
                            "description": relation_description,
                            "provenance": relation_provenance,
                        }
                    )

        output["entities"] = list(entity_registry.values())
        return output

    def _iter_normalized_relations(self, graph_data):
        """逐条返回graph_data["relations"]"""
        for rel in graph_data.get("relations", []):
            source = None
            relation = None
            target = None
            description = ""
            provenance = {}

            if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                source, relation, target = rel[0], rel[1], rel[2]
                if len(rel) >= 4:
                    description = rel[3] or ""
                if len(rel) >= 5 and isinstance(rel[4], dict):
                    provenance = rel[4]
            elif isinstance(rel, dict):
                source = rel.get("source")
                relation = rel.get("relation")
                target = rel.get("target")
                description = rel.get("description", "") or rel.get("rel_description", "")
                provenance = rel.get("provenance", {})

            if not source or not relation or not target:
                continue

            yield {
                "source": str(source).strip(),
                "relation": str(relation).strip(),
                "target": str(target).strip(),
                "description": str(description or "").strip(),
                "provenance": self._normalize_provenance(provenance),
            }

    def _build_alias_to_canonical(self, entity_registry)->dict:
        """构建从别名到标准实体名canonical name的映射表"""
        alias_to_canonical = {}
        for canonical_name, entity in entity_registry.items():
            alias_to_canonical[canonical_name] = canonical_name
            aliases = self._normalize_aliases(entity.get("aliases", []), canonical_name)
            for alias in aliases:
                alias_to_canonical[str(alias)] = canonical_name
        return alias_to_canonical

    def _resolve_name_by_alias(self, raw_name, alias_to_canonical)->str:
        """获得别名对应的实体名"""
        if raw_name is None:
            return None
        name = str(raw_name).strip()
        if not name:
            return None
        return alias_to_canonical.get(name, name)

    def _merge_entity_records(self, base_entity, incoming_entity):
        """将两个表示“同一实体”的记录合并成一个统一的实体对象"""
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
            "provenance": self._merge_provenance(
                base_entity.get("provenance", {}),
                incoming_entity.get("provenance", {}),
            ),
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

    def build_related_kg_context(self, graph_data, entity_name, max_relations=20)->dict:
        """从知识图谱里，围绕某个指定实体，提取它的标准化实体信息和与它直接相关的关系上下文"""
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
                "provenance": self._normalize_provenance(entity.get("provenance", {})),
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
                        "provenance": rel.get("provenance", {"chunk_ids": []}),
                    }
                )
            if len(related_relations) >= max_relations:
                break

        return {
            "central_entity": entity_registry[canonical],
            "related_relations": related_relations,
        }

    def merge_knowledge_graphs(self, base_graph, new_graph):
        """把两个知识图谱base_graph和new_graph合并成一个统一图谱"""
        base_graph = self._normalize_graph_input(base_graph)
        new_graph = self._normalize_graph_input(new_graph)
        merged_chunk_map = self._normalize_chunk_map(base_graph.get("chunk_map", {}))
        for chunk_id, sentence in self._normalize_chunk_map(new_graph.get("chunk_map", {})).items():
            merged_chunk_map.setdefault(chunk_id, sentence)

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
                "provenance": self._normalize_provenance(entity.get("provenance", {})),
            }

        def merge_entity_into_registry(entity_item):
            """功能说明：merge_entity_into_registry。"""
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
                "provenance": self._normalize_provenance(entity_item.get("provenance", {})),
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
            """功能说明：merge_relation_into_registry。"""
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
                    "provenance": {"chunk_ids": []},
                }
            if target not in entity_registry:
                entity_registry[target] = {
                    "name": target,
                    "type": "Unknown",
                    "description": "",
                    "attributes": {},
                    "aliases": [],
                    "provenance": {"chunk_ids": []},
                }

            rel_key = (source, self._normalize_relation_text(relation), target)
            description = str(rel_item.get("description", "")).strip()
            provenance = self._normalize_provenance(rel_item.get("provenance", {}))
            if rel_key not in relation_registry:
                relation_registry[rel_key] = {
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "description": description,
                    "provenance": provenance,
                }
                return

            relation_registry[rel_key]["description"] = self._merge_semicolon_text(
                relation_registry[rel_key].get("description", ""),
                description,
            )
            relation_registry[rel_key]["provenance"] = self._merge_provenance(
                relation_registry[rel_key].get("provenance", {}),
                provenance,
            )

        for relation in self._iter_normalized_relations(base_graph):
            merge_relation_into_registry(relation)
        for relation in self._iter_normalized_relations(new_graph):
            merge_relation_into_registry(relation)

        merged_graph = {
            "entities": list(entity_registry.values()),
            "relations": [],
            "chunk_map": merged_chunk_map,
        }
        for rel_data in relation_registry.values():
            merged_graph["relations"].append(
                {
                    "source": rel_data["source"],
                    "relation": rel_data["relation"],
                    "target": rel_data["target"],
                    "description": rel_data.get("description", ""),
                    "provenance": self._normalize_provenance(rel_data.get("provenance", {})),
                }
            )
        return merged_graph
