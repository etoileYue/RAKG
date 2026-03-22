"""图谱规范化、对齐与合并能力。"""

from copy import deepcopy
import json
import os
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

    def _normalize_graph_input(self, graph_data)->dict:
        """
        载入graph_data,如果是dict则返回,是文件路径则加载
        return {"entities": entities, "relations": relations}
        """
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

    def convert_knowledge_graph(self, input_data):
        """转换为entities + relations组织的知识图谱"""
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
        """逐条返回graph_data["relations"]"""
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
        """把两个知识图谱base_graph和new_graph合并成一个统一图谱"""
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
