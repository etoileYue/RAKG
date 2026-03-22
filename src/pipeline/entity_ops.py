"""实体规范化、去重与消歧能力。"""

from copy import deepcopy

class PipelineEntityOpsMixin:
    """实体规范化、去重与消歧方法集合。"""

    def _split_semicolon_values(self, value)->list:
        """将由;;;分隔的字符串分割为list"""
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

    def _merge_semicolon_text(self, left, right)->str:
        """将list(str)通过;;;拼接"""
        merged = self._dedupe_preserve_order(
            self._split_semicolon_values(left) + self._split_semicolon_values(right)
        )
        return ";;;".join(merged)

    def _normalize_aliases(self, aliases, canonical_name):
        """对别名aliases列表进行清洗、规范化和去重处理"""
        if aliases is None:
            aliases = []
        if not isinstance(aliases, list):
            aliases = [aliases]

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

    def ensure_entity_aliases(self, entity_dic)->dict:
        """确保entity含有aliases字段,没有则添加"""
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

    def _collapse_entities_by_name(self, entity_dic)->dict:
        """合并同名实体"""
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
                    "chunkid": entity.get("chunkid", []),
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
            current["chunkid"] = self._dedupe_preserve_order(
                current.get("chunkid", []) + entity.get("chunkid", [])
            )
            current["aliases"] = self._normalize_aliases(
                current.get("aliases", []) + entity.get("aliases", []),
                name
            )

        collapsed = {}
        for idx, (_, entity) in enumerate(grouped.items(), start=1):
            collapsed[f"entity{idx}"] = entity
        return collapsed

    def entity_Disambiguation(self, entity_dic, sim_entity_list):
        """使用并查集将两两一组的相似实体分类，分类后合并"""
        entity_dic = self.ensure_entity_aliases(entity_dic)
        parent = {}

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
                chunkids.extend(entity_dic[entity].get("chunkid", []))
                aliases.append(entity_dic[entity].get("name", ""))
                aliases.extend(entity_dic[entity].get("aliases", []))
                if entity != main_entity:
                    del entity_dic[entity]

            dedup_descriptions = self._dedupe_preserve_order(descriptions)
            dedup_chunkids = self._dedupe_preserve_order(chunkids)
            entity_dic[main_entity]["description"] = ";;;".join(dedup_descriptions)
            entity_dic[main_entity]["chunkid"] = dedup_chunkids
            entity_dic[main_entity]["aliases"] = self._normalize_aliases(
                aliases,
                entity_dic[main_entity].get("name", ""),
            )
        return entity_dic
