"""QA graph neighbor expansion and context building."""

from collections import deque


class PipelineQAContextOpsMixin:
    """Graph traversal and evidence context assembly for QA."""

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
            chunk_map = graph_index.get("chunk_map", {})
        else:
            if graph_data is None:
                raise ValueError("Either graph_data or graph_index must be provided.")
            normalized_graph = self._normalize_knowledge_graph(graph_data)
            entity_lookup, _, adjacency_out, adjacency_in = self._build_graph_indices(normalized_graph)
            chunk_map = normalized_graph.get("chunk_map", {})

        if not isinstance(chunk_map, dict):
            chunk_map = {}
        seed_nodes = seed_nodes or []
        hop_limit = 1 if max_hop <= 1 else 2

        chunk_evidence = []
        entity_evidence = []
        relation_evidence = []
        graph_paths = []
        visited_path = set()
        visited_chunk_source = set()
        visited_entity_source = set()
        visited_relation_source = set()

        def add_chunk_evidence(chunk_ids, anchor, path="", hop=0):
            normalized_chunk_ids = self._normalize_chunk_ids(chunk_ids)
            for chunk_id in normalized_chunk_ids[:8]:
                source = f"chunk:{chunk_id}"
                if source in visited_chunk_source:
                    continue
                sentence = chunk_map.get(chunk_id, "")
                if not isinstance(sentence, str) or not sentence.strip():
                    continue
                chunk_evidence.append(
                    {
                        "source": source,
                        "text": sentence.strip(),
                        "path": path,
                        "hop": hop,
                        "anchor": anchor,
                    }
                )
                visited_chunk_source.add(source)

        def add_entity_evidence(node_name):
            entity = entity_lookup.get(node_name, {"name": node_name})
            entity_prov = entity.get("provenance", {})
            chunk_ids = entity_prov.get("chunk_ids", []) if isinstance(entity_prov, dict) else []
            add_chunk_evidence(
                chunk_ids=chunk_ids,
                anchor=f"entity:{node_name}",
                path=node_name,
                hop=0,
            )

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
                        rel_prov = rel.get("provenance", {})
                        rel_chunk_ids = rel_prov.get("chunk_ids", []) if isinstance(rel_prov, dict) else []
                        add_chunk_evidence(
                            chunk_ids=rel_chunk_ids,
                            anchor=rel_source,
                            path=new_path,
                            hop=current_depth + 1,
                        )
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
                        rel_prov = rel.get("provenance", {})
                        rel_chunk_ids = rel_prov.get("chunk_ids", []) if isinstance(rel_prov, dict) else []
                        add_chunk_evidence(
                            chunk_ids=rel_chunk_ids,
                            anchor=rel_source,
                            path=new_path,
                            hop=current_depth + 1,
                        )
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
            "chunk_evidence": chunk_evidence,
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
        for item in expanded["chunk_evidence"]:
            evidence_items.append(
                {
                    "source": item["source"],
                    "text": item["text"],
                    "path": item.get("path", ""),
                }
            )
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
