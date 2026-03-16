from langchain_core.prompts import ChatPromptTemplate
from src.prompt import text2entity_en
from src.prompt import extract_entiry_centric_kg_en_v2
from src.prompt import judge_sim_entity_en
from src.prompt import question_entity_extract_prompt_cn
from src.prompt import kg_qa_answer_prompt_cn
from itertools import combinations
from collections import deque
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.llm_provider import LLMProvider
import os
import re
import hashlib

import logging
from src.logger import get_logger
import traceback
from src.utils  import retry
from src.utils import parse_similarity_response

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

logger = get_logger(name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
                    level=logging.INFO,
                    log_file="Agent.log")

debug_logger = get_logger(name=f"{os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME)}.debug",
                          level=logging.DEBUG,
                          log_file="Debug.log")

class NER_Agent():
    def __init__(self):
        self.llm_provider = LLMProvider()
        self.model = self.llm_provider.get_llm()
        self.similarity_model = self.llm_provider.get_similarity_model()
        self.embeddings = self.llm_provider.get_embedding_model()
        self.last_disambiguation_gray_queue = []
        # QA 图谱索引缓存：key -> {entity_lookup, adjacency_out/in, node_vectors...}
        self._qa_graph_index_cache = {}
        self._qa_default_graph_cache_key = None

    def _ensure_parent_dir(self, output_file):
        parent_dir = os.path.dirname(output_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def _append_jsonl(self, output_file, data):
        self._ensure_parent_dir(output_file)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def _dedupe_preserve_order(self, items):
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _is_boundary_candidate(self, similarity_score, threshold, gray_margin):
        return similarity_score <= (threshold + gray_margin)
    
    ## Add chunkid attribute
    def add_chunkid(self, ner_result, chunkid):
        new_ner_result = {}
        for entity_key, entity_value in ner_result.items():
            entity_value["chunkid"] = chunkid
            new_ner_result[entity_key] = entity_value
        return new_ner_result
    
    def extract_from_text_single(self, text_single, output_file):
        prompt = ChatPromptTemplate.from_template(text2entity_en)
        chain = prompt | self.model
        result = chain.invoke({"text": text_single})
        debug_logger.debug("-extract_from_text_single-")
        debug_logger.debug(f"text_single={text_single}, result={result}")
        if hasattr(result, 'content'):
            result_json = json.loads(result.content)
        else:
            result_json = json.loads(result)
        
        # Store text_single and result_json in a jsonl file
        combined_data = {
            "text": text_single,
            "entities": result_json
        }
        self._append_jsonl(output_file, combined_data)
        
        return result_json
    
    def rewrite(self, ner_result, entity_num):
        new_entities = {}
        # Process in original dictionary key order, extract numbers after entity and renumber
        for idx, (old_key, value) in enumerate(ner_result.items(), start=1):
            new_key = f"entity{entity_num + idx - 1}"
            new_entities[new_key] = value
        return new_entities
    
    ## Implement named entity recognition for the entire text and add chunkid field to each entity
    def extract_from_text_multiply(self, text_list, sent_to_id,output_file):
        ner_result_for_all = {}
        entity_num = 1
        for text in text_list:
            ner_result = self.extract_from_text_single(text,output_file)
            ## Add a check here - if ner_result has a state field, it means there's an issue with this chunk, so skip to the next iteration
            if 'State' in ner_result:
                continue
            # Get the number of entities in ner_result
            ner_result_num = len(ner_result)
            # Rewrite ner_result, entity numbering starts from entity_num, first entity is entity{entity_num}, subsequent entities increment
            ner_result = self.rewrite(ner_result, entity_num)

            chunkid = sent_to_id.get(text)
            if chunkid is None:
                logger.warning("Sentence not found in sentence_to_id mapping, skipping chunk.")
                continue
            entity_num += ner_result_num
            ner_result_with_chunkid = self.add_chunkid(ner_result,chunkid)
            ner_result_for_all.update(ner_result_with_chunkid)
        return ner_result_for_all
    
    
    def similarity_candidates(self, entities, threshold=0.60):
        def get_embedding_vector(text):
            result = self.embeddings.embed_documents([text])
            if isinstance(result, list) and isinstance(result[0], list):
                return result[0]
            else:
                return result
        entity_texts = {
            k: f"{v['name']} {v['type']}"
            for k, v in entities.items()
        }
        vectors = {
            k: get_embedding_vector(text)
            for k, text in entity_texts.items()
        }

        keys = list(vectors.keys())
        sim_matrix = np.zeros((len(keys), len(keys)))

        for i, j in combinations(range(len(keys)), 2):
            sim = float(cosine_similarity([vectors[keys[i]]], [vectors[keys[j]]])[0][0])
            # print(f"Similarity between {keys[i]} and {keys[j]}: {sim}")
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

    def similarity_result(self, entities, threshold=0.60, gray_margin=0.05):
        # Step 1: Use similarity_candidates for initial filtering
        candidates = self.similarity_candidates(entities, threshold=threshold)

        # Step 2: Fine-grained LLM judgment for each candidate pair, and queue gray samples
        candidates_result = []
        gray_queue = []
        for ent_pair in candidates:
            left_id, right_id, score = ent_pair
            # Extract entity objects from entities dictionary
            entity1 = entities.get(left_id)
            entity2 = entities.get(right_id)

            # Call LLM for judgment
            try:
                result = self.similarity_llm_single(entity1, entity2)
                is_boundary = self._is_boundary_candidate(score, threshold, gray_margin)
                needs_review = result.get("needs_review", False) or is_boundary

                if needs_review:
                    gray_queue.append({
                        "pair": (left_id, right_id),
                        "similarity_score": score,
                        "reason": result.get("reason", "unknown"),
                        "parse_status": result.get("parse_status", "unknown"),
                        "first_pass_result": result,
                    })
                    continue

                # Keep if LLM judges as same entity
                if result.get('result', False):
                    candidates_result.append((left_id, right_id))
            except Exception as e:
                logger.error(f"Error processing entity pair {ent_pair}: {traceback.format_exc()}")
                # print(f"Error processing entity pair {ent_pair}: {traceback.format_exc()}")
                continue  # Can log or raise exception as needed

        # Step 3: Re-check gray queue
        resolved_by_second_pass = 0
        for item in gray_queue:
            left_id, right_id = item["pair"]
            entity1 = entities.get(left_id)
            entity2 = entities.get(right_id)
            try:
                # Reuse the original disambiguation prompt for second-pass review.
                second_pass_result = self.similarity_llm_single(entity1, entity2)
                if second_pass_result.get("result", False):
                    candidates_result.append((left_id, right_id))
                    resolved_by_second_pass += 1
                item["second_pass_result"] = second_pass_result
            except Exception:
                logger.error(
                    "Error in second-pass disambiguation for pair "
                    f"{(left_id, right_id)}: {traceback.format_exc()}"
                )

        self.last_disambiguation_gray_queue = gray_queue
        logger.info(
            "Disambiguation summary: candidates=%s, gray_queue=%s, "
            "resolved_by_second_pass=%s, merged_pairs=%s",
            len(candidates),
            len(gray_queue),
            resolved_by_second_pass,
            len(candidates_result),
        )

        # Step 4: Return final filtered candidate pairs
        return candidates_result

    ## Merge similar items
    def entity_Disambiguation(self, entity_dic, sim_entity_list):
        # Step 1: Build and manage merge relationships using union-find
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

        # Initialize union-find
        for entity in entity_dic:
            parent[entity] = entity
        for pair in sim_entity_list:
            a, b = pair
            if a in entity_dic and b in entity_dic:
                union(a, b)

        # Step 2: Merge similar entities
        groups = {}
        for entity in entity_dic:
            root = find(entity)
            if root not in groups:
                groups[root] = []
            groups[root].append(entity)

        # Step 3: Process each merge group
        for group in groups.values():
            if len(group) == 1:
                continue

            # Sort by appearance order, keep name/type of first entity
            main_entity = group[0]
            descriptions = []
            chunkids = []

            for e in group:
                descriptions.append(entity_dic[e]['description'])
                chunkids.append(entity_dic[e]['chunkid'])
                if e != main_entity:
                    del entity_dic[e]  # Remove merged entities

            # Merge fields
            dedup_descriptions = self._dedupe_preserve_order(descriptions)
            dedup_chunkids = self._dedupe_preserve_order(chunkids)
            entity_dic[main_entity]['description'] = ';;;'.join(dedup_descriptions)
            entity_dic[main_entity]['chunkid'] = ';;;'.join(dedup_chunkids)

        # Step 4: Directly return merged entity dictionary
        return entity_dic

    def get_sentences_for_entity(self,entity_dic, entity_id, id_to_sentence):
        """
        Extract sentences corresponding to chunkid for a specified entity ID.

        Parameters:
            entity_dic (dict): Dictionary containing entity information.
            entity_id (str): Specified entity ID.
            id_to_sentence (dict): Dictionary mapping chunkid to sentences.

        Returns:
            list: List of sentences corresponding to the specified entity's chunkid.
        """
        if entity_id not in entity_dic:
            raise ValueError(f"Entity '{entity_id}' not found in entity_dic.")

        # Get chunkid field for specified entity
        chunkids = entity_dic[entity_id].get('chunkid', '')
        if not chunkids:
            return []  # Return empty list if no chunkid

        # Split chunkid into multiple IDs
        chunkid_list = chunkids.split(';;;')
        chunkid_list = [cid.strip() for cid in chunkid_list if cid.strip()]  # Remove empty strings

        # Find corresponding sentences from id_to_sentence
        sentences = []
        for chunkid in chunkid_list:
            if chunkid in id_to_sentence:
                sentences.append(id_to_sentence[chunkid])
            else:
                logger.warning(f"Chunk ID '{chunkid}' not found in id_to_sentence.")
                # print(f"Warning: Chunk ID '{chunkid}' not found in id_to_sentence.")

        return sentences

    def get_retriever_context(self, query, sentences, sentence_to_id,vectors,top_k=5):
        """
        Get the top_k most similar sentences as retriever context for a query.

        :param query: str, user's query text
        :param top_k: int, number of most similar sentences to return, default is 5
        :return: list of tuples, each tuple contains (sentence, similarity, sentence_id)
        """

        if not sentences or not vectors:
            return []

        # Step 1: Convert query to vector
        query_vector = self.embeddings.embed_query(query)

        # Step 2: Calculate cosine similarity between query vector and sentence vectors
        sentence_vectors = np.array(vectors)
        try:
            similarities = cosine_similarity([query_vector], sentence_vectors)[0]
        except Exception:
            logger.error("Failed to calculate sentence similarity: %s", traceback.format_exc())
            return []

        # Step 3: Select top_k most similar sentences
        top_indices = np.argsort(similarities)[::-1][:top_k]  # Sort by similarity in descending order and take top_k
        retriever_context = []
        for idx in top_indices:
            sentence = sentences[idx]
            similarity = similarities[idx]
            sentence_id = sentence_to_id[sentence]
            retriever_context.append((sentence, similarity, sentence_id))

        return retriever_context

    def get_target_kg_single(self, entity_dic, entity_id, id_to_sentence, sentences, sentence_to_id, vectors, output_file):
        chunk_text_list = self.get_sentences_for_entity(entity_dic, entity_id, id_to_sentence)
        query = entity_dic[entity_id].get('name', '')
        context = self.get_retriever_context(query, sentences, sentence_to_id, vectors, top_k=5)
        sentences = [item[0] for item in context]
        unique_sentences = self._dedupe_preserve_order(chunk_text_list + sentences)
        chunk_text = ", ".join(unique_sentences)
        prompt = ChatPromptTemplate.from_template(extract_entiry_centric_kg_en_v2)
        chain = prompt | self.model
        result = chain.invoke({"text": chunk_text, "target_entity": entity_dic[entity_id].get('name'), "related_kg": 'none'})
        debug_logger.debug("-get_target_kg_single-")
        debug_logger.debug(f"text={chunk_text}, target_entity={entity_dic[entity_id].get('name')}, related_kg=none, result={result}")
        # Handle AIMessage response from OpenAI
        if hasattr(result, 'content'):
            result_json = json.loads(result.content)
        else:
            result_json = json.loads(result)

        combined_data = {
            "chunk_text": chunk_text,
            "entity": entity_dic[entity_id],
            "kg": result_json
        }
        self._append_jsonl(output_file, combined_data)

        return result_json
    
    def get_target_kg_all(self, entity_dic, id_to_sentence,sentences,sentence_to_id,vectors,output_file):
        """
        Process all entities.
        """
        results = {}
        for entity_id in entity_dic:
            result = self.get_target_kg_single(
                entity_dic,
                entity_id,
                id_to_sentence,
                sentences,
                sentence_to_id,
                vectors,
                output_file
            )
            results[entity_id] = result
        return results

    def convert_knowledge_graph(self, input_data):
        output = {
            "entities": [],
            "relations": []
        }

        entity_registry = {}

        # First pass: Process original entities
        for entity_key in input_data:
            node_data = input_data.get(entity_key, {})
            central_entity = node_data.get("central_entity")
            if not isinstance(central_entity, dict):
                logger.warning("Skip malformed kg node for %s: missing central_entity dict.", entity_key)
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
                    "description": central_entity.get("description", ""),  # Add description field
                    "attributes": {}
                }
                if "attributes" in central_entity:
                    for attr in central_entity["attributes"]:
                        entity["attributes"][attr["key"]] = attr["value"]
                entity_registry[entity_name] = entity

        # Second pass: Process relationships
        for entity_key in input_data:
            node_data = input_data.get(entity_key, {})
            central_entity = node_data.get("central_entity")
            if not isinstance(central_entity, dict):
                continue
            source_name = central_entity.get("name")
            if not source_name:
                continue
            
            if "relationships" in central_entity:
                for rel in central_entity["relationships"]:
                    if not isinstance(rel, dict):
                        logger.warning("Skip malformed relation for %s: relation is not a dict.", entity_key)
                        continue
                    if "target_name" not in rel or "target_type" not in rel or "relation" not in rel:
                        logger.warning("Skip malformed relation for %s: missing required fields.", entity_key)
                        continue
                    # Handle target entities that might be lists
                    target_names = rel["target_name"] if isinstance(rel["target_name"], list) else [rel["target_name"]]
                    target_type = rel["target_type"]
                    
                    for target_name in target_names:
                        # Register target entity
                        if target_name not in entity_registry:
                            entity_registry[target_name] = {
                                "name": target_name,
                                "type": target_type,
                                "description": rel.get("target_description", ""),  # Add description field
                                "attributes": {}
                            }
                        
                        # Add relationship quadruple (including relation_description)
                        relation_description = rel.get("relation_description", "")
                        output["relations"].append([
                            source_name,
                            rel["relation"],
                            target_name,
                            relation_description
                        ])

        output["entities"] = list(entity_registry.values())
        return output

    def _parse_json_like_response(self, response):
        """将 LLM 返回内容尽量解析为 JSON（支持代码块与弱格式 JSON）。"""
        raw = response.content if hasattr(response, 'content') else response
        if isinstance(raw, (dict, list)):
            return raw

        text = "" if raw is None else str(raw).strip()
        if not text:
            return None

        fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, flags=re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        candidates = [text]
        object_match = re.search(r"\{[\s\S]*\}", text)
        if object_match:
            candidates.append(object_match.group(0).strip())
        array_match = re.search(r"\[[\s\S]*\]", text)
        if array_match:
            candidates.append(array_match.group(0).strip())

        for candidate in candidates:
            for normalized in (candidate, candidate.replace("'", '"')):
                try:
                    return json.loads(normalized)
                except Exception:
                    continue
        return None

    def _normalize_knowledge_graph(self, knowledge_graph):
        """统一图谱输入格式，支持 dict/JSON 字符串/JSON 文件路径。"""
        data = knowledge_graph
        if isinstance(data, str):
            if os.path.exists(data):
                with open(data, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(data)

        # 处理嵌套的 JSON 字符串
        # if isinstance(data, str):
        #     data = json.loads(data)

        if not isinstance(data, dict):
            raise ValueError("knowledge_graph must be a dict, JSON string, or a valid JSON file path.")

        entities = data.get("entities", [])
        relations = data.get("relations", [])
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ValueError("knowledge_graph must contain list fields: entities and relations.")
        return {"entities": entities, "relations": relations}

    def _build_graph_indices(self, graph_data):
        """构建实体索引与关系邻接表，便于后续实体匹配和多跳扩展。"""
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
                "description": rel_description
            }
            normalized_relations.append(rel_item)

            adjacency_out.setdefault(source, []).append(rel_item)
            adjacency_in.setdefault(target, []).append(rel_item)

            if source not in entity_lookup:
                entity_lookup[source] = {"name": source, "type": "Unknown", "description": "", "attributes": {}}
            if target not in entity_lookup:
                entity_lookup[target] = {"name": target, "type": "Unknown", "description": "", "attributes": {}}

        return entity_lookup, normalized_relations, adjacency_out, adjacency_in

    def _compute_graph_cache_key(self, knowledge_graph):
        """基于图谱内容/路径生成稳定缓存键。"""
        if isinstance(knowledge_graph, str) and os.path.exists(knowledge_graph):
            abs_path = os.path.abspath(knowledge_graph)
            return f"path:{abs_path}"

        normalized_graph = self._normalize_knowledge_graph(knowledge_graph)
        graph_str = json.dumps(normalized_graph, ensure_ascii=False, sort_keys=True)
        digest = hashlib.md5(graph_str.encode("utf-8")).hexdigest()
        return f"graph:{digest}"

    def initialize_qa_graph_index(self, knowledge_graph, cache_key=None, force_rebuild=False):
        """
        初始化并缓存 QA 检索所需索引。
        可在多次提问前先调用一次，后续复用同一 cache_key。
        """
        normalized_graph = self._normalize_knowledge_graph(knowledge_graph)
        cache_key = cache_key or self._compute_graph_cache_key(normalized_graph)

        if (not force_rebuild) and cache_key in self._qa_graph_index_cache:
            self._qa_default_graph_cache_key = cache_key
            return cache_key

        entity_lookup, normalized_relations, adjacency_out, adjacency_in = self._build_graph_indices(normalized_graph)
        node_names = list(entity_lookup.keys())
        node_texts = [self._entity_to_retrieval_text(entity_lookup[name]) for name in node_names]

        node_vectors = None
        if node_texts:
            try:
                node_vectors = np.array(self.embeddings.embed_documents(node_texts))
            except Exception:
                logger.warning("Failed to precompute node embeddings for graph index: %s", traceback.format_exc())

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
        """获取缓存索引；缺失时可按需自动初始化。"""
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

        if self._qa_default_graph_cache_key and self._qa_default_graph_cache_key in self._qa_graph_index_cache:
            return self._qa_graph_index_cache[self._qa_default_graph_cache_key]

        raise ValueError(
            "No QA graph index available. Please call initialize_qa_graph_index(...) first, "
            "or pass knowledge_graph to answer_question_with_kg/build_qa_retrieval_context."
        )

    def clear_qa_graph_index(self, cache_key=None):
        """清理指定/全部 QA 图谱索引缓存。"""
        if cache_key is None:
            self._qa_graph_index_cache.clear()
            self._qa_default_graph_cache_key = None
            return

        self._qa_graph_index_cache.pop(cache_key, None)
        if self._qa_default_graph_cache_key == cache_key:
            self._qa_default_graph_cache_key = None

    def extract_question_entities(self, question, max_entities=8):
        """从用户问题抽取检索实体，失败时回退到正则分词。"""
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
            # LLM 提取失败时，使用中英文混合分词兜底。
            merged = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_\-]{1,}", question)

        return self._dedupe_preserve_order(merged)[:max_entities]

    def _entity_to_retrieval_text(self, entity):
        """将实体关键信息拼成检索文本，用于向量相似度计算。"""
        attrs = entity.get("attributes", {})
        attr_text = ""
        if isinstance(attrs, dict):
            attr_text = "; ".join([f"{k}:{v}" for k, v in attrs.items()])
        elif isinstance(attrs, list):
            attr_text = "; ".join([str(item) for item in attrs])
        text_parts = [
            entity.get("name", ""),
            entity.get("type", ""),
            entity.get("description", ""),
            attr_text
        ]
        return " ".join([str(part) for part in text_parts if part]).strip()

    def match_question_entities_to_graph(
        self,
        question,
        question_entities,
        graph_data=None,
        graph_index=None,
        top_k=5,
        similarity_threshold=0.35
    ):
        """
        将问题实体映射到图谱节点。
        策略：字符串匹配（高精度） + 向量相似度（召回补充）。
        """
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

            # 字符串匹配：优先命中同名或子串实体。
            for name in node_names:
                name_lower = name.lower()
                string_score = 0.0
                if q_lower == name_lower:
                    string_score = 1.0
                elif q_lower in name_lower or name_lower in q_lower:
                    string_score = 0.90
                if string_score > score_map.get(name, -1):
                    score_map[name] = string_score
                    reason_map[name] = f"string_match:{q}"

            # 向量相似度：补充语义相近但字符串不完全一致的实体。
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
                    logger.warning("Failed to compute query similarity for %s: %s", q, traceback.format_exc())

        # 当字符串和阈值检索都未命中时，退化为“问题全文 TopK 语义检索”。
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
            matched.append({
                "name": name,
                "score": float(score),
                "reason": reason_map.get(name, "unknown"),
                "type": entity.get("type", ""),
                "description": entity.get("description", "")
            })
        return matched

    def expand_graph_neighbors(
        self,
        graph_data=None,
        seed_nodes=None,
        graph_index=None,
        max_hop=1,
        max_neighbors_per_node=8,
        max_paths=40
    ):
        """
        基于种子节点进行 1/2 跳扩展，收集：
        1) 实体证据（描述/属性）
        2) 关系证据（关系描述）
        3) 图谱路径
        """
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
            """将实体描述与属性转为可回溯证据条目。"""
            entity = entity_lookup.get(node_name, {"name": node_name})
            description = entity.get("description", "")
            if isinstance(description, str) and description.strip():
                source = f"entity:{node_name}.description"
                if source not in visited_entity_source:
                    entity_evidence.append({
                        "source": source,
                        "text": description.strip(),
                        "entity": node_name
                    })
                    visited_entity_source.add(source)

            attrs = entity.get("attributes", {})
            if isinstance(attrs, dict):
                for key, value in list(attrs.items())[:5]:
                    source = f"entity:{node_name}.attribute:{key}"
                    if source in visited_entity_source:
                        continue
                    text = f"{node_name} 的属性 {key}: {value}"
                    entity_evidence.append({
                        "source": source,
                        "text": str(text),
                        "entity": node_name
                    })
                    visited_entity_source.add(source)

        for seed in seed_nodes:
            if seed not in entity_lookup:
                continue
            add_entity_evidence(seed)
            # BFS 扩展路径，记录最短深度，避免无意义回环膨胀。
            queue = deque([(seed, 0, seed)])
            visited_depth = {seed: 0}

            while queue and len(graph_paths) < max_paths:
                current_node, current_depth, current_path = queue.popleft()
                if current_depth >= hop_limit:
                    continue

                outgoing = adjacency_out.get(current_node, [])[:max_neighbors_per_node]
                incoming = adjacency_in.get(current_node, [])[:max_neighbors_per_node]

                # 出边扩展：current --[relation]-> target
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
                        relation_evidence.append({
                            "source": rel_source,
                            "text": rel_desc.strip() if isinstance(rel_desc, str) and rel_desc.strip() else f"{source} --[{relation}]-> {target}",
                            "path": new_path,
                            "hop": current_depth + 1
                        })
                        visited_relation_source.add(rel_source)

                    add_entity_evidence(next_node)
                    next_depth = current_depth + 1
                    if next_depth <= hop_limit and next_depth < visited_depth.get(next_node, 999):
                        visited_depth[next_node] = next_depth
                        queue.append((next_node, next_depth, new_path))

                # 入边扩展：source --[relation]-> current
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
                        relation_evidence.append({
                            "source": rel_source,
                            "text": rel_desc.strip() if isinstance(rel_desc, str) and rel_desc.strip() else f"{source} --[{relation}]-> {target}",
                            "path": new_path,
                            "hop": current_depth + 1
                        })
                        visited_relation_source.add(rel_source)

                    add_entity_evidence(next_node)
                    next_depth = current_depth + 1
                    if next_depth <= hop_limit and next_depth < visited_depth.get(next_node, 999):
                        visited_depth[next_node] = next_depth
                        queue.append((next_node, next_depth, new_path))

        return {
            "entity_evidence": entity_evidence,
            "relation_evidence": relation_evidence,
            "graph_paths": graph_paths[:max_paths]
        }

    def build_qa_retrieval_context(
        self,
        question,
        knowledge_graph=None,
        max_hop=2,
        seed_top_k=5,
        max_context_items=30,
        cache_key=None,
        graph_index=None
    ):
        """执行完整检索流程并输出给 LLM 的上下文文本与结构化证据。"""
        if graph_index is None:
            graph_index = self._get_qa_graph_index(
                knowledge_graph=knowledge_graph,
                cache_key=cache_key,
                auto_initialize=True
            )

        question_entities = self.extract_question_entities(question)
        matched_nodes = self.match_question_entities_to_graph(
            question=question,
            question_entities=question_entities,
            graph_index=graph_index,
            top_k=seed_top_k
        )
        seed_nodes = [item["name"] for item in matched_nodes]

        # 一跳/两跳邻居扩展，收集证据与路径。
        expanded = self.expand_graph_neighbors(
            graph_index=graph_index,
            seed_nodes=seed_nodes,
            max_hop=max_hop
        )

        evidence_items = []
        for item in expanded["entity_evidence"]:
            evidence_items.append({
                "source": item["source"],
                "text": item["text"],
                "path": ""
            })
        for item in expanded["relation_evidence"]:
            evidence_items.append({
                "source": item["source"],
                "text": item["text"],
                "path": item.get("path", "")
            })

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
            "context_text": context_text
        }

    def _normalize_evidence_output(self, evidence_sources, fallback_evidence):
        """规范化 LLM 的证据输出；若为空则回退到检索证据。"""
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
            fallback.append({
                "source": item.get("source", ""),
                "quote": item.get("text", "")
            })
        return fallback

    def _format_qa_output(self, answer, evidence_sources, graph_paths):
        """将结构化结果格式化为“答案+证据来源+图谱路径”的可读文本。"""
        answer_text = answer if isinstance(answer, str) and answer.strip() else "根据现有图谱证据不足以得出确定结论"
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
        graph_index=None
    ):
        """
        对外统一问答入口：
        问题实体抽取 -> 图谱匹配 -> 多跳扩展 -> LLM 作答 -> 结果规范化。
        """
        retrieval = self.build_qa_retrieval_context(
            question=question,
            knowledge_graph=knowledge_graph,
            max_hop=max_hop,
            seed_top_k=seed_top_k,
            max_context_items=max_context_items,
            cache_key=cache_key,
            graph_index=graph_index
        )

        graph_paths_text = "\n".join([f"- {path}" for path in retrieval["graph_paths"][:10]])
        prompt = ChatPromptTemplate.from_template(kg_qa_answer_prompt_cn)
        chain = prompt | self.model
        # 将检索证据与候选路径喂给 LLM 生成最终答案。
        llm_result = chain.invoke({
            "question": question,
            "context": retrieval["context_text"],
            "graph_paths": graph_paths_text
        })

        parsed = self._parse_json_like_response(llm_result)
        if isinstance(parsed, dict):
            answer = parsed.get("answer", "")
            evidence_sources = self._normalize_evidence_output(
                parsed.get("evidence_sources", []),
                retrieval["evidence_items"]
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
            "formatted_answer": formatted_answer
        }

    def process(self):
        pass
