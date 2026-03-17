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

    def similarity_result(self, entities, threshold=0.60, gray_margin=0.05):
        candidates = self.similarity_candidates(entities, threshold=threshold)

        candidates_result = []
        gray_queue = []
        for ent_pair in candidates:
            left_id, right_id, score = ent_pair
            entity1 = entities.get(left_id)
            entity2 = entities.get(right_id)

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
                    candidates_result.append((left_id, right_id))
            except Exception:
                logger.error(
                    "Error processing entity pair %s: %s",
                    ent_pair,
                    traceback.format_exc(),
                )
                continue

        resolved_by_second_pass = 0
        for item in gray_queue:
            left_id, right_id = item["pair"]
            entity1 = entities.get(left_id)
            entity2 = entities.get(right_id)
            try:
                second_pass_result = self.similarity_llm_single(entity1, entity2)
                if second_pass_result.get("result", False):
                    candidates_result.append((left_id, right_id))
                    resolved_by_second_pass += 1
                item["second_pass_result"] = second_pass_result
            except Exception:
                logger.error(
                    "Error in second-pass disambiguation for pair %s: %s",
                    (left_id, right_id),
                    traceback.format_exc(),
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
        return candidates_result

    def entity_Disambiguation(self, entity_dic, sim_entity_list):
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

            for entity in group:
                descriptions.append(entity_dic[entity]["description"])
                chunkids.append(entity_dic[entity]["chunkid"])
                if entity != main_entity:
                    del entity_dic[entity]

            dedup_descriptions = self._dedupe_preserve_order(descriptions)
            dedup_chunkids = self._dedupe_preserve_order(chunkids)
            entity_dic[main_entity]["description"] = ";;;".join(dedup_descriptions)
            entity_dic[main_entity]["chunkid"] = ";;;".join(dedup_chunkids)

        return entity_dic

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
    ):
        chunk_text_list = self.get_sentences_for_entity(entity_dic, entity_id, id_to_sentence)
        query = entity_dic[entity_id].get("name", "")
        context = self.get_retriever_context(query, sentences, sentence_to_id, vectors, top_k=5)
        retrieved_sentences = [item[0] for item in context]
        unique_sentences = self._dedupe_preserve_order(chunk_text_list + retrieved_sentences)
        chunk_text = ", ".join(unique_sentences)

        prompt = ChatPromptTemplate.from_template(extract_entiry_centric_kg_en_v2)
        chain = prompt | self.model
        result = chain.invoke(
            {
                "text": chunk_text,
                "target_entity": entity_dic[entity_id].get("name"),
                "related_kg": "none",
            }
        )

        debug_logger.debug("-get_target_kg_single-")
        debug_logger.debug(
            "text=%s, target_entity=%s, related_kg=none, result=%s",
            chunk_text,
            entity_dic[entity_id].get("name"),
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
    ):
        results = {}
        for entity_id in entity_dic:
            result = self.get_target_kg_single(
                entity_dic,
                entity_id,
                id_to_sentence,
                sentences,
                sentence_to_id,
                vectors,
                output_file,
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
                }
                if "attributes" in central_entity:
                    for attr in central_entity["attributes"]:
                        entity["attributes"][attr["key"]] = attr["value"]
                entity_registry[entity_name] = entity

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