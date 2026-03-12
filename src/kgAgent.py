from langchain_core.prompts import ChatPromptTemplate
from src.prompt import text2entity_en
from src.prompt import extract_entiry_centric_kg_en_v2
from src.prompt import judge_sim_entity_en
from itertools import combinations
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.llm_provider import LLMProvider
import os

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

    def process(self):
        pass



