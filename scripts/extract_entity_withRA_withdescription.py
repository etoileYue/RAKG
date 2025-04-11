from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from scripts.prompt import text2entity_en
from scripts.prompt import extract_entiry_centric_kg_en_v2
from scripts.prompt import judge_sim_entity_en
from langchain_ollama import OllamaEmbeddings  
from itertools import combinations
import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config import OLLAMA_BASE_URL, DEFAULT_MODEL, EMBEDDING_MODEL,SIMILARITY_MODEL

class NER_Agent():
    def __init__(self, model=DEFAULT_MODEL):
        self.model = model # Model for NER recognition
        self.similarity_model = SIMILARITY_MODEL
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)  

    
    ## Add chunkid attribute
    def add_chunkid(self, ner_result, chunkid):
        new_ner_result = {}
        for entity_key, entity_value in ner_result.items():
            entity_value["chunkid"] = chunkid
            new_ner_result[entity_key] = entity_value
        return new_ner_result
    
    
    def extract_from_text_single(self, text_single, output_file ):
        model = OllamaLLM(model=self.model, base_url=OLLAMA_BASE_URL, format='json', temperature=0)
        prompt = ChatPromptTemplate.from_template(text2entity_en)
        chain = prompt | model
        result = chain.invoke({"text": text_single})
        # print("result")
        # print(result)
        result_json = json.loads(result)
        
        # Store text_single and result_json in a jsonl file
        with open(output_file, 'a') as f:
            combined_data = {
                "text": text_single,
                "entities": result_json
            }
            f.write(json.dumps(combined_data) + '\n')
        
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
            # print("ner_result_ori")
            # print(ner_result)
            # Get the number of entities in ner_result
            ner_result_num = len(ner_result)
            # Rewrite ner_result, entity numbering starts from entity_num, first entity is entity{entity_num}, subsequent entities increment
            ner_result = self.rewrite(ner_result, entity_num)

            entity_num += ner_result_num
            ner_result_with_chunkid = self.add_chunkid(ner_result,sent_to_id[text])
            ner_result_for_all.update(ner_result_with_chunkid)
        return ner_result_for_all
    

    def similarity_candidates(self,entities, threshold=0.60):
        entity_texts = {
            k: f"{v['name']} {v['type']}"
            for k, v in entities.items()
        }
        vectors = {k: self.embeddings.embed_documents(text) for k, text in entity_texts.items()}

        keys = list(vectors.keys())
        sim_matrix = np.zeros((len(keys), len(keys)))

        for i, j in combinations(range(len(keys)), 2):
            sim = cosine_similarity(vectors[keys[i]], vectors[keys[j]])
            sim_matrix[i][j] = sim
        # print(sim_matrix)
        # Step 3: Identify candidate pairs
        candidates = [(keys[i], keys[j]) 
                    for i, j in zip(*np.where(sim_matrix > threshold))]
        # print("Initial similarity groups")
        # print(candidates)
        return candidates

    ## Use LLM for entity disambiguation
    def similarity_llm_single(self, entity1, entity2):
        model = OllamaLLM(model=self.similarity_model, base_url=OLLAMA_BASE_URL, format='json', temperature=0)
        prompt = ChatPromptTemplate.from_template(judge_sim_entity_en)
        chain = prompt | model
        result = chain.invoke({"entity1": str(entity1),"entity2": str(entity2)})
        result_json = json.loads(result)
        # Return {'result': True} for same entities
        # Return {'result': False} for different entities
        return result_json
    
    ## First use similarity_candidates for initial filtering of all entities, similarity_candidates returns: candidates = [('entity1', 'entity48'), ('entity3', 'entity68'), ('entity4', 'entity39')]
    ## Then for each candidate in candidates, use similarity_llm_single to judge, if they are the same entity, keep the candidate, otherwise delete
    ## Finally return new candidates_result
    def similartiy_result(self, entities):
        # Step 1: Use similarity_candidates for initial filtering
        candidates = self.similarity_candidates(entities)
        
        # Step 2: Fine-grained LLM judgment for each candidate pair
        candidates_result = []
        for ent_pair in candidates:
            # Extract entity objects from entities dictionary
            entity1 = entities.get(ent_pair[0])
            entity2 = entities.get(ent_pair[1])
            
            # Call LLM for judgment
            try:
                result = self.similarity_llm_single(entity1, entity2)
                # Keep if LLM judges as same entity
                if result.get('result', False):
                    candidates_result.append(ent_pair)
            except Exception as e:
                print(f"Error processing entity pair {ent_pair}: {str(e)}")
                continue  # Can log or raise exception as needed
        
        # Step 3: Return final filtered candidate pairs
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
            descriptions = set()
            chunkids = set()

            for e in group:
                descriptions.add(entity_dic[e]['description'])
                chunkids.add(entity_dic[e]['chunkid'])
                if e != main_entity:
                    del entity_dic[e]  # Remove merged entities

            # Merge fields
            entity_dic[main_entity]['description'] = ';;;'.join(descriptions)
            entity_dic[main_entity]['chunkid'] = ';;;'.join(chunkids)

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
                print(f"Warning: Chunk ID '{chunkid}' not found in id_to_sentence.")

        return sentences

    def get_retriever_context(self, query, sentences, sentence_to_id,vectors,top_k=5):
        """
        Get the top_k most similar sentences as retriever context for a query.

        :param query: str, user's query text
        :param top_k: int, number of most similar sentences to return, default is 5
        :return: list of tuples, each tuple contains (sentence, similarity, sentence_id)
        """

        # Step 1: Convert query to vector
        query_vector = self.embeddings.embed_query(query)

        # Step 2: Calculate cosine similarity between query vector and sentence vectors
        sentence_vectors = np.array(vectors)
        similarities = cosine_similarity([query_vector], sentence_vectors)[0]

        # Step 3: Select top_k most similar sentences
        top_indices = np.argsort(similarities)[::-1][:top_k]  # Sort by similarity in descending order and take top_k
        retriever_context = []
        for idx in top_indices:
            sentence = sentences[idx]
            similarity = similarities[idx]
            sentence_id = sentence_to_id[sentence]
            retriever_context.append((sentence, similarity, sentence_id))

        return retriever_context

    def get_target_kg_sigle(self, entity_dic, entity_id, id_to_sentence, sentences, sentence_to_id, vectors, output_file):
        model = OllamaLLM(model=self.model, base_url=OLLAMA_BASE_URL, format='json', temperature=0)
        chunk_text_list = self.get_sentences_for_entity(entity_dic, entity_id, id_to_sentence)
        ## Add retriever context
        query = entity_dic[entity_id].get('name', '')
        context = self.get_retriever_context(query, sentences, sentence_to_id, vectors, top_k=5)
        # print("context")
        # print(context)
        sentences = [item[0] for item in context]
        unique_sentences = list(set(chunk_text_list + sentences))
        chunk_text = ", ".join(unique_sentences)
        prompt = ChatPromptTemplate.from_template(extract_entiry_centric_kg_en_v2)
        chain = prompt | model
        result = chain.invoke({"text": chunk_text, "target_entity": entity_dic[entity_id].get('name'), "related_kg": 'none'})
        # print("context")
        # print(chunk_text)
        # print("entity")
        # print(entity_dic[entity_id].get('name'))
        print("kg")
        result_json = json.loads(result)
        print(result_json)

        # Store chunk_text, entity, and kg in a jsonl file
        with open(output_file, 'a') as f:
            combined_data = {
                "chunk_text": chunk_text,
                "entity": entity_dic[entity_id],
                "kg": result_json
            }
            f.write(json.dumps(combined_data) + '\n')

        return result_json
    
    
    def get_target_kg_all(self, entity_dic, id_to_sentence,sentences,sentence_to_id,vectors,output_file):
        """
        Process all entities.
        """
        results = {}
        for entity_id in entity_dic:
            if entity_id in entity_dic:
                result = self.get_target_kg_sigle(entity_dic, entity_id, id_to_sentence,sentences,sentence_to_id,vectors,output_file)
                results[entity_id] = result
            else:
                print(f"Entity {entity_id} not found in entity_dic.")
        return results

    def convert_knowledge_graph(self, input_data):
        output = {
            "entities": [],
            "relations": []
        }

        entity_registry = {}

        # First pass: Process original entities
        for entity_key in input_data:
            central_entity = input_data[entity_key]["central_entity"]
            entity_name = central_entity["name"]
            
            if entity_name not in entity_registry:
                entity = {
                    "name": entity_name,
                    "type": central_entity["type"],
                    "description": central_entity.get("description", ""),  # Add description field
                    "attributes": {}
                }
                if "attributes" in central_entity:
                    for attr in central_entity["attributes"]:
                        entity["attributes"][attr["key"]] = attr["value"]
                entity_registry[entity_name] = entity

        # Second pass: Process relationships
        for entity_key in input_data:
            central_entity = input_data[entity_key]["central_entity"]
            
            if "relationships" in central_entity:
                for rel in central_entity["relationships"]:
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
                            central_entity["name"],
                            rel["relation"],
                            target_name,
                            relation_description
                        ])

        output["entities"] = list(entity_registry.values())
        return output


    def process(self):
        pass



