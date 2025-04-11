import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings 
import json
from collections import defaultdict
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL, DEFAULT_MODEL
judge_sim_entity_en = """
You are an expert in entity disambiguation. Please determine whether the following two entities refer to the same real-world entity based on their names, types, and descriptions. Consider possible abbreviations, synonyms, and contextual clues.

Entity 1: {entity1}
Entity 2: {entity2}

Respond ONLY with a JSON object: {{"result": true}} if they are the same entity, or {{"result": false}} otherwise. Do not include any explanations.
"""

relation_similarity_prompt = """
Please evaluate the semantic similarity between two sets of relationships for the same entity from different knowledge graphs. Score between 0-1 where 1 means identical meaning and coverage.

Relationship Set A:
{relations1}

Relationship Set B:
{relations2}

Consider:
1. Completeness of information
2. Semantic equivalence of relationships
3. Consistency of connected entities
4. Overall knowledge structure similarity

Return a JSON object with one key "similarity" representing the similarity score. Example: {{"similarity": 0.85}}
"""

class KGEvaluator:
    def __init__(self, kg_standard, kg_eval, embedding_model, llm_model):
        self.kg_standard = kg_standard
        self.kg_eval = kg_eval
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Preprocess KG structures
        self.std_entities, self.std_relations = self._preprocess_kg(kg_standard)
        self.eval_entities, self.eval_relations = self._preprocess_kg(kg_eval)

    def _preprocess_kg(self, kg):
        """Build entity dictionary and relation index"""
        entities = {e['name']: e for e in kg['entities']}
        relations = defaultdict(list)
        
        for rel in kg['relations']:
            # Handle potentially incomplete relation items
            rel += [''] * (4 - len(rel))
            head, rel_name, tail, desc = rel[:4]
            relations[head].append(rel)
            relations[tail].append(rel)
            
        return entities, relations

    def _get_embedding(self, text):
        """Get text embedding vector"""
        return self.embedding_model.embed_documents(text)

    def _find_candidate_matches(self, threshold=0.0):
        """Find all candidate matching entity pairs"""
        # Generate standard entity text
        std_texts = {
            name: f"{e['name']} {e.get('type','')} {e.get('description','')}"
            for name, e in self.std_entities.items()
        }
        # Generate evaluation entity text
        eval_texts = {
            name: f"{e['name']} {e.get('type','')} {e.get('description','')}" 
            for name, e in self.eval_entities.items()
        }

        # Calculate embedding vectors
        std_embeddings = {k: self._get_embedding(v) for k, v in std_texts.items()}
        eval_embeddings = {k: self._get_embedding(v) for k, v in eval_texts.items()}

        # Calculate similarity matrix
        candidates = []
        for s_name, s_vec in std_embeddings.items():
            for e_name, e_vec in eval_embeddings.items():
                sim = cosine_similarity(s_vec, e_vec)[0][0]
                if sim > threshold:
                    candidates.append((s_name, e_name, sim))
        
        # Sort by similarity
        return sorted(candidates, key=lambda x: -x[2])

    def _llm_judge_entity(self, entity1, entity2):
        """Use LLM to judge if entities are the same"""
        prompt = ChatPromptTemplate.from_template(judge_sim_entity_en)
        chain = prompt | self.llm_model
        response = chain.invoke({
            "entity1": json.dumps(entity1, ensure_ascii=False),
            "entity2": json.dumps(entity2, ensure_ascii=False)
        })
        try:
            return json.loads(response)['result']
        except:
            return False

    def _llm_relation_similarity(self, rels1, rels2):
        """Use LLM to evaluate relation similarity"""
        def format_rels(rels):
            formatted = []
            for rel in rels:
                head, r, tail, desc = rel[:4]
                desc = desc if desc and desc != 'undefine' else ""
                formatted.append(f"{head} --{r}-> {tail} {desc}".strip())
            return "\n".join(formatted) or "No relationships"
        
        prompt = ChatPromptTemplate.from_template(relation_similarity_prompt)
        chain = prompt | self.llm_model
        response = chain.invoke({
            "relations1": format_rels(rels1),
            "relations2": format_rels(rels2)
        })
        
        try:
            return json.loads(response)['similarity']
        except:
            return 0.0

    def evaluate(self):
        """Execute evaluation process"""
        # Step 1: Find all candidate matches
        candidates = self._find_candidate_matches()
        
        # Step 2: LLM confirms matches
        matched_pairs = []
        matched_eval_entities = set()
        
        for s_name, e_name, _ in candidates:
            if s_name in matched_eval_entities:
                continue  # Each standard entity is matched only once
                
            std_entity = self.std_entities[s_name]
            eval_entity = self.eval_entities[e_name]
            
            if self._llm_judge_entity(std_entity, eval_entity):
                matched_pairs.append((std_entity, eval_entity))
                matched_eval_entities.add(s_name)
        
        # Calculate metrics
        entity_cover = len(matched_pairs)
        total_std_entities = len(self.std_entities)
        cover_rate = entity_cover / total_std_entities if total_std_entities else 0.0
        
        rel_similarities = []
        for std_ent, eval_ent in matched_pairs:
            std_rels = self.std_relations.get(std_ent['name'], [])
            eval_rels = self.eval_relations.get(eval_ent['name'], [])
            sim = self._llm_relation_similarity(std_rels, eval_rels)
            rel_similarities.append(sim)
        
        avg_rel_sim = sum(rel_similarities)/len(rel_similarities) if rel_similarities else 0.0
        
        return {
            "entity_coverage_rate": round(cover_rate, 4),
            "relation_similarity": round(avg_rel_sim, 4),
            "matched_entities_count": entity_cover,
            "total_std_entities": total_std_entities
        }

# Example usage
if __name__ == "__main__":
    # Initialize models (configure according to actual needs)
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL) 
    llm = OllamaLLM(model=DEFAULT_MODEL, base_url=OLLAMA_BASE_URL, format='json')
    
    # Specify output file path
    output_file = "kg_evaluation_results.jsonl"

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i in range(1, 106):
            # Read graphrag data
            graphrag_file = f"data/processed/graphrag_graph/{i}.json"
            with open(graphrag_file, "r") as f1:
                data_graphrag = json.load(f1)
            print("Successfully imported graphrag data")

            # Read clean data
            clean_file = f"data/processed/idealKG/kg_{i}.json"
            with open(clean_file, "r") as f2:
                data_clean = json.load(f2)
            print("Successfully imported ideal data")

            # Read rakg data (note: json.loads is called twice due to data format)
            rakg_file = f"data/processed/RAKG_graph_v2/{i}.json"
            with open(rakg_file, "r", encoding="utf-8") as f3:
                data_rakg = f3.read()
            data_rakg = json.loads(data_rakg)
            data_rakg = json.loads(data_rakg)
            print("Successfully imported RAKG data")

            # Read kggen data
            kggen_file = f"data/processed/kggen_graph/{i}_trans.json"
            with open(kggen_file, "r") as f4:
                data_kggen = json.load(f4)
            print("Successfully imported kggen data")

            print("Using standard kg as reference")

            # Evaluate kggen
            evaluator = KGEvaluator(data_clean, data_kggen, embedding_model, llm)
            results_kggen = evaluator.evaluate()
            print("Evaluating kggen")
            print(f"Entity Coverage Rate: {results_kggen['entity_coverage_rate']}")
            print(f"Relation Similarity: {results_kggen['relation_similarity']}")

            # Evaluate graphrag
            evaluator = KGEvaluator(data_clean, data_graphrag, embedding_model, llm)
            results_graphrag = evaluator.evaluate()
            print("Evaluating graphrag")
            print(f"Entity Coverage Rate: {results_graphrag['entity_coverage_rate']}")
            print(f"Relation Similarity: {results_graphrag['relation_similarity']}")

            # Evaluate rakg
            evaluator = KGEvaluator(data_clean, data_rakg, embedding_model, llm)
            results_rakg = evaluator.evaluate()
            print("Evaluating rakg")
            print(f"Entity Coverage Rate: {results_rakg['entity_coverage_rate']}")
            print(f"Relation Similarity: {results_rakg['relation_similarity']}")

            # Construct result dictionary according to example requirements (adjust key names as needed)
            out_data = {
                "dataid": i,
                "kggen": {
                    "Entity Coverage Rate": results_kggen["entity_coverage_rate"],
                    "Relation Similarity": results_kggen["relation_similarity"]
                },
                "graphrag": {
                    "Entity Coverage Rate": results_graphrag["entity_coverage_rate"],
                    "Relation Similarity": results_graphrag["relation_similarity"]
                },
                "rakg": {
                    "Entity Coverage Rate": results_rakg["entity_coverage_rate"],
                    "Relation Similarity": results_rakg["relation_similarity"]
                }
            }

            # Write results to JSONL file, one JSON object per line
            out_f.write(json.dumps(out_data, ensure_ascii=False) + "\n")


