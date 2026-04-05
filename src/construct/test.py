import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm_provider import LLMProvider
from sklearn.metrics.pairwise import cosine_similarity

llm_provider = LLMProvider()
embedding = llm_provider.get_embedding_model()

entity1 = "Butterfly insect"
entity2 =  "butterfly Animal"

left = embedding.embed_query(entity1)
right = embedding.embed_query(entity2)

sim = cosine_similarity([left], [right])[0][0]

print(f"Entity 1: {entity1}")
print(f"Entity 2: {entity2}")
print(f"Cosine similarity: {sim:.6f}")
