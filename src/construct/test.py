# from langchain_core.prompts import ChatPromptTemplate
# from src.prompt import judge_sim_entity_en
# from src.llm_provider import LLMProvider
# import json

# llm_provider = LLMProvider()
# similarity_model = llm_provider.get_similarity_model()

# entity1 = {'name': 'nails', 'type': 'Body Part', 'description': 'The hard, protective plates at the tips of fingers and toes in humans, also primarily composed of keratin.', 'chunkid': 'Unusual Animal Adaptations15'}
# entity2 =  {'name': 'aye-aye', 'type': 'Animal', 'description': 'A type of lemur known for its unique elongated middle finger, native to Madagascar.', 'chunkid': 'Unusual Animal Adaptations28'}
# def similarity_llm_single(entity1, entity2):
#     prompt = ChatPromptTemplate.from_template(judge_sim_entity_en)
#     chain = prompt | similarity_model
#     result = chain.invoke({"entity1": str(entity1), "entity2": str(entity2)})
#     if hasattr(result, 'content'):
#         result_json = json.loads(result.content)
#     else:
#         result_json = json.loads(result)
#     return result_json

# res = similarity_llm_single(entity1, entity2)

# print(res)

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(sys.path)