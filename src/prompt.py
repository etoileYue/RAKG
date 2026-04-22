"""Prompt templates and runtime prompt selection."""

from src import config as prompt_config


extract_entiry_centric_kg_en = """
    You are a knowledge graph extraction assistant. Combining knowledge from other relevant knowledge graphs, you are responsible for extracting attributes and relationships related to the specified entity from the text.
    Text: {text}
    Specified entity: {target_entity}
    Relevant knowledge graphs: {related_kg}
    Requirements for you:
    1. You should comprehensively analyze the entire text and extract relationships related to the specified entity. A subgraph should be established for the specified entity.
    2. You should extract both the attributes of the specified entity and the relationships between the specified entity and other entities.
        For attribute extraction: Attributes describe the characteristics of the specified entity. For example, in "Jordan - Gender: Male," gender is an attribute.
        For relationship extraction, the head entity of the relationship must be the specified entity. For example, "Specified entity - Has - Other entity" is valid, while "Other entity - Is owned by - Specified entity" is invalid.
    3. You should determine when to classify information as a relationship and when to classify it as an attribute.
    4. Knowledge from other relevant knowledge graphs can help you more comprehensively understand the characteristics of the specified entity. Moreover, you should use this knowledge to establish reverse relationships for the specified entity, forming bidirectional relationships. For example, if "Other entity - Wife - Specified entity," you should establish "Specified entity - Husband - Other entity" to make the knowledge graph more comprehensive.
    5. In the final output, duplicate attributes should be retained only once, and duplicate relationships should be retained only once.
    6. The final output format should be:
        {{
    "central_entity": {{
        "name": "{{}}",
        "type": "{{}}",
        "attributes": [
        {{
            "key": "{{}}",
            "value": "{{}}"
        }},
            ...
        {{
            "key": "{{}}",
            "value": "{{}}"
        }}
        ],
        "relationships": [
        {{
            "relation": "{{}}",
            "target_name": "{{}}",
            "target_type": "{{}}"
        }},
        ...
        {{
            "relation": "{{}}",
            "target_name": "{{}}",
            "target_type": "{{}}"
        }}
        ]
      }}
    }}
    For example:
    {{
  "central_entity": {{
    "name": "Albert Einstein",
    "type": "Person",
    "attributes": [
      {{
        "key": "Date of Birth",
        "value": "1879-03-14"
      }},
      {{
        "key": "Occupation",
        "value": "Theoretical Physicist"
      }}
    ],
    "relationships": [
      {{
        "relation": "Proposed Theory",
        "target_name": "Theory of Relativity",
        "target_type": "Scientific Theory"
      }},
      {{
        "relation": "Graduated from",
        "target_name": "Swiss Federal Polytechnic",
        "target_type": "Educational Institution"
      }}
    ]
  }}
}}
"""

extract_entiry_centric_kg_en_v2 = """
You are a knowledge graph extraction assistant, responsible for extracting attributes and relationships related to a specified entity from the text, in combination with other relevant knowledge graphs.
Text: {text}
Target Entity: {target_entity}
Related Knowledge Graphs: {related_kg}
Requirements for you:
1. You should integrate the entire text to comprehensively extract relationships related to the specified entity and build a sub-graph for the specified entity.
2. You should extract attributes of the specified entity and relationships between the specified entity and other entities.
   - For attribute extraction: Attributes are descriptions of the characteristics of the specified entity. For example, in "Michael Jordan - Gender: Male," gender is an attribute.
   - For relationship extraction, the head entity of the relationship must be the specified entity. For example, "Specified Entity - Owns - Other Entity" is valid, while "Other Entity - Is Owned By - Specified Entity" is invalid.
3. You should determine when to classify information as a relationship and when to classify it as an attribute.
4. Utilize knowledge from other relevant knowledge graphs to gain a more comprehensive understanding of the specified entity's characteristics. You should also establish reverse relationships based on other knowledge to form bidirectional relationships. For example, if there is a relationship like "Other Entity - Wife - Specified Entity," you should establish the reverse relationship: "Specified Entity - Husband - Other Entity" to make the knowledge graph more comprehensive.
5. In the final output, duplicate attributes should be removed, and only one instance of each attribute should be retained. Similarly, duplicate relationships should also be removed, and only one instance of each relationship should be retained.
6. Evidence constraints:
   - The text is formatted as evidence blocks with `chunk_id`. Each extracted relationship must be supported by evidence.
   - For each relationship, you must provide `provenance.chunk_ids`, and every chunk id must exist in the provided evidence blocks.
   - Optional: provide `provenance.confidence` as a float between 0 and 1.
   - Pronouns must be resolved to explicit antecedents from evidence blocks before extracting relations.
   - If a relation has no supporting evidence, do not output it.
7. Output only a JSON object. Do not output markdown, explanations, or any extra text.
8. The final output format should be:
    {{
    "central_entity": {{
        "name": "{{}}",
        "type": "{{}}",
        "description": "{{}}",
        "attributes": [
        {{
            "key": "{{}}",
            "value": "{{}}"
        }},
            ...
        {{
            "key": "{{}}",
            "value": "{{}}"
        }}
        ],
        "relationships": [
        {{
            "relation": "{{}}",
            "target_name": "{{}}",
            "target_type": "{{}}",
            "target_description": "{{}}",
            "relation_description": "{{}}",
            "provenance": {{
                "chunk_ids": ["{{}}"],
                "confidence": 0.0
            }}
        }},
        ...
        {{
            "relation": "{{}}",
            "target_name": "{{}}",
            "target_type": "{{}}",
            "target_description": "{{}}",
            "relation_description": "{{}}",
            "provenance": {{
                "chunk_ids": ["{{}}"]
            }}
        }}
        ]
      }}
    }}
For example:
{{
  "central_entity": {{
    "name": "Albert Einstein",
    "type": "Person",
    "description": "Albert Einstein is widely recognized as one of the greatest physicists since Newton.",
    "attributes": [
      {{
        "key": "Date of Birth",
        "value": "1879-03-14"
      }},
      {{
        "key": "Occupation",
        "value": "Theoretical Physicist"
      }}
    ],
    "relationships": [
      {{
        "relation": "Proposed Theory",
        "target_name": "Theory of Relativity",
        "target_type": "Scientific Theory",
        "target_description": "The Theory of Relativity was proposed by Einstein in 1905. It suggests that space and time transformations are interrelated during the motion of objects, rather than being independent.",
        "relation_description": "Einstein proposed the Theory of Relativity, which is an important theory in modern physics.",
        "provenance": {{
          "chunk_ids": ["Einstein12", "Einstein13"],
          "confidence": 0.94
        }}
      }},
      {{
        "relation": "Graduated From",
        "target_name": "ETH Zurich",
        "target_type": "Educational Institution",
        "target_description": "ETH Zurich is a university located near Zurich.",
        "relation_description": "Einstein studied at ETH Zurich.",
        "provenance": {{
          "chunk_ids": ["Einstein08"]
        }}
      }}
    ]
  }}
}}
"""

extract_entiry_centric_kg_zh_v2 = """
你是一个知识图谱抽取助手，需要结合其他相关知识图谱，从文本中抽取与指定实体有关的属性和关系。
文本：{text}
指定实体：{target_entity}
相关知识图谱：{related_kg}

要求：
1. 你必须综合整个文本，围绕指定实体构建完整的实体中心子图，不能只看局部句子。
2. 你需要同时抽取指定实体的属性，以及指定实体与其他实体之间的关系。
   - 属性用于描述指定实体自身特征，例如“乔丹 - 性别: 男”中的“性别”属于属性。
   - 关系用于描述指定实体与其他实体之间的连接，且关系的头实体必须是指定实体。例如“指定实体 - 拥有 - 其他实体”是合法的，“其他实体 - 被拥有 - 指定实体”是不合法的。
3. 你必须准确区分哪些信息应归为属性，哪些信息应归为关系。
4. 你应结合其他相关知识图谱，补全指定实体的反向关系，形成更完整的双向知识。例如如果相关知识中存在“其他实体 - 妻子 - 指定实体”，你应补全“指定实体 - 丈夫 - 其他实体”。
5. 最终输出中，重复属性只保留一个，重复关系也只保留一个。
6. 证据约束：
   - 输入文本由带有 `chunk_id` 的证据块组成，每一条关系都必须有证据支撑。
   - 每条关系都必须输出 `provenance.chunk_ids`，且其中每个 chunk_id 都必须来自给定证据块。
   - 可以选择性输出 `provenance.confidence`，取值范围是 0 到 1 的浮点数。
   - 在抽取关系前，必须先完成代词、指代和共指消解，将“他/她/它/该机构/这家公司”等指代还原成明确实体。
   - 没有证据支撑的关系不要输出。
7. 只输出 JSON 对象，不要输出 markdown、解释或任何额外文本。
8. 输出格式必须为：
{{
  "central_entity": {{
    "name": "{{}}",
    "type": "{{}}",
    "description": "{{}}",
    "attributes": [
      {{
        "key": "{{}}",
        "value": "{{}}"
      }}
    ],
    "relationships": [
      {{
        "relation": "{{}}",
        "target_name": "{{}}",
        "target_type": "{{}}",
        "target_description": "{{}}",
        "relation_description": "{{}}",
        "provenance": {{
          "chunk_ids": ["{{}}"],
          "confidence": 0.0
        }}
      }}
    ]
  }}
}}
例如：
{{
  "central_entity": {{
    "name": "阿尔伯特·爱因斯坦",
    "type": "人物",
    "description": "阿尔伯特·爱因斯坦被公认为是继牛顿之后最伟大的物理学家之一。",
    "attributes": [
      {{
        "key": "出生日期",
        "value": "1879-03-14"
      }},
      {{
        "key": "职业",
        "value": "理论物理学家"
      }}
    ],
    "relationships": [
      {{
        "relation": "提出理论",
        "target_name": "相对论",
        "target_type": "科学理论",
        "target_description": "相对论是由爱因斯坦于 1905 年提出的重要物理理论。",
        "relation_description": "爱因斯坦提出了相对论，这是现代物理学的重要理论。",
        "provenance": {{
          "chunk_ids": ["Einstein12", "Einstein13"],
          "confidence": 0.94
        }}
      }},
      {{
        "relation": "毕业于",
        "target_name": "苏黎世联邦理工学院",
        "target_type": "教育机构",
        "target_description": "苏黎世联邦理工学院是一所位于苏黎世的大学。",
        "relation_description": "爱因斯坦曾就读于苏黎世联邦理工学院。",
        "provenance": {{
          "chunk_ids": ["Einstein08"]
        }}
      }}
    ]
  }}
}}
"""

fewshot_for_extract_entiry_centric_kg = """
assistant:
user:
"""


text2entity_en = """
You are a named entity recognition assistant responsible for identifying named entities from the given text.
Text: {text}
Notes:
1. First, determine whether the text contains meaningful information. If it is only meaningless symbols, directly output `{{"State": false}}`.
2. Perform named entity recognition based on the whole text.
3. Each entity must contain three fields: `name`, `type`, and `description`.
4. Output only a JSON object. Do not output markdown or explanations.
5. Output format:
{{
  "entity1": {{
    "name": "Entity Name 1",
    "type": "Entity Type 1",
    "description": "Entity Description 1"
  }},
  "entity2": {{
    "name": "Entity Name 2",
    "type": "Entity Type 2",
    "description": "Entity Description 2"
  }}
}}
"""

text2entity_zh = """
你是一个命名实体识别助手，负责从给定文本中识别命名实体。
文本：{text}
注意：
1. 先判断文本是否包含有效信息；如果只有无意义符号，直接输出 `{{"State": false}}`。
2. 必须基于整段文本进行命名实体识别。
3. 每个命名实体都需要包含 `name`、`type`、`description` 三个字段。
4. 只输出 JSON 对象，不要输出 markdown 或额外解释。
5. 输出格式：
{{
  "entity1": {{
    "name": "实体名称1",
    "type": "实体类型1",
    "description": "实体描述1"
  }},
  "entity2": {{
    "name": "实体名称2",
    "type": "实体类型2",
    "description": "实体描述2"
  }}
}}
"""

fewshot_for_ext2entity = """

"""


judge_sim_entity_en = """
You are a knowledge graph entity disambiguation assistant responsible for deciding whether two entities are essentially the same entity.
Entity 1: {entity1}
Entity 2: {entity2}
Notes:
1. First use name and type to judge whether they may refer to the same entity, then use description details to confirm.
2. Plural forms, tense variation, and minor naming variants can still refer to the same entity.
3. Output only a JSON object, exactly in one of the following forms:
   {{"result": true}}
   {{"result": false}}
"""

judge_sim_entity_zh = """
你是一个知识图谱实体消歧助手，负责判断两个实体本质上是否是同一个实体。
实体1：{entity1}
实体2：{entity2}
注意：
1. 先基于 name 和 type 判断两者是否可能是同一实体；如果可能，再结合 description 进行精确判断。
2. 复数、时态变化、轻微命名差异不必然构成不同实体。
3. 只输出 JSON 对象，且只能是以下两种格式之一：
   {{"result": true}}
   {{"result": false}}
"""

question_entity_extract_prompt_zh = """
你是问题检索助手。请从问题中抽取用于知识图谱检索的核心实体和关键词。
问题：{question}

输出要求：
1. 只输出 JSON，不要输出 markdown 或额外解释。
2. 输出格式：
{{
  "entities": ["实体1", "实体2"],
  "keywords": ["关键词1", "关键词2"]
}}
3. entities 最多 8 个，并按重要性排序。
4. 如果问题中没有明确实体，entities 可以为空数组。
5. keywords 应尽量保留对检索有帮助的概念、动作或限定词。
"""

question_entity_extract_prompt_en = """
You are a question retrieval assistant. Extract the core entities and keywords needed for knowledge graph retrieval from the question.
Question: {question}

Output requirements:
1. Output JSON only. Do not output markdown or extra explanations.
2. Output format:
{{
  "entities": ["Entity 1", "Entity 2"],
  "keywords": ["Keyword 1", "Keyword 2"]
}}
3. Return at most 8 entities, ordered by importance.
4. If the question does not mention explicit entities, `entities` can be an empty array.
5. `keywords` should preserve concepts, actions, or constraints that are useful for retrieval.
"""

kg_qa_answer_prompt_zh = """
你是知识图谱问答助手。请基于给定的图谱上下文回答问题。

问题：
{question}

检索上下文：
{context}

候选图谱路径：
{graph_paths}

请严格输出 JSON（不要 markdown）：
{{
  "answer": "最终答案",
  "evidence_sources": [
    {{
      "source": "证据来源ID",
      "quote": "证据片段"
    }}
  ],
  "graph_paths": [
    "路径1",
    "路径2"
  ]
}}

约束：
1. 只基于给定上下文作答，不得编造。
2. 若上下文中存在 `source` 以 `chunk:` 开头的原文证据，优先使用这些原文证据支撑答案。
3. `evidence_sources` 至少给 1 条，且 `source` 必须来自上下文中的 `source` 字段。
4. `graph_paths` 至少给 1 条，且路径应来自候选图谱路径。
5. 如果证据不足，`answer` 必须明确写出“根据现有图谱证据不足以得出确定结论”。
"""

kg_qa_answer_prompt_en = """
You are a knowledge graph QA assistant. Answer the question based only on the provided graph context.

Question:
{question}

Retrieved Context:
{context}

Candidate Graph Paths:
{graph_paths}

Output strict JSON only (no markdown):
{{
  "answer": "Final answer",
  "evidence_sources": [
    {{
      "source": "Evidence source ID",
      "quote": "Evidence snippet"
    }}
  ],
  "graph_paths": [
    "Path 1",
    "Path 2"
  ]
}}

Constraints:
1. Answer only from the provided context. Do not invent facts.
2. If the context contains original evidence whose `source` starts with `chunk:`, prioritize that evidence.
3. `evidence_sources` must contain at least 1 item, and every `source` must come from the context.
4. `graph_paths` must contain at least 1 path, and each path must come from the candidate graph paths.
5. If the evidence is insufficient, `answer` must explicitly say: "The current graph evidence is insufficient to reach a definite conclusion."
"""


SUPPORTED_PROMPT_LANGUAGES = {"zh", "en"}

PROMPT_REGISTRY = {
    "text2entity": {
        "zh": text2entity_zh,
        "en": text2entity_en,
    },
    "entity_similarity": {
        "zh": judge_sim_entity_zh,
        "en": judge_sim_entity_en,
    },
    "entity_centric_kg": {
        "zh": extract_entiry_centric_kg_zh_v2,
        "en": extract_entiry_centric_kg_en_v2,
    },
    "qa_question_entity_extract": {
        "zh": question_entity_extract_prompt_zh,
        "en": question_entity_extract_prompt_en,
    },
    "qa_answer": {
        "zh": kg_qa_answer_prompt_zh,
        "en": kg_qa_answer_prompt_en,
    },
}


def normalize_prompt_language(prompt_language=None):
    """Normalize and validate prompt language."""
    language = prompt_language if prompt_language is not None else getattr(
        prompt_config,
        "PROMPT_LANGUAGE",
        None,
    )
    language = str(language or "").strip().lower()
    if language not in SUPPORTED_PROMPT_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_PROMPT_LANGUAGES))
        raise ValueError(
            f"Unsupported PROMPT_LANGUAGE={language!r}. Supported values: {supported}."
        )
    return language


def get_prompt(prompt_key, prompt_language=None):
    """Return the prompt template for the semantic key and active language."""
    registry_entry = PROMPT_REGISTRY.get(prompt_key)
    if registry_entry is None:
        supported_keys = ", ".join(sorted(PROMPT_REGISTRY))
        raise ValueError(
            f"Unknown prompt key {prompt_key!r}. Supported keys: {supported_keys}."
        )

    language = normalize_prompt_language(prompt_language)
    prompt_template = registry_entry.get(language)
    if prompt_template is None:
        raise ValueError(
            f"Prompt key {prompt_key!r} does not support language {language!r}."
        )
    return prompt_template
