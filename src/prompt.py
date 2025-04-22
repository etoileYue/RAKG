

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
6. The final output format should be:
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
            "target_type": "{{}}"
            "target_description": "{{}}"
            "relation_description": "{{}}"
        }},
        ...
        {{
            "relation": "{{}}",
            "target_name": "{{}}",
            "target_type": "{{}}"
            "target_description": "{{}}"
            "relation_description": "{{}}"
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
          "relation_description": "Einstein proposed the Theory of Relativity, which is an important theory in modern physics."
        }},
        {{
          "relation": "Graduated From",
          "target_name": "ETH Zurich",
          "target_type": "Educational Institution",
          "target_description": "ETH Zurich is a university located near Zurich.",
          "relation_description": "Einstein studied at ETH Zurich."
        }}
      ]
    }}
  }}
"""

extract_entiry_centric_kg_cn_v2 = """
    你是一个知识图谱提取助手，结合其他相关知识图谱的知识，负责从文本中抽取与指定实体有关的属性和关系，
    文本：{text}
    指定实体：{target_entity}
    相关知识图谱：{related_kg}
    对你的要求：
    1、你应当综合整个文本，综合地抽取与指定实体有关的关系，针对指定实体建立一个子图谱
    2、你应当抽取指定实体的属性和指定实体与其他实体的关系，
        对于属性的抽取：属性是对指定实体特性的描述，例如：乔丹——性别：男，这里的性别就是属性
        对于关系的抽取并且这个关系的头实体必须是指定实体，例如： “指定实体——拥有——其他实体”是合法的，“其他实体——被拥有——指定实体”是非法的
    3、你应当分别什么时候把信息归类成关系、什么时候把信息归类成属性
    4、对于其他相关知识图谱的知识，这些知识有助于你更全面的理解指定实体的特征，并且，你应该结合其他知识对指定实体的关系，反向建立关系，形成双向关系，例如：其他实体——妻子——指定实体，那么你就应该建立：指定实体——丈夫——其他实体，这个反向关系，使得知识图谱更加全面
    5、最后的输出中，相同的attributes应当只保留一个，相同的relationships应当只保留一个
    6、最后的输出格式应当为：
        {{
    "central_entity": {{
        "name": "{{}}",
        "type": "{{}}",
        "description": "{{}}",
        "attributes": "{{}}",
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
            "target_description": "{{}}"
            "relation_description": "{{}}"
        }},
        ...
        {{
            "relation": "{{}}",
            "target_name": "{{}}",
            "target_type": "{{}}"
            "target_description": "{{}}"
            "relation_description": "{{}}"
        }}
        ]
      }}
    }}
    例如：
    {{
  "central_entity": {{
    "name": "阿尔伯特·爱因斯坦",
    "type": "人物",
    "description": "阿尔伯特·爱因斯坦被公认为是继牛顿之后最伟大的物理学家之一",
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
        "target_description": "相对论是由爱因斯坦于 1905 年提出来的一种物理理论，它认为物体在运动过程中，其空间和时间的变换是相互关联的，而不是相互独立的。",
        "relation_description": "爱因斯坦提出了相对论，这是现代物理学的重要理论"
      }},
      {{
        "relation": "毕业于",
        "target_name": "苏黎世联邦理工学院",
        "target_type": "教育机构",
        "target_description": "苏黎世联邦理工学院是一个位于苏黎世附近的大学",
        "relation_description": "爱因斯坦曾就读于苏黎世联邦理工学院"
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
1. First, you should determine whether the text contains any information. If it's just meaningless symbols, directly output: {{State : False}}. If the text contains information, proceed to the next step.
2. You should consider the entire text for named entity recognition.
3. The identified entities should consist of three parts: name, type, and description.
    - name: The main subject of the named entity.
    - type: The category of the subject.
    - description: A summary description of the subject, explaining what it is.
4. Since multiple named entities may be identified in a single text, you need to output them in a specific format.
    The output format should be:
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
        }},
        ...
        "entityn": {{
            "name": "Entity Name n",
            "type": "Entity Type n",
            "description": "Entity Description n"
        }}
    }}
"""

text2entity_cn = """
你是一个命名实体识别助手，负责从给定的文本中识别命名实体
文本：{text}
注意：
1、首先你应当判断这个文本是否包含任何信息，如果只是没有意义的符号，那么直接输出:{{State : Fasle}}，如果文本中存在着信息，那么进行下一步
2、你要综合考虑整个文本进行命名实体识别
3、命名的实体需要由三个部分组成分别是：name、type、description
    name是命名实体的主体
    type是这个主体的类别
    description是这个主体的描述，概括性的描述这个主体是什么
4、由于一段文本中可能识别出多个命名实体，所以你需要以一定的格式来输出
    输出的格式为：
    {{
        "entity1":{{
            "name": "实体名称1",
            "type": "实体类型1",
            "description": "实体描述1"
        }},
        "entity2":{{
            "name": "实体名称2",
            "type": "实体类型2",
            "description": "实体描述2"
        }},
        ...
        "entityn":{{
            "name": "实体名称n",
            "type": "实体类型n",
            "description": "实体描述n"
        }}
    }}
"""

fewshot_for_ext2entity = """

"""



judge_sim_entity_en = """
    You are a knowledge graph entity disambiguation assistant responsible for determining whether two entities are essentially the same entity. For example:
    Entity 1: "name": "Henan Business Daily", "type": "Media Organization", "description": "A commercial newspaper in Henan Province that provides news and information reporting." and Entity 2: "name": "Top News·Henan Business Daily", "type": "Organization Name", "description": "A news media organization located in Henan Province, responsible for reporting important local and national news and information."
    Essentially, they are the same entity.
    Entity 1: {entity1}
    Entity 2: {entity2}
    Notes:
    1. You should initially judge whether the two entities might be the same based on their names and types, and if they might be the same, analyze their descriptions in detail to determine if they are indeed the same.
    2. Your output format should be "yes" if you determine that they are the same entity, outputting: {{'result': True}}, and if you determine that they are not the same entity, outputting: {{'result': False}}.
"""

judge_sim_entity_cn = '''
    你是一个知识图谱实体消歧助手，负责判断两个实体本质上是否是同一个实体，例如：
    实体1："name": "河南商报", "type": "媒体机构", "description": "河南省的一家商业报纸，提供新闻和信息报道。"和实体2："name": "顶端新闻·河南商报", "type": "组织名", "description": "一家位于河南省的新闻媒体机构，负责报道地方及全国的重要新闻和信息。
    并且，不同实体，实体的复数、不同时态，都视为同一实体
    本质上是同一个实体
    实体1:{entity1}
    实体2:{entity2}
    注意：
    1、你应当通过name和type大体判断这两个实体是否可能相同，并且在可能相同的情况下，通过description具体分析是否相同
    2、你的输出格式应当为是，当判断确实是同一个实体时输出:{{'result':True}}，判断不是同一个实体时输出:{{'result':False}}
'''