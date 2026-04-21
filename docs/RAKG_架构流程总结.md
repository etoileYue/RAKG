# RAKG 架构流程总结
## 1. 总览：项目里已经落地的两条主链路

当前仓库可以明确分成两条主链路。

第一条是知识图谱构建链路：

`TextProcessor -> 逐句 NER -> 文档内实体消歧 -> 与已有图谱对齐 -> 实体中心关系抽取 -> 图谱转换 -> 图谱合并`

第二条是图谱问答链路：

`问题理解 -> 图节点匹配 -> 图邻域扩展 -> 证据上下文构建 -> LLM 作答`

这两条链路在代码里由同一个类 `NER_Agent` 对外统一暴露。`NER_Agent` 继承了 `NERPipeline` 与 `KnowledgeGraphQA`，因此既能做构图，也能做基于图谱的检索问答。

## 2. 核心接口与统一数据结构

当前主接口有四个：

- `NER_Agent.process(...)`
- `NER_Agent.process_all_topics(...)`
- `initialize_qa_graph_index(...)`
- `answer_question_with_kg(...)`

其中，构图阶段最终统一落到一个标准图结构上：

```json
{
  "entities": [
    {
      "name": "实体名",
      "type": "实体类型",
      "description": "实体描述",
      "attributes": {},
      "aliases": [],
      "provenance": {
        "chunk_ids": ["Topic1"]
      }
    }
  ],
  "relations": [
    {
      "source": "源实体",
      "relation": "关系名",
      "target": "目标实体",
      "description": "关系描述",
      "provenance": {
        "chunk_ids": ["Topic1", "Topic2"],
        "strategy": "heuristic_sentence_match"
      }
    }
  ],
  "chunk_map": {
    "Topic1": "原始句子"
  }
}
```

也就是说，当前图谱接口不是“裸三元组列表”，而是 `entities + relations + chunk_map` 三部分并存：

- `entities` 保存规范化后的实体节点
- `relations` 保存带 provenance 的关系记录
- `chunk_map` 保存 `chunk_id -> 原文句子` 的映射，供溯源与 QA 检索复用

## 3. 构图全过程

### 3.1 `NER_Agent.process()` 是单文档构图编排入口

`src/kgAgent.py` 中的 `NER_Agent.process()` 负责单个 topic 的完整执行。它的顺序很固定：

1. 用 `TextProcessor.process()` 切句并生成句向量。
2. 用 `extract_from_text_multiply()` 做逐句实体抽取。
3. 用 `similarity_result()` 与 `entity_Disambiguation()` 做文档内实体消歧。
4. 如果调用方传入 `existing_kg`，先执行 `align_entities_to_existing_graph()` 做跨图实体对齐。
5. 对每个标准实体调用 `get_target_kg_all()` 做实体中心关系抽取。
6. 用 `convert_knowledge_graph()` 把 LLM 子图转成统一图结构。
7. 如存在 `existing_kg`，再用 `merge_knowledge_graphs()` 做增量融合。

函数返回值里同时保留：

- `knowledge_graph`：如果传入已有图谱，则这是合并后的图
- `current_doc_kg`：仅当前文档抽出的图
- `alias_resolution`：跨图对齐阶段生成的别名到标准名映射

### 3.2 `TextProcessor.process()`：切句、`chunk_id` 与句向量

预处理在 `src/textProcess.py` 的 `TextProcessor.process()` 中完成。

第一步是切句。`split_sentences()` 使用中英文句末标点规则分句，得到 `sentences`。

第二步是给每个句子分配稳定 `chunk_id`。这里当前实现的规则不是“`topic + '_' + index`”，而是：

`chunk_id = base_name + (index + 1)`

其中 `base_name` 就是传入的 topic 名称。因此文档主题为 `Einstein` 时，句子 ID 会长成 `Einstein1`、`Einstein2` 这种形式。

第三步是向量化。`process()` 会对所有句子调用 embedding 模型，得到 `vectors`，并返回四项基础结果：

- `sentences`
- `vectors`
- `sentence_to_id`
- `id_to_sentence`

后续 NER、关系抽取、图谱 provenance 和 QA 证据都依赖这套映射。

### 3.3 `extract_from_text_multiply()`：逐句实体抽取与窗口补丁

实体抽取落在 `src/pipeline/relation_ops.py` 的 `extract_from_text_multiply()`。

这里的实现不是一次性对整篇文档做 NER，而是逐句调用 `extract_from_text_single()`。`extract_from_text_single()` 会把单句文本送入 `text2entity_en` prompt，请模型返回 JSON 结构的实体集合；如果输出被截断或不是合法 JSON，代码会回退为 `{"State": False, ...}` 的占位结构，而不是让整个流程中断。

逐句抽取之外，当前版本还有一个已经落地的“窗口补丁”机制：

- 开关：`RAKG_ENABLE_NER_WINDOW_PATCH`，默认开启
- 触发条件：句子较短，且包含代词或指代词
- 做法：把当前句与前后各一条句子拼成一个窗口，再额外调用一次 `extract_from_text_single()`
- 合并方式：`_merge_ner_results()` 会按 `name + type + description` 去重并合并两次抽取结果

这意味着当前实现已经显式处理“单句太短、指代信息不完整”的情况，而不是纯粹逐句盲抽。

最终，`extract_from_text_multiply()` 还会做两件事：

- `rewrite()`：把实体重新编号成 `entity1`、`entity2` 这类内部 ID
- `add_chunkid()`：给每个实体挂上来源 `chunkid`

因此，进入下一阶段的实体记录，至少具备：

- `name`
- `type`
- `description`
- `chunkid`

### 3.4 `similarity_result()` + `entity_Disambiguation()`：文档内实体消歧

文档内实体消歧分成“候选生成”和“实体簇合并”两步。

第一步在 `src/pipeline/similarity_ops.py` 的 `similarity_result()`。它先调用 `similarity_candidates()`，把每个实体编码为 `name + type` 文本后做 embedding 相似度计算，筛出超过阈值的候选实体对。

第二步是两轮 LLM 判定。`similarity_result()` 内部调用 `_run_two_pass_similarity_disambiguation()`，对候选对执行：

1. 第一轮 `similarity_llm_single()` 判定是否同一实体
2. 对 `needs_review` 的灰区样本再跑一轮同样的判定
3. 保留最终正匹配的实体对

这里的判定不是只看字符串相似，而是把实体整体对象传给 prompt，其中包含 `name`、`type`、`description`。

第三步在 `src/pipeline/entity_ops.py` 的 `entity_Disambiguation()`。它使用并查集把两两匹配对扩展成实体簇，然后按簇进行合并：

- 合并 `description`
- 合并 `chunkid`
- 把被吞并实体名写入 `aliases`
- 保留簇中的一个主实体作为标准记录

如果没有传入 `existing_kg`，`NER_Agent.process()` 在这一步之后还会再调用 `_collapse_entities_by_name()`，继续合并同名实体并清洗别名。

### 3.5 跨文档输入进入单文档流程的位置

`existing_kg` 是当前增量构图的入口。`NER_Agent.process()` 一开始就会通过 `_normalize_graph_input()` 规范化它，然后在文档内消歧完成后进入跨图对齐阶段。

只要 `existing_kg` 中存在实体，当前 topic 的标准实体集合就不会直接进入关系抽取，而是先送到 `align_entities_to_existing_graph()`。

### 3.6 `align_entities_to_existing_graph()`：把新实体对齐到旧图标准名

`src/pipeline/graph_ops.py` 中的 `align_entities_to_existing_graph()` 做的是“新实体对齐到已有图谱的标准节点”，而不是简单把两张图拼接。

它的具体步骤是：

1. `_build_existing_entity_lookup()` 把已有图谱实体整理成可匹配表，保留 `name/type/description/aliases`
2. `cross_similarity_result()` 在“新实体 vs 已有实体”之间跑 embedding 候选筛选 + 两轮 LLM 判定
3. 对匹配成功的新实体，直接把 `name` 改写为已有图中的标准名
4. 若旧图上的 `type` 更可靠，就覆盖新实体的类型
5. 合并旧描述与新描述
6. 把原名、旧别名和新别名统一折叠进 `aliases`

这个函数还会生成 `alias_resolution`，把：

- 新实体原名
- 旧图已有别名
- 新实体新增别名

全部映射到标准名，并把它作为 `process()` 的返回值带出去。

需要注意的是，当前代码里 `alias_resolution` 主要是“结果输出给调用方”和“表达这次对齐结论”；后续图谱真正的节点统一，仍主要依赖 `merge_knowledge_graphs()` 内部的别名解析与实体合并逻辑。

### 3.7 `get_target_kg_all()`：实体中心关系抽取

关系抽取采用“实体中心子图”模式，入口是 `src/pipeline/relation_ops.py` 的 `get_target_kg_all()`，它会遍历所有标准实体，并对每个实体调用 `get_target_kg_single()`。

当前实现里，每个实体的关系抽取上下文由三块证据共同组成。

第一块是实体原始命中句。`get_target_kg_single()` 会从实体记录里的 `chunkid` 出发，把这些句子视为核心证据。

第二块是向量检索补充句。若 `RAKG_ENABLE_EVIDENCE_CONTEXT` 开启，`_build_relation_context()` 会调用 `get_retriever_context()`：

- 用实体名作为主查询
- 同时把实体别名、描述作为约束
- 在整篇文档的句向量上做 Top-K 检索
- 默认启用 MMR 去重

第三块是邻近句补充。`_build_relation_context()` 不只保留命中句和检索句，还会把这些句子在原文顺序上的前后邻居句一起纳入候选证据。

最终上下文会整理成带元信息的 evidence block 文本，其中每个 block 都标出：

- `chunk_id`
- `source`
- `similarity`
- `rank`

这一步使得后续 prompt 可以直接引用 `chunk_id` 做关系 provenance。

### 3.8 `build_related_kg_context()`：已有图谱如何反哺当前关系抽取

如果 `process()` 收到了 `existing_kg`，那么在关系抽取前，还会为每个已对齐实体构造 `related_kg_map`。

构造方式是调用 `src/pipeline/graph_ops.py` 的 `build_related_kg_context()`。它会：

1. 把已有图谱实体和别名先规范化
2. 用别名解析把当前实体名定位到已有图谱中的标准节点
3. 抽出该节点本身的标准化实体信息
4. 收集所有与该节点直接相连的关系，默认最多 20 条

返回结果是：

- `central_entity`
- `related_relations`

随后，`get_target_kg_single()` 会把这份结构序列化成 `related_kg` 传入 `extract_entiry_centric_kg_en_v2` prompt。

因此，当前关系抽取的真实上下文不是只有“当前文档文本”，而是：

- 当前实体的原文证据
- 当前文档内的相似句与邻近句
- 可选的已有图谱局部邻域

### 3.9 Prompt 对关系输出的约束

`src/prompt.py` 中的 `extract_entiry_centric_kg_en_v2` 明确要求模型：

- 只围绕指定中心实体建子图
- 关系头实体必须是当前中心实体
- 每条关系都要给出 `provenance.chunk_ids`
- `chunk_ids` 必须来自提供的 evidence blocks
- 没有证据的关系不能输出
- 遇到代词时要先在证据块里解析其指代对象

因此，当前实现并不是“抽出关系就算完成”，而是显式要求关系绑定证据来源。

### 3.10 `convert_knowledge_graph()`：把实体中心子图统一成标准图

LLM 返回的是“一个中心实体对应一个局部子图”的结构，真正变成统一图结构是在 `src/pipeline/graph_ops.py` 的 `convert_knowledge_graph()`。

它主要做四类整理：

1. 把所有中心实体写入 `entities`
2. 把 LLM 返回的候选句 `candidate_chunks` 回填到全局 `chunk_map`
3. 把 `relationships` 转成标准关系记录对象
4. 如果关系目标实体尚不存在，则补出目标实体节点

这里关系记录的当前标准格式是：

```json
{
  "source": "中心实体",
  "relation": "关系名",
  "target": "目标实体",
  "description": "关系描述",
  "provenance": {
    "chunk_ids": ["Topic3"],
    "strategy": "llm_relation_provenance"
  }
}
```

如果模型没有给出可用的 `provenance.chunk_ids`，`convert_knowledge_graph()` 不会直接丢弃这条边，而是调用 `_infer_relation_provenance()`，在候选句里用启发式规则补推来源句。

### 3.11 `merge_knowledge_graphs()`：增量融合的最终落点

`merge_knowledge_graphs()` 是当前统一图合并的总入口，它会同时处理：

- `chunk_map` 合并
- 实体合并
- 关系合并

实体合并时，它会先基于当前实体注册表构建 `alias_to_canonical`，然后按以下顺序找归宿：

1. 实体名能否直接解析到已有标准名
2. 实体别名能否解析到已有标准名
3. 如果都不能，再作为新实体入图

合并后会保留并整合：

- `type`
- `description`
- `attributes`
- `aliases`
- `provenance`

关系合并时，则以：

`(source, normalized_relation, target)`

作为去重键，同时合并：

- `description`
- `provenance`

所以当前仓库里的“增量融合”不是简单追加列表，而是带别名解析、实体归并和关系 provenance 合并的结构化融合。

## 4. 跨文档构图：当前实现机制与当前边界

### 4.1 已实现的机制

当前代码已经实现了“新文档在已有知识图谱基础上增量构图”的完整闭环，顺序是：

1. 调用方把旧图作为 `existing_kg` 传入 `NER_Agent.process(...)`
2. 新文档实体先做文档内消歧
3. 再通过 `align_entities_to_existing_graph()` 对齐到旧图标准名
4. 再通过 `build_related_kg_context()` 把旧图局部邻域反哺给当前关系抽取
5. 当前文档子图生成后，最终通过 `merge_knowledge_graphs()` 并回旧图

也就是说，跨文档能力已经不只是“最后把两张图并起来”，而是旧图会参与新文档的实体规范化与关系抽取。

### 4.2 当前边界：`process_all_topics()` 默认不自动串行累积

`src/kgAgent.py` 的 `process_all_topics()` 支持把 `existing_kg` 传给每个 topic，但它当前的行为边界也很明确：

- 如果 `existing_kg` 带有 `global` 键，则每个 topic 都使用同一个全局已有图
- 否则，它会按 `existing_kg.get(idx)` 为当前 topic 取一个外部指定图

但它不会把第 `i` 篇 topic 刚生成的输出，自动更新成第 `i+1` 篇 topic 的输入。换句话说，当前默认流程不是“文档 1 输出自动喂给文档 2，再自动喂给文档 3”的串行全局累积。

因此，跨文档全局累积在当前版本里需要调用方显式做两种事之一：

- 传入一个固定的全局底座图 `existing_kg["global"]`
- 或在外层循环里自己把上一次输出继续作为下一次的 `existing_kg`

如果调用方什么都不传，默认行为仍然是“每篇文档独立构图”。

## 5. 检索问答过程

### 5.1 `initialize_qa_graph_index()`：图谱标准化、索引化与向量缓存

问答前，代码会先通过 `src/pipeline/qa_graph_ops.py` 的 `initialize_qa_graph_index()` 建立 QA 图索引。

这个函数会做三件事。

第一件事是规范化图输入。无论传入的是 dict、JSON 字符串还是 JSON 文件路径，都会先转成统一图对象。

第二件事是建立结构索引。`_build_graph_indices()` 会产出：

- `entity_lookup`
- `relations`
- `adjacency_out`
- `adjacency_in`

其中实体和关系上的 `provenance` 都会被标准化成 `chunk_ids` 结构，兼容旧字段。

第三件事是预计算节点向量。它会把每个实体转写成 `_entity_to_retrieval_text()`，其内容包含：

- `name`
- `type`
- `description`
- `attributes`
- `aliases`

然后统一计算 `node_vectors`，缓存到 `_qa_graph_index_cache`。

因此，QA 阶段不是每次临时扫描原始图，而是基于“邻接表 + 实体查找表 + 节点向量”的索引来检索。

### 5.2 `extract_question_entities()`：问题理解

问题理解在 `src/pipeline/qa_match_ops.py` 的 `extract_question_entities()`。

默认路径是调用 `question_entity_extract_prompt_cn`，要求模型输出：

```json
{
  "entities": ["实体1", "实体2"],
  "keywords": ["关键词1", "关键词2"]
}
```

代码会把 `entities` 与 `keywords` 合并、去重，并截到最多 8 个。

如果 LLM 解析失败，当前实现不会终止，而是退化为正则抽取中文片段或英数字 token。也就是说，“问题理解”是有容错回退的。

### 5.3 `match_question_entities_to_graph()`：字符串匹配 + 向量相似匹配

问题实体抽出后，`match_question_entities_to_graph()` 会把它们映射到图谱节点，形成 seed nodes。

当前实现是两段式匹配。

第一段是字符串匹配，支持：

- 标准名精确命中
- 别名精确命中
- 标准名或别名与问题实体的包含匹配

第二段是语义匹配。如果已经预计算好 `node_vectors`，函数会对问题实体做 embedding，再与所有图节点向量做余弦相似度比较，保留超过阈值的节点。

如果前面两段都没形成候选，并且节点向量可用，代码还会用“整个问题”再做一次语义 fallback，至少返回 Top-K 节点。

返回值 `matched_nodes` 中会保留：

- `name`
- `score`
- `reason`
- `type`
- `description`

因此，种子节点不仅是命中结果，也带有“为什么命中”的解释信息。

### 5.4 `expand_graph_neighbors()`：1 到 2 跳图邻域扩展

真正把种子节点变成检索证据的是 `src/pipeline/qa_context_ops.py` 的 `expand_graph_neighbors()`。

它会围绕 seed nodes 做 BFS 风格扩展，并把最大跳数限制在 1 或 2 跳：

- `max_hop <= 1` 时实际只走 1 跳
- 其他情况统一按 2 跳处理

扩展时同时遍历：

- 出边 `adjacency_out`
- 入边 `adjacency_in`

在遍历过程中，它会收集四类结果。

第一类是 `chunk_evidence`。来源包括：

- 种子实体自身 provenance 指向的 `chunk_ids`
- 关系 provenance 指向的 `chunk_ids`

这些 `chunk_ids` 会再回到 `chunk_map` 中取原始句子。

第二类是 `entity_evidence`，即实体描述和前几个属性值。

第三类是 `relation_evidence`，即边描述；如果边本身没有描述，就退化成 `source --[relation]-> target` 文本。

第四类是 `graph_paths`，例如：

- `A --[r]-> B`
- `A <-[r]-- C`

这些路径既是解释线索，也是后续回答 prompt 的输入。

### 5.5 `build_qa_retrieval_context()`：证据上下文拼装

`build_qa_retrieval_context()` 把前面几步串起来：

1. `extract_question_entities()` 抽问题实体与关键词
2. `match_question_entities_to_graph()` 匹配出 `matched_nodes`
3. 取节点名形成 `seed_nodes`
4. `expand_graph_neighbors()` 扩出图证据
5. 把证据压平为 `evidence_items`
6. 把证据格式化成 `context_text`

`context_text` 当前是面向 LLM 的纯文本证据清单，每条都带 `source=` 标识，若有路径还会补 `path=`。这一步的作用是把图结构证据转写成模型可直接消费的检索上下文，同时保留证据来源 ID。

### 5.6 `answer_question_with_kg()`：检索与最终作答是两个阶段

`answer_question_with_kg()` 本身只负责两件事：

1. 先调用 `build_qa_retrieval_context()` 完成图谱检索
2. 再把 `question`、`context_text`、`graph_paths` 交给 `kg_qa_answer_prompt_cn` 生成答案

也就是说，当前 QA 流程在工程上明确拆成两个阶段：

- 第一阶段是图谱检索与证据组织
- 第二阶段才是基于证据的 LLM 作答

`kg_qa_answer_prompt_cn` 强制模型输出 JSON，字段包括：

- `answer`
- `evidence_sources`
- `graph_paths`

并且 prompt 明确约束：

- 只能基于给定上下文作答
- 优先引用 `chunk:` 开头的原文证据
- `evidence_sources.source` 必须来自上下文
- `graph_paths` 必须来自候选路径
- 如果证据不足，答案必须写成“根据现有图谱证据不足以得出确定结论”

代码侧还做了额外兜底：如果 LLM 没有稳定返回合法 JSON，就回退为原始文本答案，并自动从检索结果里补证据和路径。

最终返回值会同时保留：

- `retrieval`
- `llm_output_raw`
- `answer`
- `evidence_sources`
- `graph_paths`
- `formatted_answer`

所以，当前问答输出不是单一句子，而是“答案 + 证据来源 + 图路径”的组合结果。

## 6. 结论：当前仓库里真正实现的 RAKG 主链路

如果只按当前代码事实来概括，RAKG 已经落地的是这样一条闭环：

首先，文本被切成有稳定 `chunk_id` 的句子，并生成句向量。接着系统逐句做实体抽取，对短句和代词句用窗口补丁补上下文，再通过 embedding 候选筛选和两轮 LLM 判定完成文档内实体消歧。

之后，标准实体会在需要时与已有图谱对齐到统一标准名，并带着当前文档证据和旧图局部邻域去做实体中心关系抽取。抽出的局部子图再被转换成统一的 `entities + relations + chunk_map` 图结构，并通过别名解析、实体归并和关系 provenance 合并，实现增量融合。

在问答侧，系统先把知识图谱建成适合检索的邻接索引和节点向量索引，再把问题映射到图中的 seed nodes，扩展 1 到 2 跳邻域，拼出带 `source` 和 `path` 的证据上下文，最后才由 LLM 基于这些证据生成 JSON 化答案。

因此，当前版本的 RAKG 不是“先抽三元组，再随便问答”的松耦合流程，而是一个已经把文本证据、实体规范化、图结构融合与证据约束问答连接起来的实现版本。同时，它的跨文档能力也有明确边界：增量融合机制已经具备，但多文档全局串行累积仍需调用方显式组织 `existing_kg` 的传递与更新。
