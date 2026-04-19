从你现在这套实现看，问答效果不理想，核心原因大概率不是“模型不够强”，而是这三件事叠在一起了：图谱本身不够完整、QA 检索到图的方式过粗、最终回答阶段没有回到原文做兜底。换句话说，你现在更像是“基于 KG 的 QA”，还没有真正做到“KG 增强的 RAG”。

**优先建议**
- 第一优先级是把 QA 从“只吃图证据”改成“图引导文本证据”。当前问答阶段只会先匹配图节点、扩邻居、再把图证据喂给 LLM，没有回查原始句子或段落的链路，[qa_context_ops.py](/home/ile/code/RAKG/src/pipeline/qa_context_ops.py:149)、[qa_answer_ops.py](/home/ile/code/RAKG/src/pipeline/qa_answer_ops.py:57)。这会导致一个典型问题：图里稍微缺一条边，答案就直接掉到“不足以确定”。更合理的架构是：先用 KG 找种子实体和候选路径，再反向召回原文 chunk/段落做二次证据融合，最后由 LLM 基于“图路径 + 原文片段”回答。KG 应该负责约束检索方向，而不是替代文本证据。
- 第二优先级是增强“问题感知”的路径检索，而不是 BFS 式扩邻居。你现在的做法是先取 top-k 种子，再固定 1-2 hop 扩展，然后把前 30 条证据截断给模型，[qa_match_ops.py](/home/ile/code/RAKG/src/pipeline/qa_match_ops.py:45)、[qa_context_ops.py](/home/ile/code/RAKG/src/pipeline/qa_context_ops.py:9)。这对关系型问题、多跳问题、带约束问题都不够。建议改成“候选路径生成 + 问题相关性打分 + rerank”的两阶段检索。尤其要单独建 relation/path 索引，因为很多问题真正检索的是“关系模式”，不是实体名本身。
- 第三优先级是给图谱加 provenance。当前构图时句子 `chunkid` 只在前面短暂存在，转换成最终 KG 后，实体和关系基本只剩 name/type/description/attributes/relation_description 这类二次生成结果，[relation_ops.py](/home/ile/code/RAKG/src/pipeline/relation_ops.py:94)、[graph_ops.py](/home/ile/code/RAKG/src/pipeline/graph_ops.py:124)。这会让你后面很难做高质量引用、错误定位和证据验证。建议每个 entity/relation 都保留 `doc_id`、`chunk_id`、原文 span 或原句列表、抽取置信度、抽取来源。没有 provenance 的 KG，很难支撑稳定 QA。
- 你现在的构图粒度偏细，但上下文组织又偏粗，这两头都伤效果。NER 是逐句抽实体的，[relation_ops.py](/home/ile/code/RAKG/src/pipeline/relation_ops.py:72)；关系抽取时又把命中句和语义召回句直接拼成一个逗号分隔的大串，[relation_ops.py](/home/ile/code/RAKG/src/pipeline/relation_ops.py:106)。这很容易丢掉跨句关系、代词指代、事件上下文，也会让模型把几句无关相似句误融合。更合理的是分层处理：句子抽候选实体，段落/局部窗口抽关系，文档级做实体归并和补全。
- 关系和类型需要 schema 化，否则图会越来越碎。当前关系去重只做了小写和空白归一，[graph_ops.py](/home/ile/code/RAKG/src/pipeline/graph_ops.py:11)；相似实体候选也主要基于 `name + type` 向量，[similarity_ops.py](/home/ile/code/RAKG/src/pipeline/similarity_ops.py:129)。这意味着“毕业于/就读于/曾在…学习”“作者/撰写者”“所属/隶属”这类同义关系会被拆散，后续路径召回自然就差。建议加一层受控 schema：实体类型本体、关系词表、属性词表、同义映射。你这个系统越往多文档、多轮融合走，这层越关键。
- 现有跨文档能力没有真正变成“全局图谱记忆”。`process_all_topics()` 里虽然能吃 `existing_kg`，但不会把第 i 篇输出自动滚入第 i+1 篇作为新底座，[kgAgent.py](/home/ile/code/RAKG/src/kgAgent.py:220)。这意味着如果调用方不手动串起来，系统其实更像“每篇文档独立构图”。如果你的问答依赖跨文档补全，这会非常伤。建议把全局 KG 明确成一个长期状态：每篇文档处理后立即 merge，实体索引和 QA 索引也增量更新。
- 建议把“文本中观察到的事实”和“由已有 KG 推导/补出的事实”分层存储。你当前关系抽取 prompt 明确鼓励模型基于相关 KG 建反向关系，[prompt.py](/home/ile/code/RAKG/src/prompt.py:78)。这对召回有帮助，但也很容易把“文本明示事实”和“推断出来的补充边”混在一起。一旦混在同一层，问答就容易显得有理但不够真。更稳的架构是把边分成 `observed`、`normalized`、`inferred`、`imported` 四类；回答时优先使用 observed，其余只作为候选补充。
- 问题理解现在太偏“实体抽取”，不够“查询分解”。当前只抽 `entities + keywords`，[qa_match_ops.py](/home/ile/code/RAKG/src/pipeline/qa_match_ops.py:20)。但很多问题真正决定检索质量的是：答案类型、关系意图、时间约束、否定/比较、期望 hop 数。建议把 query planner 独立出来，先判断这是实体属性问答、关系问答、多跳推理、比较问答还是证据不足型问题，再决定走哪条检索路径和多大 hop。
- 中文场景下，抽取和消歧 prompt 目前主要还是英文模板，[prompt.py](/home/ile/code/RAKG/src/prompt.py:256)、[prompt.py](/home/ile/code/RAKG/src/prompt.py:326)，而 QA prompt 是中文，[prompt.py](/home/ile/code/RAKG/src/prompt.py:351)。如果你的语料主要是中文，这种“抽取阶段英文 schema、回答阶段中文 schema”的混搭，通常会损伤结构化抽取稳定性。它不是最核心的问题，但会持续拖后腿。

**我认为最值当先做的三件事**
1. 把 QA 改成“KG 找路径，原文给证据，LLM 做融合回答”。
2. 给 entity/relation 加 provenance 和置信度，把最终引用落回原句而不是只引用 KG 描述。
3. 把路径检索改成 question-aware rerank，而不是固定 seed + 固定 hop + 固定截断。

**如果按收益/投入比排序**
- 最高收益：混合检索架构、路径 rerank、provenance。
- 中期关键：schema 化关系/类型、全局 KG 增量更新。
- 后续增强：查询分解、observed/inferred 分层、按领域定制抽取模板。

如果你愿意，我下一步可以继续基于你这套代码结构，给你画一版“更适合 RAKG 的目标架构图”和“最小改造路线图”，我会尽量贴着你现有模块命名来讲，不会空泛。