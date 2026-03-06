# RAKG 模块分析说明

## 1. 文档范围
- 编排入口: `src/construct/RAKG.py`
- 核心能力: `src/kgAgent.py`

目标是梳理代码职责、处理流程、关键数据结构，并指出问题与改进方向。

## 2. 模块职责

### 2.1 `src/construct/RAKG.py`
负责端到端处理编排:
- 读取主题数据
- 调用 `TextProcessor` 进行文本切分与向量准备
- 调用 `NER_Agent` 执行实体抽取、消歧、KG 抽取
- 将结果转换并写入输出目录

### 2.2 `src/kgAgent.py`
封装实体与知识图谱抽取核心逻辑:
- 分段 NER (`extract_from_text_single` / `extract_from_text_multiply`)
- 候选相似实体筛选 + LLM 判重 (`similarity_candidates` / `similarity_result`)
- 并查集实体消歧 (`entity_Disambiguation`)
- 检索上下文与实体中心 KG 抽取 (`get_target_kg_single` / `get_target_kg_all`)
- 统一图结构转换 (`convert_knowledge_graph`)

## 3. 处理流程（端到端）
1. `process_all_topics(...)` 读取 topic 列表并迭代处理。  
   位置: `src/construct/RAKG.py:34`
2. 每个 topic 调用 `TextProcessor.process()` 产出: `sentences / sentence_to_id / id_to_sentence / vectors`。  
   位置: `src/construct/RAKG.py:62`
3. NER 阶段:
   - 正常路径: `extract_from_text_multiply(...)`
   - 跳过路径: `get_ner_result_from_file(...)` 从 JSONL 恢复
4. 相似实体候选筛选 + LLM 精判。  
   位置: `src/construct/RAKG.py:73`, `src/kgAgent.py:129`
5. 并查集合并实体（消歧）。  
   位置: `src/construct/RAKG.py:75`, `src/kgAgent.py:155`
6. 对每个实体抽取 entity-centric KG。  
   位置: `src/construct/RAKG.py:76`, `src/kgAgent.py:299`
7. 聚合为 `{entities, relations}` 统一结构。  
   位置: `src/construct/RAKG.py:79`, `src/kgAgent.py:313`
8. 结果落盘。  
   位置: `src/construct/RAKG.py:84`

## 4. 关键数据结构

### 4.1 `text_split`（来自 `TextProcessor.process()`）
预期字段:
- `sentences`: `list[str]`
- `sentence_to_id`: `dict[str, str]`
- `id_to_sentence`: `dict[str, str]`
- `vectors`: `list[list[float]]`

### 4.2 NER 中间结构
统一后形态（示意）:
```json
{
  "entity1": {
    "name": "...",
    "type": "...",
    "description": "...",
    "chunkid": "..."
  }
}
```

### 4.3 最终 KG 结构
```json
{
  "entities": [
    {
      "name": "...",
      "type": "...",
      "description": "...",
      "attributes": {}
    }
  ],
  "relations": [
    ["source", "relation", "target", "relation_description"]
  ]
}
```

## 5. 主要问题（按优先级）

### P0
1. 输出 JSON 被二次序列化，文件中保存的是 JSON 字符串，不是对象。  
   位置: `src/construct/RAKG.py:80`, `src/construct/RAKG.py:86`
2. `convert_to_valid_json` 将所有字符串中的 `'` 替换为 `"`，可能破坏文本语义，不是可靠的 JSON 修复手段。  
   位置: `src/construct/RAKG.py:21-24`

### P1
3. 相似度矩阵写入依赖 NumPy 已弃用的隐式数组到标量转换，未来版本可能报错。  
   位置: `src/kgAgent.py:110-112`
4. 异常处理粒度过粗，主流程中异常被吞掉，仅写日志，缺少结构化失败输出。  
   位置: `src/construct/RAKG.py:90-92`
5. 多处路径和参数硬编码，迁移环境或复现实验成本高。  
   位置: `src/construct/RAKG.py:68`, `:72`, `:76`, `:97-99`
6. JSONL 输出路径父目录未显式创建，可能在首次运行时失败。  
   位置: `src/kgAgent.py:53`, `:289`

### P2
7. 非确定性输出: 使用 `set()` 去重后直接拼接，句子和字段顺序不稳定，影响可复现性。  
   位置: `src/kgAgent.py:203-204`, `:276`
8. 可维护性问题: 冗余条件与未使用导入。  
   - `start_index` 恒为 1 (`src/construct/RAKG.py:46`)
   - `if entity_id in entity_dic` 在当前循环恒真 (`src/kgAgent.py:304-306`)
   - `re` 在 `kgAgent.py` 中未使用 (`src/kgAgent.py:6`)

## 6. 改进建议
1. 修复输出链路: `convert_knowledge_graph` 返回 dict 后直接 `json.dump(dict, ...)`，移除“先 `dumps` 再 `dump`”。
2. 删除 `convert_to_valid_json` 的单引号替换逻辑，改为严格 schema 校验 + 重试/降级。
3. 将相似度写入改为显式标量: `sim = cosine_similarity(...)[0][0]`。
4. 参数化配置: 路径、阈值、`top_k`、`done_offset`、跳过列表等迁移到配置文件或 CLI。
5. 增强鲁棒性: 对 LLM 输出增加 JSON Schema 或 Pydantic 校验，异常时记录可追踪上下文。
6. 提升可复现性: 对合并后的 `chunkid`、`description`、句子列表做稳定排序后再拼接。
7. 增加任务汇总报告: 成功数、失败数、失败原因、失败样本索引。
8. 轻量性能优化: 把 prompt/chain 构建挪到初始化，减少循环内重复创建。

## 7. 建议的后续落地顺序
1. 先做 P0（输出正确性）
2. 再做 P1（稳定性与可部署性）
3. 最后做 P2（可复现性与可维护性）
