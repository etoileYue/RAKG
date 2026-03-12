# evaluate_MINE_RAKG.py 与 evaluate_MINE_graphrag.py 说明文档

## 1. 说明范围
- [evaluate_MINE_RAKG.py](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py)
- [evaluate_MINE_graphrag.py](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_graphrag.py)

这两个脚本都用于评测图检索上下文是否覆盖标准答案，核心链路一致：
1. 为图节点生成 embedding。
2. 用答案文本检索 top-k 节点。
3. 从图中扩展上下文。
4. 调用 Ollama LLM 判断“上下文是否包含答案”并输出 `0/1`。
5. 统计 accuracy 并写入 JSON 文件。

## 2. 通用评测流程

### 2.1 关键依赖
- `networkx`：构建有向图。
- `sentence_transformers` + `cosine_similarity`：节点检索。
- `ollama.Client`：二分类评估。
- `src.config`：读取 `OLLAMA_BASE_URL` 与 `DEFAULT_MODEL`（graphrag 额外导入了 `EMBEDDING_MODEL`，但未实际使用）。

### 2.2 核心函数（两文件都有）
- `retrieve_relevant_nodes(...)`：按余弦相似度取 top-k（默认 8）。
- `retrieve_context(...)`：从候选节点向外按深度（默认 2）收集关系文本。
- `gpt_evaluate_response(...)`：提示模型仅返回 `"1"` 或 `"0"`。
- `evaluate_accuracy(...)`：逐题评估并计算总体准确率。

### 2.3 输出格式（两文件一致）
结果文件是 JSON 数组，前 N 条为逐题记录，最后一条为总体准确率：
```json
[
  {
    "correct_answer": "...",
    "retrieved_context": "...",
    "evaluation": 1
  },
  {
    "accuracy": "85.33%"
  }
]
```

## 3. evaluate_MINE_RAKG.py 细节

### 3.1 输入输出
- 图输入：`data/processed/RAKG_graph_v2_1/{i}.json`（第 285 行）。
- 结果输出：`data/processed/RAKG_graph_v2_1/{i}_results.json`（第 288 行）。
- 文件范围：`i=1..105`（第 284 行）。

### 3.2 图加载与上下文构造
- `load_graph_from_json`（第 13 行）支持节点属性：`type`、`description`、`attributes`。
- 关系为四元组：`source, rel, target, rel_description`（第 34 行）。
- `retrieve_context` 中会拼接节点信息和关系描述（第 72-93 行），上下文信息相对更丰富。

### 3.3 已识别注意点
- `json.loads` 被执行两次（第 16、18 行），意味着输入文件内容是“JSON 字符串包裹 JSON”格式；若输入改成普通 JSON 会失败。
- 写结果前未创建目录（第 166 行直接写文件），依赖目标目录已存在。
- 导入了 `os` 但未使用（第 7 行）。

## 4. evaluate_MINE_graphrag.py 细节

### 4.1 输入输出
- 图输入目录：`data/graphrag_qwen/ragtest{i}/output`（第 296 行）。
- 从该目录读取：
  - `entities.parquet`（第 47 行）
  - `relationships.parquet`（第 48 行）
- 结果输出：`data/processed/graphrag_graph/{i}_results.json`（第 299 行）。
- 文件范围：`i=1..105`（第 295 行）。

### 4.2 图加载与上下文构造
- `load_graph_from_multiple_parquet`（第 40 行）把 `title` 去重后作为节点（第 62-63 行）。
- 边关系文本来自 `relationships.parquet.description`（第 70 行）。
- `retrieve_context` 只拼接三元句子 `"source relation target."`（第 99-100 行），不包含节点描述信息。

### 4.3 已识别注意点
- `ensure_directory_exists` 在写结果前会自动建目录（第 14-20、174 行），输出更稳健。
- `load_graph_from_json` 存在但在主流程未使用（第 22-38 行）。
- 导入了 `EMBEDDING_MODEL`，但实际仍硬编码 `"all-MiniLM-L6-v2"`（第 11、294 行）。

## 5. 两脚本差异对比

| 维度 | RAKG | GraphRAG |
|---|---|---|
| 图数据源 | 单文件 JSON | Parquet（entities + relationships） |
| 边结构 | 4 元组（含关系描述） | 表格行（description + weight） |
| 节点 embedding 文本 | `name + type` | `name` |
| 上下文信息量 | 节点属性 + 关系描述 | 关系三元句为主 |
| 输出目录创建 | 无显式创建 | 有 `ensure_directory_exists` |
| 额外依赖 | 无 `pandas` | 需要 `pandas/pyarrow` |

## 6. 风险与改进建议

### 6.1 主要风险
1. `all_questions_answers` 在两个脚本中是超大硬编码常量，维护成本高且加载慢。
2. LLM 输出被直接 `int(content)` 转换，若返回非 `0/1` 会抛异常。
3. 检索 query 使用的是“标准答案文本”而不是问题文本，评估偏向“答案可匹配性”，不是问答闭环能力。
4. 文件内中文注释与 system prompt 存在乱码，可能来自编码不一致。

### 6.2 建议优先级
1. 把 `all_questions_answers` 外置到数据文件（JSON/JSONL），脚本只做读取。
2. 对 `gpt_evaluate_response` 增加输出清洗与兜底（如正则提取首个 `0|1`）。
3. 参数化 `k`、`depth`、模型名、输入输出目录和文件范围。
4. 统一两脚本的目录创建与日志策略。
5. GraphRAG 版本考虑将节点描述信息纳入 embedding 或上下文，提升判定质量。

## 7. 运行前最小检查清单
1. Ollama 服务可连通，`OLLAMA_BASE_URL` 与 `DEFAULT_MODEL` 正确。
2. `sentence-transformers` 模型可用（首次运行需下载模型）。
3. RAKG 侧确认 `data/processed/RAKG_graph_v2_1/*.json` 格式与双层 `json.loads` 假设一致。
4. GraphRAG 侧确认每个 `ragtest{i}/output` 下存在 `entities.parquet` 与 `relationships.parquet`。
