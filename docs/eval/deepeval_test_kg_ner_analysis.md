# deepeval_test_kg.py 与 deepeval_test_ner.py 说明文档

## 1. 文档范围
- [deepeval_test_kg.py](/e:/code/Source/Repos/RAKG/src/eval/llm_eval/deepeval_test_kg.py)
- [deepeval_test_ner.py](/e:/code/Source/Repos/RAKG/src/eval/llm_eval/deepeval_test_ner.py)

两者都使用 `deepeval.metrics.FaithfulnessMetric` 对抽取结果做忠实度评测，输入为 JSONL，输出为 JSONL。

## 2. 脚本一：`deepeval_test_kg.py`

### 2.1 目标
对实体中心 KG 抽取结果中的 `attributes` 与 `relationships` 逐项评分。

### 2.2 关键实现
- 输入目录：`data/processed/llmasjudge/rel_data`（第 9 行）
- 输出目录：`data/processed/llmasjudge/rel_results`（第 10 行）
- 处理文件编号：`103~105`（第 16 行）
- 评分阈值：`threshold=0.7`（第 23 行）
- 使用提示模板：`extract_entiry_centric_kg_en`（第 6, 56, 90 行）
- 输出模式：`open(output_file, "a")` 追加写入（第 29 行）

### 2.3 处理流程
1. 创建输出目录（第 13 行）。
2. 逐文件读取 `output_kg_{i}.jsonl`（第 18, 28 行）。
3. 每条记录提取 `chunk_text`、`entity`、`kg`（第 32-34 行）。
4. 将 `chunk_text` 作为 `retrieval_context`（第 37 行）。
5. 对每个 attribute/relationship 构造 `actual_output`（第 46-52, 80-86 行）。
6. 构造 `LLMTestCase` 并调用 `metric.measure()` 评分（第 55-63, 89-97 行）。
7. 写出 `score` 与 `reason` 到结果文件（第 65-75, 99-109 行）。

### 2.4 输入数据要求（每行 JSON）
```json
{
  "chunk_text": "原文片段",
  "entity": { "name": "实体名", "type": "实体类型" },
  "kg": {
    "central_entity": {
      "attributes": [],
      "relationships": []
    }
  }
}
```

### 2.5 输出字段
```json
{
  "chunk_text": "...",
  "entity_name": "...",
  "entity_type": "...",
  "actual_output": "{...字符串化JSON...}",
  "score": 0.0,
  "reason": "..."
}
```

## 3. 脚本二：`deepeval_test_ner.py`

### 3.1 目标
对 NER 结果中的每个实体（含 description）进行忠实度评分。

### 3.2 关键实现
- 输入目录：`data/processed/llmasjudge/ner_data`（第 6 行）
- 输出目录：`data/processed/llmasjudge/ner_results`（第 11 行）
- 处理文件编号：`103~105`（第 9 行）
- 评分阈值：`threshold=0.9`（第 27 行）
- 使用提示模板：`text2entity_en`（第 4, 33 行）
- 输出模式：`open(output_file, "w")` 覆盖写入（第 14 行）
- 单行异常保护：`try/except`（第 17, 60 行）

### 3.3 处理流程
1. 逐文件读取 `output_text_ner_{i}.jsonl`（第 10, 13 行）。
2. 每条记录提取 `text` 与 `entities`（第 19-20 行）。
3. 对每个实体构造 `actual_output`，将 `description` 放入 `attributes`（第 34-43 行）。
4. 构造 `LLMTestCase` 并评分（第 32-49 行）。
5. 写出 `text/entity_data/score/reason`（第 51-58 行）。
6. 某行异常时打印错误并继续（第 60-61 行）。

### 3.4 输入数据要求（每行 JSON）
```json
{
  "text": "原文",
  "entities": {
    "实体名A": {
      "type": "实体类型",
      "description": "实体描述"
    }
  }
}
```

### 3.5 输出字段
```json
{
  "text": "...",
  "entity_name": "...",
  "entity_data": {
    "type": "...",
    "description": "..."
  },
  "score": 0.0,
  "reason": "..."
}
```

## 4. 两个脚本的关键差异

| 维度 | KG 脚本 | NER 脚本 |
|---|---|---|
| 阈值 | 0.7 | 0.9 |
| 评分对象粒度 | attribute/relationship 单项 | 实体单项 |
| metric 实例创建位置 | 每个文件创建一次 | 每个实体创建一次 |
| 输出写入模式 | 追加 (`a`) | 覆盖 (`w`) |
| 异常处理 | 无逐行保护 | 有逐行 `try/except` |
| 结果字段 | 含 `actual_output` 字符串 | 含 `entity_data` 对象 |

## 5. 已识别风险与建议

### 5.1 共性风险
- 文件编号硬编码为 `103~105`，与注释“1-105”不一致（KG: 第 16 行，NER: 第 9 行）。
- 注释与末尾提示文本存在乱码，建议统一 UTF-8 编码后修复。
- 都依赖 `prompt.py` 内模板变量，命名变更会直接影响运行。

### 5.2 KG 特有风险
- `from deepeval import evaluate` 未使用（第 3 行），可删除。
- 使用追加模式可能导致重复运行后结果累积（第 29 行）。
- 无异常处理，坏行会中断整文件处理。

### 5.3 NER 特有风险
- 输出目录未显式 `makedirs`，若目录不存在会直接报错。
- 每实体新建 metric，逻辑清晰但有轻微初始化开销。

## 6. 可落地改进（优先级）
1. 将输入目录、输出目录、文件范围、阈值改为命令行参数。
2. 给 KG 脚本补充逐行异常处理，并记录坏行号。
3. 统一输出结构，增加 `run_id`/时间戳便于多轮对比。
4. 统一注释编码与日志文本，避免乱码。
5. 为两脚本补充最小输入样例与 smoke test。
