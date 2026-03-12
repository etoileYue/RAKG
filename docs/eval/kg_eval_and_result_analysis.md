# kg_eval.py 与 kg_eval_result.py 说明文档

## 1. 文档范围
- [kg_eval.py](/e:/code/Source/Repos/RAKG/src/eval/ideal_kg_eval/kg_eval.py)
- [kg_eval_result.py](/e:/code/Source/Repos/RAKG/src/eval/ideal_kg_eval/kg_eval_result.py)

## 2. 系统性说明：这段代码到底在做什么
这两个文件共同组成一个“两阶段评测流水线”：
1. `kg_eval.py` 负责逐数据集对比多个候选知识图（`kggen`/`graphrag`/`rakg`）与标准图（`idealKG`），输出逐条评测结果到 `kg_evaluation_results.jsonl`。
2. `kg_eval_result.py` 读取这个 JSONL，把三种方法的两项指标做总体统计（均值与方差），用于横向对比整体表现稳定性。

换句话说，第一段代码解决“每个样本谁更接近标准图”，第二段代码解决“在 105 个样本上总体谁更好、波动多大”。

## 3. kg_eval.py 详细说明

### 3.1 评测目标
对于每个 `dataid`（1~105），将候选图和标准图做结构语义对齐，输出两项核心指标：
- `Entity Coverage Rate`：标准图实体被候选图成功匹配的覆盖比例。
- `Relation Similarity`：已匹配实体对应关系集合的平均语义相似度。

### 3.2 核心流程（主程序）
主流程位于第 170 行以后：
1. 初始化 Embedding 与 LLM（第 172-173 行）。
2. 循环 `i=1..105`（第 179 行）。
3. 读取四类图数据（第 181-204 行）：
   - 标准图：`data/processed/idealKG/kg_{i}.json`
   - 候选图：`kggen`、`graphrag`、`rakg`
4. 对每个候选图构建 `KGEvaluator(...).evaluate()`（第 208-227 行）。
5. 写一行 JSON 到 `kg_evaluation_results.jsonl`（第 176、247 行）。

### 3.3 KGEvaluator 在做什么
`KGEvaluator` 是评测核心（第 36 行起）：

1. 预处理图结构（第 47-59 行）
- 建立 `name -> entity` 字典。
- 建立“实体名 -> 相关关系列表”的索引（关系同时挂到 head/tail）。

2. 用 embedding 召回候选实体对（第 65-91 行）
- 将实体文本组织为 `name + type + description`（第 68-75 行）。
- 计算标准图与候选图实体两两余弦相似度，得到候选匹配对。

3. 用 LLM 做实体同一性判断（第 93-104 行）
- Prompt 要求输出 `{"result": true/false}`。
- 仅保留 LLM 判定为同一实体的对齐对。

4. 用 LLM 做关系集合语义相似度评估（第 106-126 行）
- 对每个匹配实体，比较两边关系集合。
- Prompt 输出 `{"similarity": 0~1}`。

5. 计算并返回指标（第 148-167 行）
- `entity_coverage_rate = matched_entities_count / total_std_entities`
- `relation_similarity = 匹配实体的关系相似度均值`

### 3.4 输入与输出
- 输入：四套 KG 数据文件（标准 + 三种方法）。
- 输出：`kg_evaluation_results.jsonl`，每行结构示例：
```json
{
  "dataid": 1,
  "kggen": { "Entity Coverage Rate": 0.8123, "Relation Similarity": 0.7011 },
  "graphrag": { "Entity Coverage Rate": 0.7560, "Relation Similarity": 0.6888 },
  "rakg": { "Entity Coverage Rate": 0.8340, "Relation Similarity": 0.7199 }
}
```

### 3.5 已识别实现注意点
1. `_get_embedding` 使用 `embed_documents(text)`（第 63 行），传入是字符串；按 LangChain 习惯通常应传列表或改用 `embed_query`，否则可能依赖底层兼容行为。
2. 匹配阶段变量命名有误导：`matched_eval_entities` 实际存的是 `s_name`（标准实体，见第 135、138、146 行），逻辑仍是“每个标准实体最多匹配一次”。
3. `rakg` 输入做了双重 `json.loads`（第 196-197 行），说明该数据是“JSON 字符串包裹 JSON”；如果格式变更会直接失败。
4. 默认 `threshold=0.0`（第 65 行），会放入全部候选对，LLM 调用量可能较大。
5. 异常时直接回退 `False` 或 `0.0`（第 101-104、123-126 行），会压低分数但不报详细错误上下文。

## 4. kg_eval_result.py 详细说明

### 4.1 作用
对 `kg_eval.py` 产出的 JSONL 结果做聚合统计，输出三种方法在两项指标上的：
- 均值（mean）
- 方差（variance）

### 4.2 流程
1. 初始化 6 个列表存放指标（第 5-10 行）。
2. 逐行读取 `kg_evaluation_results.jsonl` 并抽取三种方法指标（第 13-30 行）。
3. 对每个方法/指标计算 `mean` 和 `variance`（第 33-46 行）。
4. 打印统计结果（第 49-54 行）。

### 4.3 输出含义
- 均值：整体平均效果。
- 方差：跨样本波动，越小通常代表稳定性越好。

## 5. 两文件关系与使用顺序
1. 先运行 [kg_eval.py](/e:/code/Source/Repos/RAKG/src/eval/ideal_kg_eval/kg_eval.py) 生成 `kg_evaluation_results.jsonl`。
2. 再运行 [kg_eval_result.py](/e:/code/Source/Repos/RAKG/src/eval/ideal_kg_eval/kg_eval_result.py) 做总体统计。

如果第二个文件先执行，会因为缺少结果文件而失败。

## 6. 建议改进
1. 将路径、数据范围（1..105）、阈值、模型参数改为命令行参数。
2. 把 `kg_eval_result.py` 输出改为结构化文件（JSON/CSV），便于后续画图与报告复用。
3. 在 `kg_eval.py` 中记录每个实体匹配过程（候选相似度、LLM判定）以提升可审计性。
4. 给 LLM 输出增加格式校验与重试策略，减少解析失败导致的 0 分。
