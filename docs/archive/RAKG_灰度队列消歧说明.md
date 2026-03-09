# RAKG 灰度队列消歧说明

## 背景

原始流程中，`similarity_llm_single` 的返回如果解析失败或格式异常，会回落为 `False`，导致潜在同实体直接被拒绝，影响召回。

本次策略：

- 解析失败样本与边界样本进入灰度队列。
- 需要二次判定时，不引入新提示词，直接再次调用原有 `judge_sim_entity_en` 逻辑。

## 主要改动

### 1. 解析结果增加灰度标记（`src/utils.py`）

- `parse_similarity_response(...)` 现在除了 `result`，还返回：
  - `parse_status`
  - `needs_review`
  - `reason`
  - `raw_excerpt`
- 解析失败不再只是“直接 false”，而是标记为 `needs_review=true`，交给灰度队列。

### 2. 候选携带相似度分值（`src/kgAgent.py`）

- `similarity_candidates(...)` 返回 `(entity_a, entity_b, similarity_score)`。
- 用于识别边界样本：`score <= threshold + gray_margin`。

### 3. 灰度队列与二次判定（`src/kgAgent.py`）

- `similarity_result(...)` 流程：
  1. 第一轮调用 `similarity_llm_single(...)`。
  2. `needs_review=true` 或边界样本进入灰度队列。
  3. 灰度队列二次判定时，直接再次调用 `similarity_llm_single(...)`（即原提示词）。
  4. 二次结果为 `true` 才加入最终可合并对。

说明：已删除自定义 `similarity_llm_second_pass(...)` 提示词方法，不再维护单独 second-pass prompt。

### 4. 可观测性

- `self.last_disambiguation_gray_queue` 保存最近一次灰度队列明细。
- 日志新增汇总：`candidates / gray_queue / resolved_by_second_pass / merged_pairs`。

## 影响

- 优点：减少“解析失败即误拒绝”，提升召回稳定性。
- 成本：灰度样本会额外触发一次同提示词判定，增加少量时延。
