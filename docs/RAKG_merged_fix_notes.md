# RAKG P0/P1 修复说明

## 修复范围
- `src/construct/RAKG.py`
- `src/kgAgent.py`

本次仅修复此前分析文档中列出的 P0 与 P1 问题，不包含 P2 可维护性优化项。

## P0 问题修复

### P0-1 输出 JSON 二次序列化
- 问题: 最终写盘前先 `json.dumps(...)` 生成字符串，再 `json.dump(...)`，导致输出文件存的是字符串而不是 JSON 对象。
- 修改:
  - 移除旧的字符串化转换流程。
  - 新增 `_validate_json_serializable(data)`，只做可序列化校验，不改变数据内容。
  - 最终直接 `json.dump(dict, ...)` 写出对象结构。
- 代码位置:
  - `src/construct/RAKG.py:17-20`
  - `src/construct/RAKG.py:98`
  - `src/construct/RAKG.py:103-104`

### P0-2 单引号替换破坏语义
- 问题: 旧逻辑会将所有字符串中的 `'` 替换为 `"`，可能破坏原始文本。
- 修改:
  - 删除基于全量字符串替换的 `convert_to_valid_json` 思路。
  - 改为仅验证 JSON 可序列化，不再修改字符串内容。
- 代码位置:
  - `src/construct/RAKG.py:17-20`

## P1 问题修复

### P1-1 NumPy 隐式标量转换风险
- 问题: `cosine_similarity(...)` 返回二维数组，直接赋值给标量位点依赖弃用行为。
- 修改:
  - 显式取标量: `float(cosine_similarity(...)[0][0])`。
- 代码位置:
  - `src/kgAgent.py:116-119`

### P1-2 异常只记录日志、缺少结构化失败结果
- 问题: 主循环异常被捕获后仅日志输出，缺乏汇总与可追踪失败清单。
- 修改:
  - 在处理循环中累计 `failed_topics`（包含 index/topic/error）。
  - 处理结束后写出 `process_summary.json`（总数、成功数、失败数、失败详情）。
  - 函数返回 summary，便于上层流程接入。
- 代码位置:
  - `src/construct/RAKG.py:48-49`
  - `src/construct/RAKG.py:109-115`
  - `src/construct/RAKG.py:118-128`

### P1-3 路径和参数硬编码
- 问题: NER/KG 中间产物路径固定在函数体内，迁移环境不方便。
- 修改:
  - `process_all_topics(...)` 增加可配置参数:
    - `done_offset=1`
    - `skip_ner_list=None`
    - `ner_output_dir=None`
    - `rel_output_dir=None`
  - 在函数内进行参数归一化，默认值保留原行为但支持外部覆盖。
- 代码位置:
  - `src/construct/RAKG.py:23-30`
  - `src/construct/RAKG.py:35-38`
  - `src/construct/RAKG.py:69-70`
  - `src/construct/RAKG.py:135-142`

### P1-4 JSONL 输出目录可能不存在
- 问题: 以追加模式写 JSONL 时，父目录不存在会直接失败。
- 修改:
  - `NER_Agent` 新增 `_ensure_parent_dir(output_file)`。
  - 在 NER 输出和 KG 输出前统一确保父目录存在。
  - 编排层也在任务开始时确保 `output_dir/ner_output_dir/rel_output_dir` 存在。
- 代码位置:
  - `src/kgAgent.py:34-37`
  - `src/kgAgent.py:59`
  - `src/kgAgent.py:296-297`
  - `src/construct/RAKG.py:40-43`

### P1-5 Skip NER 缓存文件缺失时行为不明确
- 问题: `skip_ner_list` 命中但缓存文件不存在时，会在下游阶段抛出不直观错误。
- 修改:
  - 增加显式文件存在性检查，不存在则抛出 `FileNotFoundError` 并进入失败汇总。
- 代码位置:
  - `src/construct/RAKG.py:71-77`

## 兼容性说明
- `process_all_topics(...)` 仍可按原方式调用；新增参数均提供默认值。
- 默认目录值保持与历史路径一致，避免影响现有离线数据组织方式。

## 未纳入本次修复
- P2 类问题（如非确定性顺序、冗余条件、未使用导入）未在本次变更中处理。


---

# RAKG P2 修复说明

## 修复范围
- `src/kgAgent.py`

本次针对此前分析中的 P2 问题进行修复，重点是可复现性与可维护性。

## P2 问题与修复映射

### P2-1 非确定性输出（`set()` 去重导致顺序不稳定）
- 问题:
  - 消歧后 `description/chunkid` 使用 `set` 聚合再拼接，字段顺序不稳定。
  - `get_target_kg_single` 对句子去重使用 `set`，上下文拼接顺序不稳定。
- 修改:
  - 新增 `_dedupe_preserve_order(items)`，按首次出现顺序去重。
  - `entity_Disambiguation` 中将 `description/chunkid` 改为列表采集后稳定去重再拼接。
  - `get_target_kg_single` 中句子去重改为稳定去重，保证 `chunk_text` 可复现。
- 代码位置:
  - `src/kgAgent.py:39-46`
  - `src/kgAgent.py:207-215`
  - `src/kgAgent.py:291`

### P2-2 冗余条件分支（可维护性）
- 问题:
  - `get_target_kg_all` 中 `for entity_id in entity_dic` 后再次判断 `if entity_id in entity_dic`，逻辑恒真且冗余。
- 修改:
  - 删除恒真分支，改为直接调用 `get_target_kg_single`，保持行为不变、代码更清晰。
- 代码位置:
  - `src/kgAgent.py:320-330`

### P2-3 未使用导入（可维护性）
- 问题:
  - `kgAgent.py` 中 `import re` 未使用。
- 修改:
  - 移除未使用导入。
- 代码位置:
  - `src/kgAgent.py:6`（已删除）

## 行为影响说明
- 本次不改变核心业务语义（实体抽取、关系抽取策略不变）。
- 主要影响是输出更稳定（同输入下文本拼接顺序更一致）以及代码可维护性提升。


---

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


---

# RAKG 代码审查修复说明

## 范围
- `src/construct/RAKG.py`
- `src/kgAgent.py`
- `src/utils.py`

## 修复目标
本次修复针对代码审查中发现的稳定性问题，重点避免以下场景导致流程中断：
- 输出目录参数为空导致的启动失败
- Windows 默认编码导致的 JSONL 写入异常
- LLM 输出结构不稳定导致的 KeyError/TypeError
- 句子映射缺失导致的 KeyError
- 缓存 JSONL 脏数据导致的读取失败
- 空向量/维度异常导致的相似度计算失败

## 具体修改

### 1) `src/construct/RAKG.py`
- 为 `output_dir` 增加非空校验。
- 当 `ner_output_dir` 或 `rel_output_dir` 未传入时，自动回退到：
  - `<output_dir>/ner_data`
  - `<output_dir>/rel_data`
- 处理 topic 前增加字段校验，确保 `topic` 与 `content` 非空。

影响：避免默认参数下 `os.makedirs(None, ...)` 崩溃，同时对输入数据缺字段场景给出明确错误。

### 2) `src/kgAgent.py`
- 新增 `_append_jsonl()` 统一 JSONL 输出：
  - 固定 `encoding='utf-8'`
  - `ensure_ascii=False`
- 将 `extract_from_text_single()` 和 `get_target_kg_single()` 的写文件逻辑切换到 `_append_jsonl()`。
- 在 `extract_from_text_multiply()` 中，`sent_to_id` 改为 `get()` + 缺失告警 + 跳过。
- 在 `get_retriever_context()` 中增加空输入短路；相似度计算异常时记录错误并返回空结果。
- 在 `convert_knowledge_graph()` 中增加防御性解析：
  - `central_entity` 非 dict 时跳过
  - `name` 缺失时跳过
  - relation 不是 dict 或缺关键字段时跳过并告警

影响：提升模型输出不稳定时的容错能力，减少运行时硬崩溃。

### 3) `src/utils.py`
- `get_ner_result_from_file()` 增加 JSONL 容错：
  - 非法 JSON 行跳过
  - 缺 `text`、`entities` 非 dict 跳过
  - `sent_to_id` 映射缺失时跳过
- 保留原逻辑中的实体重编号与 `chunkid` 注入。

影响：缓存文件出现坏行或半结构化数据时不再导致全量失败。

## 验证
已执行：
- `python -m py_compile src/construct/RAKG.py src/kgAgent.py src/utils.py`

结果：通过（语法层面无错误）。

## 兼容性说明
- 本次不改变核心算法流程（实体抽取、消歧、关系抽取逻辑保持不变）。
- 主要新增输入校验与异常兜底，行为变化体现在“遇到坏数据时跳过并记录日志”，而非直接中断。


