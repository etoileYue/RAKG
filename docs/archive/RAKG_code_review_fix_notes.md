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
