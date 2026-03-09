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
