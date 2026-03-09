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
