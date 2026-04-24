# MultiFieldQA-ZH 分阶段评测

这个目录提供 LongBench `multifieldqa_zh` 的分阶段评测入口，默认面向：

`dataset/longbench/multifieldqa_zh/test.jsonl`

主脚本：

`src/eval/multifieldqa_zh/evaluate_multifieldqa_zh.py`

## 设计目标

- 把 `build / answer / score` 三个阶段拆开执行，避免 200 条长文本一次性重跑。
- 默认按 `sample_id` 增量续跑，已完成样本自动跳过。
- 支持 `--start` / `--end` / `--limit` 只跑局部样本。
- 单条样本失败不会中断整批任务，错误会落盘到结果文件。

## 三个子命令

### 1. build

把每条样本的 `context` 当作单篇长文，逐样本构图。

示例：

```bash
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh build
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh build --limit 1
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh build --start 10 --end 20
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh build --force --limit 5
```

默认输出：

- `data/eval/multifieldqa_zh/graphs/{sample_index}.json`
- `data/eval/multifieldqa_zh/build_manifest.jsonl`
- `data/eval/multifieldqa_zh/build_summary.json`
- `data/eval/multifieldqa_zh/build_cache/`
- `data/eval/multifieldqa_zh/build_cache/checkpoint_state.json`

`build_manifest.jsonl` 以 `sample_id` 为主键记录每条样本的构图状态、图文件路径和错误信息。
`checkpoint_state.json` 记录每条样本的 NER / SIM / REL 阶段状态和缓存文件路径。构图中断后再次运行 `build` 时，会在输入数据一致且未使用 `--force` 的情况下复用已完成阶段的缓存。
若 checkpoint 机制加入前已经留下 `build_cache/ner_data`、`build_cache/sim_data` 或 `build_cache/rel_data` 阶段缓存，`build` 会在首次生成 checkpoint 时自动开启兼容续跑；若旧图文件已存在但 manifest 缺失，也会自动回填 manifest 并跳过该样本。

### 2. answer

基于已构好的图，对每条 `input` 执行 RAKG 问答。

当前固定 QA 参数：

- `max_hop=2`
- `seed_top_k=5`
- `max_context_items=30`

示例：

```bash
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh answer
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh answer --limit 1
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh answer --force --start 0 --end 10
```

默认输出：

- `data/eval/multifieldqa_zh/predictions.jsonl`
- `data/eval/multifieldqa_zh/answer_summary.json`

`predictions.jsonl` 每行至少包含：

- `sample_id`
- `question`
- `answers`
- `pred_answer`
- `formatted_answer`
- `retrieval.context_text`
- `retrieval.evidence_items`
- `graph_paths`
- `graph_path`

### 3. score

基于 `predictions.jsonl` 做自动评分。

包含三类分数：

- 官方 LongBench `qa_f1_zh_score`
- `answer_judge`：LLM 判断预测答案与任一参考答案是否语义等价
- `retrieval_judge`：LLM 判断检索上下文是否覆盖任一参考答案的关键信息

示例：

```bash
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh score
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh score --skip-llm-judge
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh score --judge-model-mode answer
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh score --judge-model-mode retrieval
```

默认输出：

- `data/eval/multifieldqa_zh/scored_results.jsonl`
- `data/eval/multifieldqa_zh/score_summary.json`

`score_summary.json` 至少包含：

- `count`
- `avg_f1`
- `answer_judge_accuracy`
- `retrieval_judge_accuracy`
- `error_count`

## 公共参数

三个子命令都支持：

- `--dataset-path`
  默认 `dataset/longbench/multifieldqa_zh/test.jsonl`
- `--output-root`
  默认 `data/eval/multifieldqa_zh`
- `--start`
  数据集起始下标，包含
- `--end`
  数据集结束下标，不包含
- `--limit`
  切片后最多处理多少条
- `--force`
  关闭跳过逻辑，强制重跑目标范围

## 断点续跑语义

- `build`：若 `build_manifest.jsonl` 中该 `sample_id` 已是成功状态，且图文件仍存在，则默认跳过；若图文件已存在但 manifest 缺失，会先回填成功记录；若某条样本上次只完成了部分构图阶段，则根据 `build_cache/checkpoint_state.json` 和对应缓存文件从已完成阶段后继续。对没有 checkpoint 的旧阶段缓存，首次运行也会检测并启用续跑。
- `answer`：若 `predictions.jsonl` 中该 `sample_id` 已是成功状态，则默认跳过。
- `score`：若 `scored_results.jsonl` 中该 `sample_id` 已经具备当前请求需要的评分字段，则默认跳过。
- 使用 `--force` 可以覆盖以上跳过逻辑。

## 依赖说明

- `build` 和 `answer` 依赖仓库现有的 RAKG 构图与 QA 运行环境。
- `score` 的官方中文 F1 依赖 `jieba` 分词。
- 如果启用 LLM judge，`score` 还依赖当前仓库配置的 LLM 提供方。

## 推荐最小流程

先用 1 条样本做冒烟：

```bash
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh build --limit 1
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh answer --limit 1
python -m src.eval.multifieldqa_zh.evaluate_multifieldqa_zh score --limit 1 --skip-llm-judge
```

确认输出结构无误后，再跑全量。
