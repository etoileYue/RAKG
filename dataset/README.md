# Dataset Utilities

## Multi-Doc-QA-Chinese Raw -> Retrieval JSONL

将 Hugging Face 数据集 `yuyijiong/Multi-Doc-QA-Chinese` 的 `raw` 子集转换成 dense retrieval / embedding 训练可直接消费的 JSONL。

输出格式为每行一个样本：

```json
{
  "id": "train-000001",
  "query": "问题文本",
  "positives": ["相关文档全文"],
  "negatives": ["无关文档1", "无关文档2"],
  "answer": "答案文本",
  "meta": {
    "source_dataset": "yuyijiong/Multi-Doc-QA-Chinese",
    "subset": "raw",
    "split": "train",
    "negative_count": 2
  }
}
```

### 运行方式

```bash
python dataset/convert_multidocqa_raw_to_retrieval.py \
  --data-dir raw \
  --split train \
  --output data/processed/multidocqa_chinese/raw_retrieval_train.jsonl \
  --max-negatives 20
```

如果自动字段映射不准确，可手动覆盖：

```bash
python dataset/convert_multidocqa_raw_to_retrieval.py \
  --query-field question \
  --positive-field relevant_doc \
  --negative-field irrelevant_docs \
  --answer-field answer
```

仅做字段探测和清洗统计，不写文件：

```bash
python dataset/convert_multidocqa_raw_to_retrieval.py --dry-run
```

### 自检

```bash
python -m unittest dataset/test_convert_multidocqa_raw_to_retrieval.py
```

## LongBench 子集下载到本地

将 LongBench 的指定子集下载到仓库内，便于先查看原始样本结构。

默认会下载以下两个中文子集：

- `dureader`
- `multifieldqa_zh`

输出目录默认是 `data/raw/longbench/<subset>/test.jsonl`。

### 运行方式

```bash
python dataset/download_longbench_subsets.py
```

显式指定子集：

```bash
python dataset/download_longbench_subsets.py \
  --subsets dureader multifieldqa_zh
```

修改输出目录：

```bash
python dataset/download_longbench_subsets.py \
  --output-root /path/to/longbench
```

脚本会同时打印每个子集的字段、样本数估计以及前几条样本预览，便于快速检查数据格式。

## MultiFieldQA-ZH 扩展问答生成

`dataset/generate_multifieldqa_zh_qa.py` 会读取原始 MultiFieldQA-ZH 文本，为每篇 `context` 额外生成问答对，并把原始数据集自带问答合并到同一个 QA 级 JSONL。

默认路径：

- 输入：`data/multifieldqa_zh/test.jsonl`
- 输出：`data/multifieldqa_zh/expanded_qa.jsonl`
- 汇总：`data/multifieldqa_zh/qa_generation_summary.json`

最小运行：

```bash
python dataset/generate_multifieldqa_zh_qa.py --limit 1
```

常用参数：

- `--qa-count`：每篇文本新增生成问答数，默认 `9`；原始问答会额外合并，因此默认每篇最多 10 条。
- `--start` / `--end` / `--limit`：按原始文本行号切片。
- `--force`：重写目标范围内记录。
- `--rate-limit-max-retries` / `--rate-limit-initial-wait` / `--rate-limit-max-wait`：LLM 限流重试参数。

输出 schema：

```json
{
  "qa_id": "sample-id#gen001",
  "source_sample_id": "sample-id",
  "source_sample_index": 0,
  "question": "问题",
  "answers": ["答案"],
  "qa_source": "generated",
  "evidence": "原文证据片段"
}
```

原始问答记录使用 `qa_id={source_sample_id}#original`、`qa_source=original`。生成问答要求 `answer` 和 `evidence` 都能在原文中逐字定位；校验失败的问答会被丢弃，并在 summary 的 `shortfall_count` 和逐样本统计中记录。

自检：

```bash
python -m unittest dataset/test_generate_multifieldqa_zh_qa.py
```
