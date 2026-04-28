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

`dataset/generate_multifieldqa_zh_qa.py` 会读取原始 MultiFieldQA-ZH 文本，为每篇 `context` 额外生成问答对，并把原始数据集自带问答合并到同一个按样本分组的 JSONL。

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
- `--force`：重新生成目标范围内记录。
- `--rate-limit-max-retries` / `--rate-limit-initial-wait` / `--rate-limit-max-wait`：LLM 限流重试参数。

输出 schema：

```json
{
  "source_sample_index": 0,
  "length": 9593,
  "dataset": "multifieldqa_zh",
  "language": "zh",
  "all_classes": null,
  "qa_pairs": [
    {
      "qa_id": "0",
      "question": "原始问题",
      "answers": ["原始答案"],
      "qa_source": "original",
      "evidence": ""
    },
    {
      "qa_id": "1",
      "question": "生成问题",
      "answers": ["生成答案"],
      "qa_source": "generated",
      "evidence": "模型给出的证据"
    }
  ]
}
```

每行对应一条原始样本，不再输出 `source_sample_id`。`qa_id` 仅在样本内唯一，`"0"` 固定为原始问答，生成问答从 `"1"` 递增。生成问答只会因空 `question`、空答案或重复问答被丢弃；`answer` 或 `evidence` 未在原文中逐字出现时仍会保留，并在 summary 的 `unmatched_answer_count`、`unmatched_evidence_count`、`shortfall_count` 和逐样本统计中记录。

自检：

```bash
python -m unittest dataset/test_generate_multifieldqa_zh_qa.py
```
