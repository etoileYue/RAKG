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
