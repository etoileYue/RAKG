 # 将 raw/ 转成检索训练用 query/positives/negatives JSONL

  ## Summary

  基于 yuyijiong/Multi-Doc-QA-Chinese 的 raw/ 子集，产出一份面向 dense retrieval / embedding 训练的 JSONL 文件。每
  条样本保留原始问题作为 query，保留原始相关文档作为唯一 positive，并将无关文档截断为前 15-20 个 negatives。默认推
  荐固定为 20 个 negatives，除非数据实际不足。

  ## Key Changes

  - 新增一个只做数据转换的脚本，输入为 Hugging Face 数据集 raw/，输出为 jsonl。
  - 输出结构统一为每行一个样本，字段如下：

    {
      "id": "sample-000001",
      "query": "问题文本",
      "positives": ["相关文档全文"],
      "negatives": ["无关文档1", "无关文档2"],
      "answer": "答案文本",
      "meta": {
        "source_dataset": "yuyijiong/Multi-Doc-QA-Chinese",
        "subset": "raw",
        "negative_count": 20
      }
    }
  - 字段映射原则：
      - query: 原始问题字段
      - positives: 原始唯一相关文档，包装成长度为 1 的列表
      - negatives: 原始无关文档列表，按原顺序截断到 20
      - answer: 原始答案字段，保留用于调试和后续构造 hard negative，但不作为检索训练输入必需字段
  - 清洗规则固定，避免实现时再做选择：
      - 去掉首尾空白
      - 过滤空字符串文档
      - 如果 positive 为空，直接丢弃该样本
      - 如果 negatives 少于 1，直接丢弃该样本
      - 不做分块、不做去重召回、不额外生成 hard negatives
  - 输出文件建议放在仓库的数据中间产物目录，例如 data/processed/multidocqa_chinese/raw_retrieval_train.jsonl，避免
    和现有评测数据混在一起。

  ## Implementation Details

  - 读取方式：
      - 使用 datasets.load_dataset("yuyijiong/Multi-Doc-QA-Chinese", data_dir="raw")
      - 脚本内部先打印一次字段名，兼容数据集字段可能与数据卡描述不完全一致的情况
      - 用一个显式的字段映射层，把真实字段名映射到 query / positive_doc / negative_docs / answer
  - 转换逻辑：
      - 遍历 train split；如果存在其他 split，则分别输出 train/dev/test 三个文件；如果只有 train，则只输出一个文件
      - 每个样本生成稳定 id，优先用原始样本 id，没有则用 split + 行号
      - negatives 默认截断为前 20 个；参数化支持改成 15
      - 保留 answer 和 meta，方便后续排查误标和再加工
  - 脚本接口建议固定为：
      - --data-dir raw
      - --output path/to/output.jsonl
      - --max-negatives 20
      - --split train
  - 如果后续需要喂给 sentence-transformers、自定义 bi-encoder 或 BGE 微调脚本，这个结构可以直接二次映射；不在这一步
    引入训练框架耦合字段。

  ## Test Plan

  - 结构校验：
      - 随机抽查前 3 条样本，确认存在 query / positives / negatives
      - 断言 positives 为非空列表，且长度固定为 1
      - 断言 negatives 为非空列表，且长度不超过 20
  - 数据质量校验：
      - 统计总样本数、被过滤样本数、平均 negative 数
      - 检查 positive 不应出现在 negatives 中；若出现，转换阶段去重并记录计数
      - 检查文本字段无 None、空串、纯空白
  - 可用性校验：
      - 用一条样本手工确认输出可被常见训练读取逻辑消费，例如 json.loads(line) 后直接访问 sample["query"]、
        sample["positives"]、sample["negatives"]

  ## Assumptions

  - 你要的是 document-level retrieval 训练数据，不是 chunk-level retrieval。
  - positive 保持整篇相关文档，不做切块。
  - 输出以 JSONL 为唯一正式产物，不额外保存 Hugging Face Dataset 格式。
  - negatives 使用原始无关文档并截断到 20，不引入 hard negative。
  - 保留 answer 仅用于调试和后续扩展，不作为检索模型输入的一部分。
