[文件：evaluate_MINE_RAKG.py](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py)

**整体流程**
1. `main()` 逐个读取 `1..105` 的图文件，初始化同一个 `SentenceTransformer("all-MiniLM-L6-v2")`，每个图都做嵌入、检索、LLM评估并写结果文件（[evaluate_MINE_RAKG.py:281](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:281), [evaluate_MINE_RAKG.py:284](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:284)）。
2. 每条样本只用 `qa["answer"]`（不是 question）作为检索查询和判定目标（[evaluate_MINE_RAKG.py:132](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:132), [evaluate_MINE_RAKG.py:133](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:133)）。

**每个函数流程**

1. `load_graph_from_json(file_path)`（[13](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:13)）
1. 读文件后执行两次 `json.loads`（[16](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:16), [18](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:18)），说明输入很可能是“JSON字符串包着JSON对象”。
2. 建 `nx.DiGraph()`。
3. `entities` -> 节点属性：`type`/`description`/`attributes`（[25-30](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:25)）。
4. `relations` -> 有向边属性：`relation` + `rel_description`（[33-36](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:33)）。

2. `generate_embeddings(graph, model)`（[40](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:40)）
1. 对每个节点拼接文本 `full_text = "{node} {type}"` 并编码（[46-51](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:46)）。
2. 对每种关系标签 `rel` 单独编码（[54-56](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:54)）。
3. 返回 `node_embeddings, relation_embeddings`（[59](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:59)）。

3. `retrieve_relevant_nodes(query, node_embeddings, model, k=8)`（[61](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:61)）
1. 编码 `query`。
2. 对所有节点向量算余弦相似度（[63](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:63)）。
3. 排序后取前 `k=8`（[65-66](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:65)）。

4. `retrieve_context(node, graph, depth=2)`（[69](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:69)）
1. DFS 递归，最多深度2。
2. 加入节点文本：`name/type/description`（[75-80](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:75)）。
3. 遍历**出边邻居**（`graph.neighbors`），加入关系文本：`source --[relation]-> target --description:rel_description--`（[88-91](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:88)）。
4. 使用 `set` 去重后返回 list（[70](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:70), [94](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:94)）。

5. `gpt_evaluate_response(correct_answer, context)`（[97](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:97)）
1. prompt 结构：`Context` + `Correct Answer` + 二分类任务说明（[98-108](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:98)）。
2. 调 `ollama client.chat`，`temperature=0.0`，`num_predict=2`，要求只回 `1/0`（[109-119](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:109)）。
3. `int(content)` 作为判定结果（[121-123](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:121)）。

6. `evaluate_accuracy(...)`（[126](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:126)）
1. 对每个答案：
   1. 用 `correct_answer` 做检索 query（[132-133](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:132)）。
   2. 取 top-k 节点后拼每个节点的 `retrieve_context`（[135-136](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:135)）。
   3. `context_text = " ".join(context)`（[137](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:137)）。
   4. 调 LLM 判定并累计正确数（[139-145](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:139)）。
2. 计算 accuracy，写入 `output_file`（[162-168](/e:/code/Source/Repos/RAKG/src/eval/MINE_eval/evaluate_MINE_RAKG.py:162)）。

**重点1：向量化时到底包含哪些信息**
1. 节点向量化输入只包含：
   1. `node` 名称
   2. `node_data['type']`
2. 明确**不包含**：
   1. `description`
   2. `attributes`
   3. 邻接关系和 `rel_description`
3. 关系向量化输入只包含关系标签 `relation` 字符串本身（不含 source/target，不含 rel_description）。
4. 检索查询向量是 `correct_answer` 文本本身，不是问题文本。

**重点2：给 LLM 判定的 context 如何构造**
1. 先用 `correct_answer` 检索出 top 8 节点。
2. 对每个节点做深度2的出边扩展，收集：
   1. 节点块：`name/type/description`
   2. 关系块：`source --[relation]-> target --description:rel_description--`
3. 单个起点内部用 `set` 去重，但不同起点之间仍可能重复。
4. 最终把所有块直接空格拼接成一段长字符串 `context_text`，作为 prompt 里的 `Context` 字段给 LLM。  
5. LLM只做“context是否包含correct_answer信息”的二元判断，返回 `1/0`。  

**补充观察（影响解读）**
1. `relation_embeddings` 已计算但在当前流程未被使用。
2. `retrieve_context` 只沿有向出边走，可能漏掉入边信息。
3. `set -> list` 会导致 context 顺序不稳定。