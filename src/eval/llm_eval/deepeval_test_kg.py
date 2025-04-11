import json
import os
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from prompt import extract_entiry_centric_kg_en

# 基础路径配置
input_dir = "data/processed/llmasjudge/rel_data"
output_dir = "data/processed/llmasjudge/rel_results"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历处理1-105号文件
for i in range(103, 106):
    # 动态生成输入输出路径
    input_file = os.path.join(input_dir, f"output_kg_{i}.jsonl")
    output_file = os.path.join(output_dir, f"evaluation_results_kg_{i}_intern.jsonl")
    
    # 初始化评估指标（每个文件独立计算）
    metric = FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )

    # 读取 JSONL 文件并逐行处理
    with open(input_file, "r", encoding="utf-8") as infile, \
        open(output_file, "a", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            chunk_text = data["chunk_text"]
            entity = data["entity"]
            kg = data["kg"]

            # 将 chunk_text 放入 retrieval_context
            retrieval_context = [chunk_text]

            # 遍历 attributes 和 relationships
            attributes = kg["central_entity"].get("attributes", [])
            relationships = kg["central_entity"].get("relationships", [])

            # 遍历 attributes
            for attr in attributes:
                # 将 attribute 放入 actual_output
                actual_output = json.dumps({
                    "central_entity": {
                        "name": entity["name"],
                        "type": entity["type"],
                        "attributes": [attr]
                    }
                })

                # 创建测试用例
                test_case = LLMTestCase(
                    input= extract_entiry_centric_kg_en,
                    actual_output=actual_output,
                    retrieval_context=retrieval_context
                )

                # 进行评估
                metric.measure(test_case)
                print(metric.score)
                # 创建结果记录
                result_record = {
                    "chunk_text": chunk_text,
                    "entity_name": entity["name"],
                    "entity_type": entity["type"],
                    "actual_output": actual_output,
                    "score": metric.score,
                    "reason": metric.reason
                }

                # 将结果写入输出文件
                outfile.write(json.dumps(result_record, ensure_ascii=False) + "\n")

            # 遍历 relationships
            for rel in relationships:
                # 将 relationship 放入 actual_output
                actual_output = json.dumps({
                    "central_entity": {
                        "name": entity["name"],
                        "type": entity["type"],
                        "relationships": [rel]
                    }
                })

                # 创建测试用例
                test_case = LLMTestCase(
                    input= extract_entiry_centric_kg_en,
                    actual_output=actual_output,
                    retrieval_context=retrieval_context
                )

                # 进行评估
                metric.measure(test_case)
                print(metric.score)
                # 创建结果记录
                result_record = {
                    "chunk_text": chunk_text,
                    "entity_name": entity["name"],
                    "entity_type": entity["type"],
                    "actual_output": actual_output,
                    "score": metric.score,
                    "reason": metric.reason
                }

                # 将结果写入输出文件
                outfile.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    print(f"Evaluation results have been written to {output_file}")

print("所有文件处理完成！")