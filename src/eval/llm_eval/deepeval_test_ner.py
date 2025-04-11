import json
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from prompt import text2entity_en

base_path = "data/processed/llmasjudge/ner_data"

# 处理1到105个文件
for i in range(103, 106):
    input_file = f"{base_path}/output_text_ner_{i}.jsonl"
    output_file = f"data/processed/llmasjudge/ner_results/evaluation_results_ner_{i}.jsonl"

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            try:
                data = json.loads(line)
                text = data["text"]
                entities = data["entities"]
                retrieval_context = [text]

                # 处理每个实体
                for entity_name, entity_data in entities.items():
                    # 每个实体使用新的metric实例
                    metric = FaithfulnessMetric(
                        threshold=0.9,
                        include_reason=True
                    )
                    
                    # 构建测试用例
                    test_case = LLMTestCase(
                        input=text2entity_en,
                        actual_output=json.dumps({
                            "central_entity": {
                                "name": entity_name,
                                "type": entity_data["type"],
                                "attributes": [{
                                    "key": "description",
                                    "value": entity_data["description"]
                                }]
                            }
                        }),
                        retrieval_context=retrieval_context
                    )

                    # 执行评估
                    metric.measure(test_case)
                    print(metric.score)
                    # 写入结果
                    result_record = {
                        "text": text,
                        "entity_name": entity_name,
                        "entity_data": entity_data,
                        "score": metric.score,
                        "reason": metric.reason
                    }
                    outfile.write(json.dumps(result_record, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error in {input_file}: {str(e)}")

print("批量评估完成，结果已保存至对应文件。")