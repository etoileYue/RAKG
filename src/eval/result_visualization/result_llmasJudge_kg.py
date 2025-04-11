import json
import os

# 文件路径前缀和后缀
base_path = "data/processed/llmasjudge/rel_results/evaluation_results_kg_"
file_extension = "_intern.jsonl"

# 遍历文件1到5
for i in range(1, 106):
    file_path = f"{base_path}{i}{file_extension}"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        continue
    
    # 初始化统计变量
    total_count = 0
    pass_count = 0
    
    # 打开并读取文件
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                # 解析JSON数据
                data = json.loads(line.strip())
                
                # 更新总数
                total_count += 1
                
                # 检查score是否为1
                if data.get("score", 0) == 1.0:
                    pass_count += 1
            except json.JSONDecodeError as e:
                continue
    
    # 计算通过率
    if total_count > 0:
        pass_rate = (pass_count / total_count) * 100
    else:
        pass_rate = 0.0
    
