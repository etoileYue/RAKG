import json
import os
import matplotlib.pyplot as plt


base_path = "data/processed/llmasjudge/ner_results/evaluation_results_ner_"
file_extension = ".jsonl"


pass_rates = []


for i in range(1, 106):
    file_path = f"{base_path}{i}{file_extension}"
    

    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，跳过")
        continue
    

    total_count = 0
    pass_count = 0
    

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:   
            try:

                data = json.loads(line.strip())
                
                total_count += 1

                if data.get("score", 0) == 1.0:
                    pass_count += 1
            except json.JSONDecodeError as e:
                print(f"解析JSON时出错: {e}, 跳过该行")
                continue
    

    if total_count > 0:
        pass_rate = (pass_count / total_count) * 100
    else:
        pass_rate = 0.0

    

    pass_rates.append(pass_rate)

# 绘制箱线图
plt.figure(figsize=(10, 6))
plt.boxplot(pass_rates, vert=True, patch_artist=True)
plt.title("Pass Rate Distribution Across Files")
plt.ylabel("Pass Rate (%)")
plt.xticks([1], ["Files"])
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()