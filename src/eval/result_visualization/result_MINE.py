import os
import json
import matplotlib.pyplot as plt
from collections import Counter


directory = "data/processed/RAKG_graph_v1"##

def extract_accuracy(directory):
    accuracy_list = []
    for i in range(1, 106):
        filename = f"{i}_results.json"
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # 提取最后一个元素的accuracy字段并转换为浮点数
                if isinstance(data, list) and len(data) > 0:
                    last_entry = data[-1]
                    accuracy_str = last_entry.get("accuracy", "")
                    if accuracy_str and accuracy_str.endswith('%'):
                        accuracy = float(accuracy_str.strip('%'))
                        accuracy_list.append(accuracy)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"跳过文件 {filename}: {str(e)}")
    return accuracy_list

def plot_accuracy(accuracy_list):
    if not accuracy_list:
        print("未找到有效数据")
        return
    

    average = sum(accuracy_list) / len(accuracy_list)
    

    accuracy_counts = Counter(accuracy_list)
    sorted_accuracies = sorted(accuracy_counts.keys())
    counts = [accuracy_counts[acc] for acc in sorted_accuracies]
    

    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_accuracies, counts, width=0.8)
    

    plt.axvline(average, color='red', linestyle='--', linewidth=2, label=f'Average Accuracy: {average:.2f}%')
    
    # Annotate the values at the top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height}', ha='center', va='bottom')
    
    # Chart beautification
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Accuracy Distribution Statistics (Including Average Line)', fontsize=14)
    plt.xticks(sorted_accuracies, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

accuracy_data = extract_accuracy(directory)
plot_accuracy(accuracy_data)



