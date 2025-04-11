import json
import pandas as pd


data = []
with open("data/processed/stat_results_RAKG.jsonl") as f:
    for line in f:
        data.append(json.loads(line))


df = pd.DataFrame(data)

stats = df.groupby("method").agg({
    "entitynum": ["mean", "median", "std"],
    "rel_ratio": ["mean", "median", "std"]
})
print(stats)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x="method", y="entitynum", data=df, palette="Pastel1")
plt.title("Entity Number Distribution")

plt.subplot(1, 2, 2)
sns.boxplot(x="method", y="rel_ratio", data=df, palette="Pastel1")
plt.title("Relation Ratio Distribution")

plt.savefig('src/eval/result_visualization/num_eval.png')  
plt.tight_layout()
plt.show()