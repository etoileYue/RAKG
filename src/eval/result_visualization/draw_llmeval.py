import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = []
with open('src/eval/result_visualization/llmasjudge_eval.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))


df = pd.DataFrame(data)

df = df.rename(columns={
    'entitypass': 'Entity Fidelity',
    'relationpass': 'Relationship Fidelity'
})


print(df.describe())




plt.figure(figsize=(13, 9))
sns.set_style("whitegrid")
sns.set_palette("pastel")
df_melt = df.melt(value_vars=['Entity Fidelity', 'Relationship Fidelity'],
                  var_name='Evaluation Metric',
                  value_name='score')

ax = sns.boxplot(
    x='Evaluation Metric',
    y='score',
    hue='Evaluation Metric',
    data=df_melt[df_melt['score'] > 0],
    width=0.5,
    fliersize=8,
    palette=["#66CCFF", "#FF6666"],
    legend=False
)

means = df_melt[df_melt['score'] > 0].groupby('Evaluation Metric')['score'].mean()
for i, mean in enumerate(means):
    ax.text(i, mean + 1, f'{mean:.2f}', ha='center', color='black', weight='semibold',fontsize=20)

sns.despine()
plt.title('Distribution of Entity and Relation Passing Scores in RAKG Framework',
          fontsize=16, fontweight='bold')
plt.xlabel('Evaluation Metric', fontsize=16)
plt.ylabel('Score (%)', fontsize=16)
plt.ylim(60, 105)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(axis='y', linestyle='--', linewidth=0.5)


plt.savefig('src/eval/result_visualization/llmasjudge_boxplot.png')

plt.show()

