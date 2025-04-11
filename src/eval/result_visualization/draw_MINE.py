import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import gaussian_kde


plt.rcParams.update({
    'font.size': 16,       
    'axes.labelsize': 14,   
    'legend.fontsize': 14, 
    'xtick.labelsize': 12,  
    'ytick.labelsize': 12   
})

# 1. Load data
with open("src/eval/result_visualization/accuracy_results.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# 2. Parse accuracy
method_accuracies = {"graphrag": [], "kggen": [], "RAKG": []}
for entry in data:
    method = entry["method"]
    acc = float(entry["accuracy"].replace("%", ""))
    method_accuracies[method].append(round(acc, 2)) 

# 3. Define bin centers 
acc_bins = [round(i / 15 * 100, 2) for i in range(16)]
x = np.array(acc_bins)
bar_width = (x[1] - x[0]) / 4


frequencies = {}
for method in method_accuracies:
    counter = Counter(method_accuracies[method])
    frequencies[method] = [counter.get(b, 0) for b in acc_bins]

max_count = max([max(frequencies[m]) for m in frequencies])


plt.figure(figsize=(14, 7))


plt.bar(x - bar_width, frequencies["graphrag"], width=bar_width, label="graphrag", color="skyblue", edgecolor='black', alpha=0.6)
plt.bar(x, frequencies["kggen"], width=bar_width, label="kggen", color="lightgreen", edgecolor='black', alpha=0.6)
plt.bar(x + bar_width, frequencies["RAKG"], width=bar_width, label="RAKG", color="salmon", edgecolor='black', alpha=0.6)


x_vals = np.linspace(0, 100, 500)
colors = {"graphrag": "skyblue", "kggen": "lightgreen", "RAKG": "salmon"}
for method in method_accuracies:
    accs = method_accuracies[method]
    if len(accs) > 2:
        kde = gaussian_kde(accs)
        kde_vals = kde.evaluate(x_vals)
        kde_vals_scaled = kde_vals / kde_vals.max() * max_count
        plt.plot(x_vals, kde_vals_scaled, color=colors[method], linewidth=2)


averages = {
    "RAKG": 95.81,
    "KGGen": 86.48,
    "GraphRAG": 89.71
}
color_map = {
    "RAKG": "salmon",
    "KGGen": "lightgreen",
    "GraphRAG": "skyblue"
}
for name, acc in averages.items():
    plt.axvline(x=acc, color=color_map[name], linestyle='--', linewidth=1.5, label=f"{name}: {acc:.2f}%")


plt.xticks(x, [f"{val:.2f}%" for val in x], rotation=45)
plt.xlabel("Accuracy (%)")
plt.ylabel("Frequency")
plt.title("Distribution of Accuracy for Each Method + KDE", fontsize=15, fontweight="bold")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("src/eval/result_visualization/MINE_kde_and_bars.png", dpi=300)
plt.show()