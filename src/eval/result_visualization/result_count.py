import json
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path, has_attributes=True):

    entities = []
    relations_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            entities.append(data['entities'])
            # 根据文件类型处理关系数据
            if has_attributes:
                relations_data.append(data['relations'] + data.get('attributes', 0))
            else:
                relations_data.append(data['relations'])
    return entities, relations_data


rak_ent, rak_rel = load_data('data/processed/stat_results_RAKG.jsonl')
kggen_ent, kggen_rel = load_data('data/processed/stat_results_KGGEN.jsonl', False)

plt.figure(figsize=(12, 7))
positions = [1, 2, 4, 5]  
labels = ['RAKG\nEntities',  'RAKG\nRelations+Attr', 'KGGEN\nEntities','KGGEN\nRelations']
colors = ['#1f77b4',  '#1f77b4', '#ff7f0e','#ff7f0e'] 


box = plt.boxplot(
    [rak_ent,  rak_rel, kggen_ent, kggen_rel],
    positions=positions,
    widths=0.6,
    patch_artist=True,
    labels=labels,
    showfliers=True
)


for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)


plt.title('Distribution Comparison Between Two Datasets', fontsize=14)
plt.ylabel('Count', fontsize=12)
plt.xticks(fontsize=10, rotation=0)


plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.gca().set_axisbelow(True)


legend_elements = [
    plt.Rectangle((0,0),1,1, fc='#1f77b4', alpha=0.8, label='RAKG'),
    plt.Rectangle((0,0),1,1, fc='#ff7f0e', alpha=0.8, label='KGGEN')
]
plt.legend(handles=legend_elements, loc='upper right')


plt.tight_layout()
plt.show()