import pandas as pd
import json
from pathlib import Path

def convert_to_graph_json(entities_df, relations_df, output_path):

    entities = entities_df.apply(lambda row: {
        "name": row["title"],         # Map title field to name
        "description": row.get("description", "")  # Handle potentially missing fields
    }, axis=1).tolist()


    relations = []
    for _, row in relations_df.iterrows():
        relation_entry = [
            row["source"],           # Head entity
            ":has_relation_with",     # Fixed relation type (adjust as needed)
            row["target"],           # Tail entity
            row.get("description", "")  # Relation description
        ]
        relations.append(relation_entry)

    # Build complete data structure
    graph_data = {
        "entities": entities,
        "relations": relations
    }

    # Write to JSON file (with Chinese encoding handling)
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, 
                 ensure_ascii=False,  # Preserve non-ASCII characters
                 indent=2)           # Pretty print format

    print(f"JSON file saved to: {output_path.resolve()}")
def load_graph_from_multiple_parquet(base_path,output_path):
    # Merge all entity data
    entities_dfs = []
    relations_dfs = []
    

    # Build specific file paths
    entity_path = Path(base_path) / "entities.parquet"
    relation_path = Path(base_path) / "relationships.parquet"
    
    # Read and collect data
    entities_dfs.append(pd.read_parquet(entity_path))
    relations_dfs.append(pd.read_parquet(relation_path))
    all_entities = pd.concat(entities_dfs, ignore_index=True)
    all_relations = pd.concat(relations_dfs, ignore_index=True)

    convert_to_graph_json(all_entities, all_relations, output_path)

    return 0

for i in range(1, 106):
    json_file = f"data/graphrag_qwen/ragtest{i}/output"
    # output_file = json_file.replace(".json", "_results.json")
    output_file = f"data/processed/graphrag_graph{i}_results.json"
    print(f"Processing file: {json_file}")
    load_graph_from_multiple_parquet(json_file,output_file)