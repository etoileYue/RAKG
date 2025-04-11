import json


import json
import os

base_dir = "data/processed/kggen_graph"
output_file = "stat_results_KGGEN.jsonl"

with open(output_file, "w") as out_f:
    for i in range(1, 106):
        file_path = os.path.join(base_dir, f"{i}.json")
        try:
            with open(file_path, "r") as f:
                data = json.load(f)  
                # Calculate statistics
                result = {
                    "file_name": f"{i}.json",
                    "entities": len(data["entities"]),
                    "relations": len(data["relations"])
                }
                
                # Write to JSONL file
                out_f.write(json.dumps(result) + "\n")
                
        except FileNotFoundError:
            print(f"File {file_path} does not exist")
        except json.JSONDecodeError:
            print(f"File {file_path} has invalid JSON format")
        except KeyError as e:
            print(f"File {file_path} is missing key field: {str(e)}")

