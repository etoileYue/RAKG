from scripts.text_spilt_embed import TextProcessor
from scripts.extract_entity_withRA_withdescription import NER_Agent
import json
import os

def convert_to_valid_json(data):
    def format_value(obj):
        """Recursively handle single quote issues in all values"""
        if isinstance(obj, dict):
            return {k: format_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [format_value(item) for item in obj]
        if isinstance(obj, str):
            # Convert single quotes to double quotes (choose whether to keep based on requirements)
            return obj.replace("'", '"')
        return obj

    # Process the entire data structure
    processed_data = format_value(data)
    # Convert to strict JSON format
    return json.dumps(processed_data, 
                     indent=2, 
                     ensure_ascii=False,
                     separators=(',', ': '))

def process_all_topics(json_path, output_dir):
    # Load JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        topics = json.load(file)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize related tools
    ner_agent = NER_Agent()

    start_index = 1
    # Iterate through each topic
    for idx, topic_data in enumerate(topics,start=1):
        if idx < start_index:
            continue  # Skip previous entries
        try:
            print(f"Processing topic {idx}/{len(topics)}: {topic_data['topic']}")

            # Get text and topic
            text = topic_data['content']
            topic = topic_data['topic']

            # Split text
            processor = TextProcessor(text, topic)
            text_split = processor.process()

            # Extract NER and knowledge graph
            # Initial NER
            ner_result = ner_agent.extract_from_text_multiply(text_split['sentences'], text_split['sentence_to_id'],output_file=f"data/processed/llmasjudge/ner_data/output_text_ner_{idx}.jsonl")
            sim = ner_agent.similartiy_result(ner_result)
            # NER with entity disambiguation
            entity_list_process = ner_agent.entity_Disambiguation(ner_result, sim)
            kg_result = ner_agent.get_target_kg_all(entity_list_process, text_split['id_to_sentence'],text_split['sentences'],text_split['sentence_to_id'],text_split['vectors'],output_file=f"data/processed/llmasjudge/rel_data/output_kg_{idx}.jsonl")
            print("kg_result")
            print(kg_result)
            converted_kg = ner_agent.convert_knowledge_graph(kg_result)
            kg_json = convert_to_valid_json(converted_kg)
            print("kg_json")
            print(kg_json)
            # Save to file
            output_path = os.path.join(output_dir, f"{idx}.json")
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(kg_json, outfile, ensure_ascii=False, indent=4)

            print(f"Saved KG for topic {topic_data['topic']} to {output_path}")
        except Exception as e:
            print(f"Error generating knowledge graph for entry {idx}: {str(e)}")
        

# Example call
if __name__ == "__main__":
    json_path = "data/raw/MINE.json"  # Replace with your JSON file path
    output_dir = "data/processed/RAKG_graph_v1"  # Output directory
    process_all_topics(json_path, output_dir)