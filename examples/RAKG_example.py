import os
import sys
import json
import argparse

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.textPrcess import TextProcessor
from src.kgAgent import NER_Agent

def convert_to_valid_json(data):
    """
    Convert Python dictionary to a string format that strictly conforms to JSON specifications
    Automatically handles: double quotes for keys, special character escaping, single quote conversion, etc.
    """
    def format_value(obj):
        """Recursively process all values to handle single quote issues"""
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

def process_text(text, topic, output_path):
    """Process a single text and generate a knowledge graph"""
    # try:
    # Initialize related tools
    ner_agent = NER_Agent()

    # Split text
    processor = TextProcessor(text, topic)
    text_split = processor.process()

    # Create temporary output file paths
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    temp_ner_file = os.path.join(output_dir, "temp_ner_output.jsonl")
    temp_kg_file = os.path.join(output_dir, "temp_kg_output.jsonl")

    # Extract NER and knowledge graph
    ner_result = ner_agent.extract_from_text_multiply(text_split['sentences'], text_split['sentence_to_id'], output_file=temp_ner_file)
    sim = ner_agent.similartiy_result(ner_result)
    entity_list_process = ner_agent.entity_Disambiguation(ner_result, sim)
    kg_result = ner_agent.get_target_kg_all(entity_list_process, text_split['id_to_sentence'],
                                            text_split['sentences'], text_split['sentence_to_id'],
                                            text_split['vectors'], output_file=temp_kg_file)
    
    print("Knowledge Graph Result:")
    print(kg_result)
    
    converted_kg = ner_agent.convert_knowledge_graph(kg_result)
    kg_json = convert_to_valid_json(converted_kg)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(kg_json, outfile, ensure_ascii=False, indent=4)
    
    # Clean up temporary files
    if os.path.exists(temp_ner_file):
        os.remove(temp_ner_file)
    if os.path.exists(temp_kg_file):
        os.remove(temp_kg_file)
    
    print(f"Knowledge graph has been saved to: {output_path}")
    return True
    # except Exception as e:
    #     print(f"Error processing text: {str(e)}")
    #     return False

def process_file(input_path, output_dir):
    """Process all topics in the input file"""
    try:
        # Load JSON file
        with open(input_path, 'r', encoding='utf-8') as file:
            topics = json.load(file)

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, topic_data in enumerate(topics, start=1):
            try:
                print(f"Processing topic {idx}/{len(topics)}: {topic_data['topic']}")
                output_path = os.path.join(output_dir, f"{idx}.json")
                process_text(topic_data['content'], topic_data['topic'], output_path)
            except Exception as e:
                print(f"Error processing topic {idx}: {str(e)}")
                continue
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Knowledge Graph Generation Tool')
    parser.add_argument('--input', '-i', required=True, help='Input file path or direct text')
    parser.add_argument('--output', '-o', required=True, help='Output file or directory path')
    parser.add_argument('--topic', '-t', help='Topic name (required when input is direct text)')
    parser.add_argument('--is-text', action='store_true', help='Specify whether input is direct text')

    args = parser.parse_args()

    if args.is_text:
        if not args.topic:
            print("Error: Topic name (--topic) must be provided when input is direct text")
            sys.exit(1)
        process_text(args.input, args.topic, args.output)
    else:
        process_file(args.input, args.output)

if __name__ == "__main__":
    main()