from src.textProcess import TextProcessor
from src.kgAgent import NER_Agent
import json
import os
from typing import Any

import logging
from src.logger import get_logger
import traceback
from src.utils import get_ner_result_from_file

logger = get_logger(name="AgentLog",
                    level=logging.INFO,
                    log_file="Agent.log")


def _validate_json_serializable(data: Any) -> Any:
    """Validate JSON serializability without mutating string content."""
    json.dumps(data, ensure_ascii=False)
    return data


def process_all_topics(
    json_path,
    output_dir,
    done_offset=1,
    skip_ner_list=None,
    ner_output_dir=None,
    rel_output_dir=None,
):
    # Load JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        topics = json.load(file)

    # Normalize parameters
    skip_ner_set = set(skip_ner_list or [])
    if not output_dir:
        raise ValueError("output_dir cannot be empty.")
    ner_output_dir = ner_output_dir or os.path.join(output_dir, "ner_data")
    rel_output_dir = rel_output_dir or os.path.join(output_dir, "rel_data")

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ner_output_dir, exist_ok=True)
    os.makedirs(rel_output_dir, exist_ok=True)

    # Initialize related tools
    ner_agent = NER_Agent()

    processed_count = 0
    failed_topics = []

    # Iterate through each topic
    for idx, topic_data in enumerate(topics,start=1):
        if idx < done_offset:
            continue
        try:
            topic = topic_data.get('topic')
            text = topic_data.get('content')
            if not topic or not text:
                raise ValueError("topic_data must contain non-empty 'topic' and 'content'.")

            logger.info(f"Processing topic {idx}/{len(topics)}: {topic}")
            # print(f"Processing topic {idx}/{len(topics)}: {topic_data['topic']}")

            # Split text
            processor = TextProcessor(text, topic)
            text_split = processor.process()

            # Extract NER and knowledge graph
            # Initial NER
            ner_file_path = os.path.join(ner_output_dir, f"output_text_ner_{idx}.jsonl")
            rel_file_path = os.path.join(rel_output_dir, f"output_kg_{idx}.jsonl")
            if idx in skip_ner_set:
                file_path = ner_file_path
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"NER cache file does not exist for skipped index {idx}: {file_path}"
                    )
                ner_result = get_ner_result_from_file(file_path, text_split['sentence_to_id'])
                logger.info(f"Skip ner during processing text{idx}")
            else:
                ner_result = ner_agent.extract_from_text_multiply(
                    text_split['sentences'],
                    text_split['sentence_to_id'],
                    output_file=ner_file_path
                )
            sim = ner_agent.similarity_result(ner_result)
            # NER with entity disambiguation
            entity_list_process = ner_agent.entity_Disambiguation(ner_result, sim)
            kg_result = ner_agent.get_target_kg_all(
                entity_list_process,
                text_split['id_to_sentence'],
                text_split['sentences'],
                text_split['sentence_to_id'],
                text_split['vectors'],
                output_file=rel_file_path
            )
            # print("kg_result")
            # print(kg_result)
            kg_json = _validate_json_serializable(ner_agent.convert_knowledge_graph(kg_result))
            # print("kg_json")
            # print(kg_json)
            # Save to file
            output_path = os.path.join(output_dir, f"{idx}.json")
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(kg_json, outfile, ensure_ascii=False, indent=4)

            logger.info(f"Saved KG for topic {topic_data['topic']} to {output_path}")
            processed_count += 1
            # print(f"Saved KG for topic {topic_data['topic']} to {output_path}")
        except Exception as e:
            logger.error(f"Error generating knowledge graph for entry {idx}: {traceback.format_exc()}")
            failed_topics.append({
                "index": idx,
                "topic": topic_data.get("topic", "<unknown>"),
                "error": str(e),
            })
            # print(f"Error generating knowledge graph for entry {idx}: {str(e)}")
    
    summary = {
        "total_topics": len(topics),
        "processed_topics": processed_count,
        "failed_topics_count": len(failed_topics),
        "failed_topics": failed_topics,
    }
    summary_path = os.path.join(output_dir, "process_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    logger.info(f"Processing summary saved to {summary_path}")
    return summary

# Example call
if __name__ == "__main__":
    json_path = "../../data/raw/MINE.json"  # Replace with your JSON file path
    output_dir = "../../data/reproduce/processed/RAKG_graph_re"  # Output directory
    skip_ner_list = [17]
    process_all_topics(
        json_path,
        output_dir,
        done_offset=17,
        skip_ner_list=skip_ner_list,
        ner_output_dir="../../data/reproduce/processed/llmasjudge/ner_data",
        rel_output_dir="../../data/reproduce/processed/llmasjudge/rel_data",
    )
