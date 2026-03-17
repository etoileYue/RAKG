import json
import logging
import os
import traceback

from src.kg_agent import KnowledgeGraphQA
from src.kg_agent import NERPipeline
from src.llm_provider import LLMProvider
from src.logger import get_logger
from src.textProcess import TextProcessor
from src.utils import get_ner_result_from_file
from src.utils import validate_json_serializable

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file="Agent.log",
)


class NER_Agent(NERPipeline, KnowledgeGraphQA):
    def __init__(self):
        self.llm_provider = LLMProvider()
        self.model = self.llm_provider.get_llm()
        self.similarity_model = self.llm_provider.get_similarity_model()
        self.embeddings = self.llm_provider.get_embedding_model()
        self.last_disambiguation_gray_queue = []
        self._qa_graph_index_cache = {}
        self._qa_default_graph_cache_key = None

    def process(
        self,
        topic_data,
        idx,
        total_topics,
        output_dir,
        ner_output_dir,
        rel_output_dir,
        skip_ner_set=None,
    ):
        skip_ner_set = skip_ner_set or set()

        topic = topic_data.get("topic")
        text = topic_data.get("content")
        if not topic or not text:
            raise ValueError("topic_data must contain non-empty 'topic' and 'content'.")

        logger.info("Processing topic %s/%s: %s", idx, total_topics, topic)

        processor = TextProcessor(text, topic)
        text_split = processor.process()

        ner_file_path = os.path.join(ner_output_dir, f"output_text_ner_{idx}.jsonl")
        rel_file_path = os.path.join(rel_output_dir, f"output_kg_{idx}.jsonl")

        if idx in skip_ner_set:
            if not os.path.exists(ner_file_path):
                raise FileNotFoundError(
                    f"NER cache file does not exist for skipped index {idx}: {ner_file_path}"
                )
            ner_result = get_ner_result_from_file(ner_file_path, text_split["sentence_to_id"])
            logger.info("Skip ner during processing text%s", idx)
        else:
            ner_result = self.extract_from_text_multiply(
                text_split["sentences"],
                text_split["sentence_to_id"],
                output_file=ner_file_path,
            )

        sim = self.similarity_result(ner_result)
        entity_list_process = self.entity_Disambiguation(ner_result, sim)
        kg_result = self.get_target_kg_all(
            entity_list_process,
            text_split["id_to_sentence"],
            text_split["sentences"],
            text_split["sentence_to_id"],
            text_split["vectors"],
            output_file=rel_file_path,
        )

        kg_json = validate_json_serializable(self.convert_knowledge_graph(kg_result))
        output_path = os.path.join(output_dir, f"{idx}.json")
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(kg_json, outfile, ensure_ascii=False, indent=4)

        logger.info("Saved KG for topic %s to %s", topic, output_path)
        return {"index": idx, "topic": topic, "output_path": output_path}

    def process_all_topics(
        self,
        json_path,
        output_dir,
        done_offset=0,
        skip_ner_list=None,
        ner_output_dir=None,
        rel_output_dir=None,
    ):
        with open(json_path, "r", encoding="utf-8") as file:
            topics = json.load(file)

        skip_ner_set = set(skip_ner_list or [])
        if not output_dir:
            raise ValueError("output_dir cannot be empty.")

        ner_output_dir = ner_output_dir or os.path.join(output_dir, "ner_data")
        rel_output_dir = rel_output_dir or os.path.join(output_dir, "rel_data")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ner_output_dir, exist_ok=True)
        os.makedirs(rel_output_dir, exist_ok=True)

        processed_count = 0
        failed_topics = []

        for idx, topic_data in enumerate(topics, start=1):
            if idx < done_offset:
                continue

            try:
                self.process(
                    topic_data=topic_data,
                    idx=idx,
                    total_topics=len(topics),
                    output_dir=output_dir,
                    ner_output_dir=ner_output_dir,
                    rel_output_dir=rel_output_dir,
                    skip_ner_set=skip_ner_set,
                )
                processed_count += 1
            except Exception as e:
                logger.error(
                    "Error generating knowledge graph for entry %s: %s",
                    idx,
                    traceback.format_exc(),
                )
                failed_topics.append(
                    {
                        "index": idx,
                        "topic": topic_data.get("topic", "<unknown>"),
                        "error": str(e),
                    }
                )

        summary = {
            "total_topics": len(topics),
            "processed_topics": processed_count,
            "failed_topics_count": len(failed_topics),
            "failed_topics": failed_topics,
        }
        summary_path = os.path.join(output_dir, "process_summary.json")
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        logger.info("Processing summary saved to %s", summary_path)
        return summary
