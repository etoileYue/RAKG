import json
import logging
import os
import traceback

from src.llm_provider import LLMProvider
from src.logger import get_logger
from src.pipeline import NERPipeline
from src.pipeline import KnowledgeGraphQA
from src.textProcess import TextProcessor
from src.utils import get_ner_result_from_file
from src.utils import get_kg_result_from_file
from src.utils import validate_json_serializable

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

LOG_FILE_ENV_KEY = "RAKG_LOGGER_FILE"
DEFAULT_LOGGER_FILE = "Agent.log"


logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file=os.getenv(LOG_FILE_ENV_KEY, DEFAULT_LOGGER_FILE),
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
        ner_output_dir,
        rel_output_dir,
        sim_output_dir,
        graph_output_dir,
        skip_ner=False,
        skip_sim=False,
        skip_rel=False,
        existing_kg=None,
    ):
        """使用 NERPipeline 执行单个 topic 的处理流程。"""
        existing_kg = self._normalize_graph_input(existing_kg)
        has_existing_kg = bool(existing_kg.get("entities"))

        topic = topic_data.get("topic")
        text = topic_data.get("content")
        if not topic or not text:
            raise ValueError("topic_data must contain non-empty 'topic' and 'content'.")

        logger.info("Processing topic %s/%s with NERPipeline: %s", idx, total_topics, topic)

        processor = TextProcessor(text, topic)
        text_split = processor.process()

        ner_file_path = os.path.join(ner_output_dir, f"output_text_ner_{idx}.jsonl")
        rel_file_path = os.path.join(rel_output_dir, f"output_kg_{idx}.jsonl")
        sim_file_path = os.path.join(sim_output_dir, f"output_sim_{idx}.json")

        if skip_ner:
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

        if skip_sim:
            if not os.path.exists(sim_file_path):
                raise FileNotFoundError(
                    f"SIM cache file does not exist for skipped index {idx}: {sim_file_path}"
                )
            with open(sim_file_path, "r", encoding="utf-8") as sim_file:
                entity_list_process = json.load(sim_file)
            logger.info("Skip sim during processing text%s", idx)
        else:
            sim = self.similarity_result(ner_result) if ner_result else []
            entity_list_process = self.entity_Disambiguation(ner_result, sim) if ner_result else {}
            # entity_list_process = self.ensure_entity_aliases(entity_list_process)

            with open(sim_file_path, "w", encoding="utf-8") as sim_file:
                json.dump(
                    validate_json_serializable(entity_list_process),
                    sim_file,
                    ensure_ascii=False,
                    indent=4,
                )

        alias_resolution = {} # 别名解析表
        if has_existing_kg:
            entity_list_process, alias_resolution = self.align_entities_to_existing_graph(
                new_entities=entity_list_process,
                existing_graph=existing_kg,
            )
            logger.info(f"During processing topic{idx}: {topic}, merge the KG into existing KG.")
        else:
            entity_list_process = self._collapse_entities_by_name(entity_list_process)
            # 构建别名解析表
            for _, entity in entity_list_process.items():
                alias_resolution[entity.get("name", "")] = entity.get("name", "")
                for alias in entity.get("aliases", []):
                    alias_resolution[alias] = entity.get("name", "")

        related_kg_map = {}
        if has_existing_kg:
            for entity_id, entity in entity_list_process.items():
                related_context = self.build_related_kg_context(
                    graph_data=existing_kg,
                    entity_name=entity.get("name"),
                )
                if related_context:
                    related_kg_map[entity_id] = related_context

        if skip_rel:
            if not os.path.exists(rel_file_path):
                raise FileNotFoundError(
                    f"kg_reuslt cache file does not exist for skipped index {idx}: {rel_file_path}"
                )
            kg_result = get_kg_result_from_file(rel_file_path)
            logger.info("Skip rel during processing text%s", idx)
        else:
            kg_result = self.get_target_kg_all(
                entity_list_process,
                text_split["id_to_sentence"],
                text_split["sentences"],
                text_split["sentence_to_id"],
                text_split["vectors"],
                output_file=rel_file_path,
                related_kg_map=related_kg_map,
            )

        current_doc_kg = self.convert_knowledge_graph(
            kg_result,
            chunk_map=text_split.get("id_to_sentence", {}),
        )
        aliases_by_name = {}
        for _, entity in entity_list_process.items():
            canonical_name = entity.get("name")
            if not canonical_name:
                continue
            aliases_by_name.setdefault(canonical_name, [])
            aliases_by_name[canonical_name].extend(entity.get("aliases", []))

        for entity in current_doc_kg.get("entities", []):
            if not isinstance(entity, dict):
                continue
            canonical_name = entity.get("name", "")
            merged_aliases = entity.get("aliases", []) + aliases_by_name.get(canonical_name, [])
            entity["aliases"] = self._normalize_aliases(merged_aliases, canonical_name)

        merged_kg = (
            self.merge_knowledge_graphs(existing_kg, current_doc_kg)
            if has_existing_kg
            else current_doc_kg
        )
        # kg_json = validate_json_serializable(merged_kg)
        output_path = os.path.join(graph_output_dir, f"{idx}.json")
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(merged_kg, outfile, ensure_ascii=False, indent=4)

        # self.last_disambiguation_gray_queue = self.last_disambiguation_gray_queue
        logger.info("Saved KG for topic %s to %s", topic, output_path)
        return {
            "index": idx,
            "topic": topic,
            "output_path": output_path,
            "knowledge_graph": merged_kg,
            "current_doc_kg": current_doc_kg,
            "alias_resolution": alias_resolution,
        }

    def process_all_topics(
        self,
        json_path,
        output_dir,
        skip_construct=None,
        skip_ner_set=None,
        skip_sim_set=None,
        skip_rel_set=None,
        existing_kg=None
    ):
        with open(json_path, "r", encoding="utf-8") as file:
            topics = json.load(file)

        if not output_dir:
            raise ValueError("output_dir cannot be empty.")

        ner_output_dir = os.path.join(output_dir, "ner_data")
        rel_output_dir = os.path.join(output_dir, "rel_data")
        sim_output_dir = os.path.join(output_dir, "sim_data")
        graph_output_dir = os.path.join(output_dir, "RAKG_graph_re")


        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ner_output_dir, exist_ok=True)
        os.makedirs(rel_output_dir, exist_ok=True)
        os.makedirs(sim_output_dir, exist_ok=True)
        os.makedirs(graph_output_dir, exist_ok=True)

        has_existing_kg = bool(existing_kg)
        if has_existing_kg:
            global_kg = existing_kg.get("global", None)
        processed_count = 0
        failed_topics = []

        for idx, topic_data in enumerate(topics, start=1):
            if idx in skip_construct:
                continue
            
            cur_existing_kg = None
            if has_existing_kg:
                if global_kg:
                    cur_existing_kg = global_kg
                else:
                    cur_existing_kg = existing_kg.get(idx, None)

            skip_cur_ner = idx in skip_ner_set
            skip_cur_sim = idx in skip_sim_set
            skip_cur_rel = idx in skip_rel_set

            try:
                self.process(
                    topic_data=topic_data,
                    idx=idx,
                    total_topics=len(topics),
                    ner_output_dir=ner_output_dir,
                    rel_output_dir=rel_output_dir,
                    sim_output_dir=sim_output_dir,
                    graph_output_dir=graph_output_dir,
                    skip_ner=skip_cur_ner,
                    skip_sim=skip_cur_sim,
                    skip_rel=skip_cur_rel,
                    existing_kg=cur_existing_kg,
                )
                processed_count += 1
            except Exception as e:
                topic_name = topic_data.get("topic", "<unknown>")
                logger.error(
                    "Failed topic index=%s topic=%s error=%s",
                    idx,
                    topic_name,
                    str(e),
                )
                logger.error(traceback.format_exc())
                failed_topics.append(
                    {
                        "index": idx,
                        "topic": topic_name,
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
