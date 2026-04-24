import json
import hashlib
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

import src.config as config_module
from src.llm_provider import LLMProvider
from src.logger import get_logger
from src.pipeline import NERPipeline
from src.pipeline import KnowledgeGraphQA
from src.pipeline.shared import debug_logger
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
    CHECKPOINT_VERSION = 1
    CHECKPOINT_FILE_NAME = "checkpoint_state.json"
    STAGE_PENDING = "pending"
    STAGE_RUNNING = "running"
    STAGE_DONE = "done"
    STAGE_FAILED = "failed"
    STAGE_ORDER = ("ner", "sim", "rel")

    def __init__(self):
        self.llm_provider = LLMProvider()
        self.model = self.llm_provider.get_llm()
        self.similarity_model = self.llm_provider.get_similarity_model()
        self.embeddings = self.llm_provider.get_embedding_model()
        self.disambiguation_config = self.set_disambiguation_config()
        self.reset_disambiguation_runtime_state()
        self.last_disambiguation_gray_queue = []
        self._qa_graph_index_cache = {}
        self._qa_default_graph_cache_key = None

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _checkpoint_path(cls, output_dir: str) -> str:
        return os.path.join(output_dir, cls.CHECKPOINT_FILE_NAME)

    @staticmethod
    def _compute_input_fingerprint(topics: list[dict[str, Any]]) -> str:
        payload = json.dumps(topics, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_checkpoint_state(self, checkpoint_path: str) -> dict[str, Any] | None:
        if not os.path.exists(checkpoint_path):
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
                data = json.load(checkpoint_file)
        except Exception as exc:
            logger.warning("Failed to read checkpoint state from %s: %s", checkpoint_path, exc)
            return None

        if not isinstance(data, dict):
            logger.warning("Checkpoint file has invalid shape, expected object: %s", checkpoint_path)
            return None
        return data

    @staticmethod
    def _write_checkpoint_state(checkpoint_path: str, checkpoint_state: dict[str, Any]) -> None:
        with open(checkpoint_path, "w", encoding="utf-8") as checkpoint_file:
            json.dump(checkpoint_state, checkpoint_file, ensure_ascii=False, indent=2)

    @classmethod
    def _normalize_stage_status(cls, value: Any) -> str:
        if value in {
            cls.STAGE_PENDING,
            cls.STAGE_RUNNING,
            cls.STAGE_DONE,
            cls.STAGE_FAILED,
        }:
            return str(value)
        return cls.STAGE_PENDING

    def _ensure_topic_checkpoint_entry(
        self,
        checkpoint_state: dict[str, Any],
        *,
        idx: int,
        topic_name: str,
        ner_file_path: str,
        sim_file_path: str,
        rel_file_path: str,
        graph_file_path: str,
    ) -> dict[str, Any]:
        topics_state = checkpoint_state.setdefault("topics", {})
        topic_key = str(idx)
        entry = topics_state.get(topic_key)
        if not isinstance(entry, dict):
            entry = {}
            topics_state[topic_key] = entry

        stages = entry.get("stages", {})
        if not isinstance(stages, dict):
            stages = {}
        normalized_stages = {}
        for stage in self.STAGE_ORDER:
            normalized_stages[stage] = self._normalize_stage_status(stages.get(stage))
        entry["stages"] = normalized_stages

        artifacts = entry.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        artifacts["ner_file"] = os.path.abspath(ner_file_path)
        artifacts["sim_file"] = os.path.abspath(sim_file_path)
        artifacts["rel_file"] = os.path.abspath(rel_file_path)
        artifacts["graph_file"] = os.path.abspath(graph_file_path)
        entry["artifacts"] = artifacts

        counts = entry.get("counts", {})
        if not isinstance(counts, dict):
            counts = {}
        counts["ner_lines"] = int(counts.get("ner_lines") or 0)
        counts["rel_lines"] = int(counts.get("rel_lines") or 0)
        entry["counts"] = counts

        entry["topic_name"] = str(topic_name or "")
        entry["updated_at"] = self._now_iso()
        return entry

    def _update_stage_checkpoint(
        self,
        checkpoint_state: dict[str, Any] | None,
        checkpoint_path: str | None,
        *,
        topic_entry: dict[str, Any] | None,
        stage: str,
        status: str,
        ner_lines: int | None = None,
        rel_lines: int | None = None,
    ) -> None:
        if not checkpoint_state or not checkpoint_path or not topic_entry:
            return

        stages = topic_entry.setdefault("stages", {})
        stages[stage] = self._normalize_stage_status(status)

        counts = topic_entry.setdefault("counts", {})
        if ner_lines is not None:
            counts["ner_lines"] = int(ner_lines)
        if rel_lines is not None:
            counts["rel_lines"] = int(rel_lines)

        topic_entry["updated_at"] = self._now_iso()
        self._write_checkpoint_state(checkpoint_path, checkpoint_state)

    def _prepare_checkpoint_state(
        self,
        *,
        output_dir: str,
        topics: list[dict[str, Any]],
        topic_indices: list[int] | None = None,
        ner_output_dir: str,
        rel_output_dir: str,
        sim_output_dir: str,
        graph_output_dir: str,
        force_rebuild: bool = False,
    ) -> tuple[dict[str, Any], str, bool]:
        checkpoint_path = self._checkpoint_path(output_dir)
        existing_state = self._read_checkpoint_state(checkpoint_path)
        fingerprint = self._compute_input_fingerprint(topics)

        if (
            existing_state
            and not force_rebuild
            and existing_state.get("input_fingerprint") != fingerprint
        ):
            raise ValueError(
                "Input fingerprint mismatch for existing output_dir checkpoint. "
                "Use a different output_dir or set force_rebuild=true."
            )

        checkpoint_loaded = bool(
            existing_state
            and isinstance(existing_state, dict)
            and existing_state.get("version") == self.CHECKPOINT_VERSION
            and existing_state.get("input_fingerprint") == fingerprint
        )

        if force_rebuild or not checkpoint_loaded:
            checkpoint_state: dict[str, Any] = {
                "version": self.CHECKPOINT_VERSION,
                "input_fingerprint": fingerprint,
                "topics": {},
            }
        else:
            checkpoint_state = existing_state
            checkpoint_state["version"] = self.CHECKPOINT_VERSION
            checkpoint_state["input_fingerprint"] = fingerprint
            checkpoint_state.setdefault("topics", {})

        if topic_indices is not None and len(topic_indices) != len(topics):
            raise ValueError("topic_indices must have the same length as topics.")

        indexed_topics = (
            zip(topic_indices, topics)
            if topic_indices is not None
            else enumerate(topics, start=1)
        )
        for idx, topic_data in indexed_topics:
            topic_name = str(topic_data.get("topic", ""))
            ner_file_path = os.path.join(ner_output_dir, f"output_text_ner_{idx}.jsonl")
            rel_file_path = os.path.join(rel_output_dir, f"output_kg_{idx}.jsonl")
            sim_file_path = os.path.join(sim_output_dir, f"output_sim_{idx}.json")
            graph_file_path = os.path.join(graph_output_dir, f"{idx}.json")
            self._ensure_topic_checkpoint_entry(
                checkpoint_state,
                idx=idx,
                topic_name=topic_name,
                ner_file_path=ner_file_path,
                sim_file_path=sim_file_path,
                rel_file_path=rel_file_path,
                graph_file_path=graph_file_path,
            )

        self._write_checkpoint_state(checkpoint_path, checkpoint_state)
        return checkpoint_state, checkpoint_path, checkpoint_loaded

    @staticmethod
    def _load_sim_cache(sim_file_path: str) -> tuple[dict[str, Any] | None, bool]:
        if not os.path.exists(sim_file_path):
            return None, False

        try:
            with open(sim_file_path, "r", encoding="utf-8") as sim_file:
                sim_data = json.load(sim_file)
        except Exception as exc:
            logger.warning("Invalid SIM cache file %s: %s", sim_file_path, exc)
            return None, False

        if not isinstance(sim_data, dict):
            logger.warning("Invalid SIM cache structure in %s: expected JSON object.", sim_file_path)
            return None, False
        return sim_data, True

    def _get_runtime_model_names(self) -> dict[str, str]:
        if hasattr(self.llm_provider, "get_model_names"):
            model_names = self.llm_provider.get_model_names()
            if isinstance(model_names, dict):
                return {
                    "main_model": str(model_names.get("main_model", "")),
                    "similarity_model": str(model_names.get("similarity_model", "")),
                    "embedding_model": str(model_names.get("embedding_model", "")),
                }
        return {
            "main_model": "",
            "similarity_model": "",
            "embedding_model": "",
        }

    def _log_runtime_configuration(
        self,
        *,
        total_topics: int,
        existing_kg: Any,
        force_rebuild: bool,
        auto_resume: bool,
    ) -> None:
        model_names = self._get_runtime_model_names()
        disambiguation = self.get_disambiguation_config()
        logger.info(
            "Run config: total_topics=%s main_model=%s similarity_model=%s embedding_model=%s",
            total_topics,
            model_names["main_model"],
            model_names["similarity_model"],
            model_names["embedding_model"],
        )
        logger.info(
            "Run config: prompt_language=%s use_openai=%s llm_parallel_enabled=%s llm_parallel_max_workers=%s",
            getattr(config_module, "PROMPT_LANGUAGE", "zh"),
            bool(getattr(config_module, "USE_OPENAI", False)),
            bool(getattr(config_module, "LLM_PARALLEL_ENABLED", False)),
            int(getattr(config_module, "LLM_PARALLEL_MAX_WORKERS", 0)),
        )
        logger.info(
            "Run config: similarity_threshold=%s type_gate_enabled=%s per_entity_top_k=%s description_max_chars=%s direct_merge_enabled=%s existing_kg=%s force_rebuild=%s auto_resume=%s",
            disambiguation["similarity_threshold"],
            disambiguation["type_gate_enabled"],
            disambiguation["per_entity_top_k"],
            disambiguation["description_max_chars"],
            disambiguation["direct_merge_enabled"],
            bool(existing_kg),
            bool(force_rebuild),
            bool(auto_resume),
        )

    def _build_similarity_stage_plan(
        self,
        entities: dict[str, Any],
        *,
        threshold: float,
    ) -> dict[str, Any]:
        if not entities:
            return {
                "candidates": [],
                "optimized": {
                    "raw_candidates": 0,
                    "after_type_gate": 0,
                    "after_topk": 0,
                    "direct_merged": 0,
                    "direct_pairs": [],
                    "llm_candidates": [],
                },
            }

        resolved_config = self.get_disambiguation_config()
        candidates = self.similarity_candidates(
            left_entities=entities,
            right_entities=None,
            threshold=threshold,
        )
        optimized = self._optimize_similarity_candidates(
            candidates=candidates,
            left_entities=entities,
            right_entities=entities,
            same_side_compare=True,
            disambiguation_config=resolved_config,
        )
        return {
            "candidates": candidates,
            "optimized": optimized,
        }

    def process(
        self,
        topic_data,
        idx,
        total_topics,
        ner_output_dir,
        rel_output_dir,
        sim_output_dir,
        graph_output_dir,
        skip_ner=None,
        skip_sim=None,
        skip_rel=None,
        existing_kg=None,
        checkpoint_state=None,
        checkpoint_path=None,
        auto_resume=True,
    ):
        """使用 NERPipeline 执行单个 topic 的处理流程。"""
        existing_kg = self._normalize_graph_input(existing_kg)
        has_existing_kg = bool(existing_kg.get("entities"))
        resolved_disambiguation_config = self.set_disambiguation_config()
        self.reset_disambiguation_runtime_state()
        similarity_threshold = resolved_disambiguation_config["similarity_threshold"]

        topic = topic_data.get("topic")
        text = topic_data.get("content")
        if not topic or not text:
            raise ValueError("topic_data must contain non-empty 'topic' and 'content'.")

        logger.info("Topic %s/%s started: %s", idx, total_topics, topic)

        processor = TextProcessor(text, topic)
        text_split = processor.process()

        ner_file_path = os.path.join(ner_output_dir, f"output_text_ner_{idx}.jsonl")
        rel_file_path = os.path.join(rel_output_dir, f"output_kg_{idx}.jsonl")
        sim_file_path = os.path.join(sim_output_dir, f"output_sim_{idx}.json")
        output_path = os.path.join(graph_output_dir, f"{idx}.json")

        topic_checkpoint = None
        if checkpoint_state is not None and checkpoint_path is not None:
            topic_checkpoint = self._ensure_topic_checkpoint_entry(
                checkpoint_state,
                idx=idx,
                topic_name=topic,
                ner_file_path=ner_file_path,
                sim_file_path=sim_file_path,
                rel_file_path=rel_file_path,
                graph_file_path=output_path,
            )
            self._write_checkpoint_state(checkpoint_path, checkpoint_state)

        explicit_skip_ner = skip_ner if isinstance(skip_ner, bool) else None
        explicit_skip_sim = skip_sim if isinstance(skip_sim, bool) else None
        explicit_skip_rel = skip_rel if isinstance(skip_rel, bool) else None

        should_try_resume_ner = explicit_skip_ner is True or (
            explicit_skip_ner is None and auto_resume
        )
        should_try_resume_sim = explicit_skip_sim is True or (
            explicit_skip_sim is None and auto_resume
        )
        should_try_resume_rel = explicit_skip_rel is True or (
            explicit_skip_rel is None and auto_resume
        )

        ner_result = {}
        ner_reused = False
        if should_try_resume_ner:
            ner_result = get_ner_result_from_file(ner_file_path, text_split["sentence_to_id"]) if os.path.exists(ner_file_path) else {}
            if ner_result:
                ner_reused = True
                logger.info("NER stage: reuse cache, skip stage")
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="ner",
                    status=self.STAGE_DONE,
                    ner_lines=len(ner_result),
                )

        if explicit_skip_ner is True and not ner_reused:
            self._update_stage_checkpoint(
                checkpoint_state,
                checkpoint_path,
                topic_entry=topic_checkpoint,
                stage="ner",
                status=self.STAGE_FAILED,
            )
            if not os.path.exists(ner_file_path):
                raise FileNotFoundError(
                    f"NER cache file does not exist for skipped index {idx}: {ner_file_path}"
                )
            raise ValueError(
                f"NER cache file has no usable records for skipped index {idx}: {ner_file_path}"
            )

        if not ner_reused:
            try:
                logger.info("NER stage: tasks=%s", len(text_split["sentences"]))
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="ner",
                    status=self.STAGE_RUNNING,
                )
                ner_result = self.extract_from_text_multiply(
                    text_split["sentences"], text_split["sentence_to_id"], output_file=ner_file_path
                )
                logger.info("NER stage finished: extracted_entities=%s", len(ner_result))
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="ner",
                    status=self.STAGE_DONE,
                    ner_lines=len(ner_result),
                )
            except Exception:
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="ner",
                    status=self.STAGE_FAILED,
                )
                raise

        sim_reused = False
        entity_list_process: dict[str, Any] = {}
        if should_try_resume_sim:
            cached_sim, sim_valid = self._load_sim_cache(sim_file_path)
            if sim_valid and cached_sim is not None:
                entity_list_process = cached_sim
                sim_reused = True
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="sim",
                    status=self.STAGE_DONE,
                )

        if explicit_skip_sim is True and not sim_reused:
            self._update_stage_checkpoint(
                checkpoint_state,
                checkpoint_path,
                topic_entry=topic_checkpoint,
                stage="sim",
                status=self.STAGE_FAILED,
            )
            if not os.path.exists(sim_file_path):
                raise FileNotFoundError(
                    f"SIM cache file does not exist for skipped index {idx}: {sim_file_path}"
                )
            raise ValueError(
                f"SIM cache file has invalid structure for skipped index {idx}: {sim_file_path}"
            )

        if not sim_reused:
            try:
                sim_plan = self._build_similarity_stage_plan(
                    ner_result,
                    threshold=similarity_threshold,
                )
                optimized = sim_plan["optimized"]
                logger.info(
                    "SIM stage: raw_candidates=%s llm_candidates=%s direct_merged=%s",
                    optimized["raw_candidates"],
                    len(optimized["llm_candidates"]),
                    optimized["direct_merged"],
                )
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="sim",
                    status=self.STAGE_RUNNING,
                )
                sim = (
                    self.similarity_result(
                        ner_result,
                        threshold=similarity_threshold,
                        prepared_run=sim_plan,
                    )
                    if ner_result
                    else []
                )
                entity_list_process = self.entity_Disambiguation(ner_result, sim) if ner_result else {}
                logger.info(
                    "SIM stage finished: input_entities=%s merged_entities=%s",
                    len(ner_result),
                    len(entity_list_process),
                )

                with open(sim_file_path, "w", encoding="utf-8") as sim_file:
                    json.dump(
                        validate_json_serializable(entity_list_process),
                        sim_file,
                        ensure_ascii=False,
                        indent=4,
                    )
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="sim",
                    status=self.STAGE_DONE,
                )
            except Exception:
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="sim",
                    status=self.STAGE_FAILED,
                )
                raise
        else:
            logger.info("SIM stage: 复用缓存，跳过本阶段")

        alias_resolution = {} # 别名解析表
        if has_existing_kg:
            entity_list_process, alias_resolution = self.align_entities_to_existing_graph(
                new_entities=entity_list_process,
                existing_graph=existing_kg,
                threshold=similarity_threshold,
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

        rel_reused = False
        if should_try_resume_rel:
            kg_result = get_kg_result_from_file(rel_file_path) if os.path.exists(rel_file_path) else {}
            if kg_result:
                rel_reused = True
                logger.info("REL stage: 复用缓存，跳过本阶段")
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="rel",
                    status=self.STAGE_DONE,
                    rel_lines=len(kg_result),
                )

        if explicit_skip_rel is True and not rel_reused:
            self._update_stage_checkpoint(
                checkpoint_state,
                checkpoint_path,
                topic_entry=topic_checkpoint,
                stage="rel",
                status=self.STAGE_FAILED,
            )
            if not os.path.exists(rel_file_path):
                raise FileNotFoundError(
                    f"kg_reuslt cache file does not exist for skipped index {idx}: {rel_file_path}"
                )
            raise ValueError(
                f"REL cache file has no usable records for skipped index {idx}: {rel_file_path}"
            )

        if not rel_reused:
            try:
                logger.info("REL stage: tasks=%s", len(entity_list_process))
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="rel",
                    status=self.STAGE_RUNNING,
                )
                kg_result = self.get_target_kg_all(
                    entity_list_process,
                    text_split["id_to_sentence"],
                    text_split["sentences"],
                    text_split["sentence_to_id"],
                    text_split["vectors"],
                    output_file=rel_file_path,
                    related_kg_map=related_kg_map,
                )
                logger.info("REL stage finished: extracted_entities=%s", len(kg_result))
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="rel",
                    status=self.STAGE_DONE,
                    rel_lines=len(kg_result),
                )
            except Exception:
                self._update_stage_checkpoint(
                    checkpoint_state,
                    checkpoint_path,
                    topic_entry=topic_checkpoint,
                    stage="rel",
                    status=self.STAGE_FAILED,
                )
                raise

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
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(merged_kg, outfile, ensure_ascii=False, indent=4)
        if topic_checkpoint is not None:
            topic_checkpoint.setdefault("artifacts", {})["graph_file"] = os.path.abspath(output_path)
            topic_checkpoint["updated_at"] = self._now_iso()
            self._write_checkpoint_state(checkpoint_path, checkpoint_state)

        logger.info("Topic %s/%s completed: %s -> %s", idx, total_topics, topic, output_path)
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
        existing_kg=None,
        force_rebuild=False,
        on_topic_start: Callable[[int, int, str], None] | None = None,
        on_topic_success: Callable[[int, int, str, dict[str, Any]], None] | None = None,
        on_topic_failed: Callable[[int, int, str, Exception], None] | None = None,
        is_cancel_requested: Callable[[], bool] | None = None,
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

        checkpoint_state, checkpoint_path, checkpoint_loaded = self._prepare_checkpoint_state(
            output_dir=output_dir,
            topics=topics,
            ner_output_dir=ner_output_dir,
            rel_output_dir=rel_output_dir,
            sim_output_dir=sim_output_dir,
            graph_output_dir=graph_output_dir,
            force_rebuild=force_rebuild,
        )

        skip_construct = skip_construct or set()
        skip_ner_set = skip_ner_set or set()
        skip_sim_set = skip_sim_set or set()
        skip_rel_set = skip_rel_set or set()
        auto_resume = (not force_rebuild) and checkpoint_loaded
        self.set_disambiguation_config()
        self._log_runtime_configuration(
            total_topics=len(topics),
            existing_kg=existing_kg,
            force_rebuild=force_rebuild,
            auto_resume=auto_resume,
        )

        has_existing_kg = bool(existing_kg)
        global_kg = None
        if has_existing_kg:
            if isinstance(existing_kg, dict):
                global_kg = existing_kg.get("global", None)
            else:
                global_kg = existing_kg
        processed_count = 0
        failed_topics = []
        produced_graph_paths = []

        for idx, topic_data in enumerate(topics, start=1):
            if is_cancel_requested and is_cancel_requested():
                raise InterruptedError("Task canceled by user")

            if idx in skip_construct:
                continue

            cur_existing_kg = None
            if has_existing_kg:
                if global_kg:
                    cur_existing_kg = global_kg
                elif isinstance(existing_kg, dict):
                    cur_existing_kg = existing_kg.get(idx, None)
                else:
                    cur_existing_kg = existing_kg

            topic_name = topic_data.get("topic", f"topic_{idx}")
            if on_topic_start:
                on_topic_start(idx, len(topics), topic_name)

            if force_rebuild:
                skip_cur_ner = False
                skip_cur_sim = False
                skip_cur_rel = False
            else:
                skip_cur_ner = True if idx in skip_ner_set else None
                skip_cur_sim = True if idx in skip_sim_set else None
                skip_cur_rel = True if idx in skip_rel_set else None

            try:
                result = self.process(
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
                    checkpoint_state=checkpoint_state,
                    checkpoint_path=checkpoint_path,
                    auto_resume=auto_resume,
                )
                processed_count += 1
                produced_graph_paths.append(result["output_path"])
                if on_topic_success:
                    on_topic_success(idx, len(topics), topic_name, result)
            except Exception as e:
                logger.error(
                    "Failed topic index=%s topic=%s error=%s",
                    idx,
                    topic_name,
                    str(e),
                )
                debug_logger.error("Topic failure traceback:\n%s", traceback.format_exc())
                failed_topics.append(
                    {
                        "index": idx,
                        "topic": topic_name,
                        "error": str(e),
                    }
                )
                topic_checkpoint = checkpoint_state.get("topics", {}).get(str(idx))
                if isinstance(topic_checkpoint, dict):
                    for stage in self.STAGE_ORDER:
                        if topic_checkpoint.get("stages", {}).get(stage) == self.STAGE_RUNNING:
                            self._update_stage_checkpoint(
                                checkpoint_state,
                                checkpoint_path,
                                topic_entry=topic_checkpoint,
                                stage=stage,
                                status=self.STAGE_FAILED,
                            )
                            break
                if on_topic_failed:
                    on_topic_failed(idx, len(topics), topic_name, e)

        summary = {
            "total_topics": len(topics),
            "processed_topics": processed_count,
            "failed_topics_count": len(failed_topics),
            "failed_topics": failed_topics,
            "graph_paths": produced_graph_paths,
        }
        summary_path = os.path.join(output_dir, "process_summary.json")
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        logger.info("Processing summary saved to %s", summary_path)
        return summary
