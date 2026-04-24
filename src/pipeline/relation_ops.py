"""NER抽取与实体中心关系抽取能力。"""

import json
import os
from langchain_core.prompts import ChatPromptTemplate
from src.llm_executor import LLMExecutor
from src.llm_executor import LLMTask
from src.llm_executor import LLMTaskError
from src.prompt import get_prompt
from src.pipeline.shared import debug_logger
from src.pipeline.shared import logger
from src.utils import normalize_chunk_ids

class PipelineRelationOpsMixin:
    """NER抽取与关系抽取方法集合。"""

    _PRONOUN_HINTS_EN = {
        "he", "she", "it", "they", "them", "his", "her", "hers", "their", "theirs",
        "him", "this", "that", "these", "those", "its", "we", "our", "ours", "you", "your",
    }
    _PRONOUN_HINTS_ZH = {"他", "她", "它", "他们", "她们", "它们", "其", "该", "这", "那", "此"}

    @staticmethod
    def _is_feature_enabled(env_key, default=True):
        value = os.getenv(env_key)
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _normalize_chunk_ids(self, chunk_ids):
        return normalize_chunk_ids(chunk_ids)

    @staticmethod
    def _is_length_limit_error(exc):
        text = str(exc)
        return (
            "LengthFinishReasonError" in text
            or "length limit was reached" in text.lower()
            or exc.__class__.__name__ == "LengthFinishReasonError"
        )

    @staticmethod
    def _extract_partial_text_from_exception(exc):
        completion = getattr(exc, "completion", None)
        if completion and getattr(completion, "choices", None):
            try:
                partial = completion.choices[0].message.content
                if partial is not None:
                    return str(partial)
            except Exception:
                return None
        return None

    def _invoke_with_partial_fallback(self, chain, payload):
        """Invoke LLM and return partial text if response is truncated by length."""
        try:
            return chain.invoke(payload)
        except Exception as exc:
            if not self._is_length_limit_error(exc):
                raise
            partial_text = self._extract_partial_text_from_exception(exc)
            if partial_text:
                logger.warning(
                    "Captured truncated LLM content from LengthFinishReasonError (chars=%d).",
                    len(partial_text),
                )
                return partial_text
            raise

    def _get_llm_executor(self):
        executor = getattr(self, "_llm_executor", None)
        if executor is None:
            executor = LLMExecutor()
            self._llm_executor = executor
        return executor

    @staticmethod
    def _require_task_result(task_result):
        if isinstance(task_result, LLMTaskError):
            logger.error(
                "LLM task failed: kind=%s metadata=%s error=%s",
                task_result.kind,
                task_result.metadata,
                task_result.message,
            )
            debug_logger.error("LLM task traceback:\n%s", task_result.traceback_text)
            task_result.reraise()
        return task_result

    def _invoke_text2entity_payload(self, payload):
        prompt = ChatPromptTemplate.from_template(get_prompt("text2entity"))
        chain = prompt | self.model
        return self._invoke_with_partial_fallback(chain, payload)

    def _parse_ner_response(self, text_single, result):
        debug_logger.debug("-extract_from_text_single-")
        debug_logger.debug("text_single=%s, result=%s", text_single, result)

        raw_text = result.content if hasattr(result, "content") else str(result)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(
                "NER output is truncated/non-JSON. Returning fallback with raw text. len=%d",
                len(raw_text),
            )
            return {"State": False, "_truncated_raw_text": raw_text}

    def _append_ner_record(self, output_file, text, entities):
        self._append_jsonl(output_file, {"text": text, "entities": entities})

    def _build_ner_task(self, text_single, *, metadata=None):
        return LLMTask(
            kind="ner",
            payload={"text": text_single},
            invoke_fn=self._invoke_text2entity_payload,
            parser=lambda result, text=text_single: self._parse_ner_response(text, result),
            metadata=metadata or {},
        )

    def extract_from_text_single(self, text_single, output_file):
        """调用LLM提取实体"""
        task = self._build_ner_task(text_single, metadata={"phase": "single"})
        result_json = self._require_task_result(self._get_llm_executor().invoke(task))
        self._append_ner_record(output_file, text_single, result_json)
        return result_json

    def extract_from_text_multiply(self, text_list, sent_to_id, output_file):
        """调用LLM对多文本提取实体"""
        executor = self._get_llm_executor()
        ner_result_for_all = {}
        entity_num = 1
        primary_specs = []
        patch_specs = []

        for idx, text in enumerate(text_list):
            chunkid = sent_to_id.get(text)
            if chunkid is None:
                logger.warning("Sentence not found in sentence_to_id mapping, skipping chunk.")
                continue

            spec = {
                "idx": idx,
                "text": text,
                "chunk_ids_for_entity": [chunkid],
            }
            primary_specs.append(spec)

            if self._is_feature_enabled("RAKG_ENABLE_NER_WINDOW_PATCH", True) and self._should_enable_ner_window_patch(text):
                window_text, window_chunk_ids = self._build_ner_window_patch_context(
                    text_list=text_list,
                    sent_to_id=sent_to_id,
                    center_index=idx,
                    window_size=1,
                )
                spec["chunk_ids_for_entity"] = self._dedupe_preserve_order(spec["chunk_ids_for_entity"] + window_chunk_ids)
                if window_text and window_text != text:
                    patch_specs.append(
                        {
                            "idx": idx,
                            "text": window_text,
                        }
                    )

        primary_results = executor.invoke_batch(
            [
                self._build_ner_task(
                    spec["text"],
                    metadata={"phase": "primary", "index": spec["idx"]},
                )
                for spec in primary_specs
            ]
        )
        patch_results = executor.invoke_batch(
            [
                self._build_ner_task(
                    spec["text"],
                    metadata={"phase": "window_patch", "index": spec["idx"]},
                )
                for spec in patch_specs
            ]
        )
        patch_results_by_index = {}
        patch_text_by_index = {}
        for spec, task_result in zip(patch_specs, patch_results):
            patch_text_by_index[spec["idx"]] = spec["text"]
            patch_results_by_index[spec["idx"]] = self._require_task_result(task_result)

        for spec, task_result in zip(primary_specs, primary_results):
            ner_result = self._require_task_result(task_result)
            self._append_ner_record(output_file, spec["text"], ner_result)
            merged_ner_result = {}
            if isinstance(ner_result, dict) and "State" not in ner_result:
                merged_ner_result = dict(ner_result)

            patched_ner_result = patch_results_by_index.get(spec["idx"])
            if patched_ner_result is not None:
                self._append_ner_record(output_file, patch_text_by_index[spec["idx"]], patched_ner_result)
                merged_ner_result = self._merge_ner_results(merged_ner_result, patched_ner_result)

            if not merged_ner_result:
                continue

            ner_result_num = len(merged_ner_result)
            ner_result = self.rewrite(merged_ner_result, entity_num)

            entity_num += ner_result_num
            ner_result_with_chunkid = self.add_chunkid(ner_result, spec["chunk_ids_for_entity"])
            ner_result_for_all.update(ner_result_with_chunkid)
        return ner_result_for_all

    def _should_enable_ner_window_patch(self, text):
        text_str = str(text or "").strip()
        if not text_str:
            return False

        lowered = text_str.lower()
        tokens = [token for token in lowered.replace(",", " ").replace(".", " ").split() if token]
        pronoun_hits_en = sum(1 for token in tokens if token in self._PRONOUN_HINTS_EN)
        pronoun_hits_zh = sum(text_str.count(item) for item in self._PRONOUN_HINTS_ZH)
        is_short_sentence = len(text_str) <= 60 or len(tokens) <= 8
        return (pronoun_hits_en + pronoun_hits_zh) >= 1 and is_short_sentence

    def _build_ner_window_patch_context(self, text_list, sent_to_id, center_index, window_size=1):
        start = max(0, center_index - window_size)
        end = min(len(text_list), center_index + window_size + 1)
        window_texts = []
        window_chunk_ids = []
        for idx in range(start, end):
            sentence = text_list[idx]
            if isinstance(sentence, str) and sentence.strip():
                window_texts.append(sentence.strip())
            chunk_id = sent_to_id.get(sentence)
            if chunk_id is not None:
                window_chunk_ids.append(str(chunk_id).strip())
        return " ".join(window_texts).strip(), self._normalize_chunk_ids(window_chunk_ids)

    def _merge_ner_results(self, left_result, right_result):
        merged = {}
        seen_keys = set()

        def add_items(raw_result):
            if not isinstance(raw_result, dict) or "State" in raw_result:
                return
            for entity in raw_result.values():
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("name", "")).strip()
                entity_type = str(entity.get("type", "Unknown")).strip()
                description = str(entity.get("description", "")).strip()
                if not name:
                    continue
                dedupe_key = (name.lower(), entity_type.lower(), description.lower())
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                merged[f"entity{len(merged) + 1}"] = {
                    "name": name,
                    "type": entity_type or "Unknown",
                    "description": description,
                }

        add_items(left_result)
        add_items(right_result)
        return merged

    def _resolve_entity_terms(self, entity):
        if not isinstance(entity, dict):
            return []
        name = str(entity.get("name", "")).strip()
        aliases = entity.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = [aliases]
        terms = [name] if name else []
        for alias in aliases:
            alias_text = str(alias).strip()
            if alias_text:
                terms.append(alias_text)
        return self._dedupe_preserve_order(terms)

    def _compose_context_block_text(self, evidence_blocks):
        lines = []
        for block in evidence_blocks:
            chunk_id = block.get("chunk_id", "")
            source = block.get("source", "unknown")
            similarity = block.get("similarity")
            rank = block.get("rank")
            sentence = str(block.get("text", "")).strip()
            similarity_text = f"{float(similarity):.4f}" if isinstance(similarity, (int, float)) else "NA"
            rank_text = str(rank) if rank is not None else "NA"
            lines.append(
                f"[chunk_id={chunk_id}; source={source}; similarity={similarity_text}; rank={rank_text}]"
            )
            lines.append(sentence)
            lines.append("")
        return "\n".join(lines).strip()

    def _build_relation_context(
        self,
        entity,
        entity_chunk_ids,
        id_to_sentence,
        sentences,
        sentence_to_id,
        vectors,
        top_k=5,
        min_similarity=0.2,
        dedupe_threshold=0.97,
        neighbor_window=2,
    ):
        query = str((entity or {}).get("name", "")).strip()
        entity_terms = self._resolve_entity_terms(entity or {})
        retriever_context = self.get_retriever_context(
            query=query,
            sentences=sentences,
            sentence_to_id=sentence_to_id,
            vectors=vectors,
            top_k=top_k,
            min_similarity=min_similarity,
            dedupe_threshold=dedupe_threshold,
            use_mmr=True,
            entity_terms=entity_terms,
            entity_description=(entity or {}).get("description", ""),
            entity_constraint_min_similarity=max(min_similarity - 0.05, 0.1),
        )

        ordered_chunk_ids = [str(chunk_id).strip() for chunk_id in id_to_sentence.keys()]
        order_index = {chunk_id: idx for idx, chunk_id in enumerate(ordered_chunk_ids)}
        source_meta = {}

        for chunk_id in entity_chunk_ids:
            if chunk_id not in source_meta:
                source_meta[chunk_id] = {"source_tags": set(), "similarity": None, "rank": None}
            source_meta[chunk_id]["source_tags"].add("entity_chunk")

        retrieved_chunk_ids = []
        for rank, item in enumerate(retriever_context, start=1):
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            sentence, similarity, chunk_id = item[0], item[1], item[2]
            chunk_id = str(chunk_id).strip()
            if not chunk_id:
                continue
            retrieved_chunk_ids.append(chunk_id)
            if chunk_id not in source_meta:
                source_meta[chunk_id] = {"source_tags": set(), "similarity": None, "rank": None}
            source_meta[chunk_id]["source_tags"].add("retrieval")
            source_meta[chunk_id]["similarity"] = float(similarity)
            source_meta[chunk_id]["rank"] = rank

        neighborhood_chunk_ids = []
        seeds = self._normalize_chunk_ids(entity_chunk_ids + retrieved_chunk_ids)
        for chunk_id in seeds:
            center_idx = order_index.get(chunk_id)
            if center_idx is None:
                continue
            for offset in range(-neighbor_window, neighbor_window + 1):
                if offset == 0:
                    continue
                neighbor_idx = center_idx + offset
                if 0 <= neighbor_idx < len(ordered_chunk_ids):
                    neighbor_id = ordered_chunk_ids[neighbor_idx]
                    neighborhood_chunk_ids.append(neighbor_id)
                    if neighbor_id not in source_meta:
                        source_meta[neighbor_id] = {"source_tags": set(), "similarity": None, "rank": None}
                    source_meta[neighbor_id]["source_tags"].add("neighbor")

        candidate_chunk_ids = self._normalize_chunk_ids(entity_chunk_ids + retrieved_chunk_ids + neighborhood_chunk_ids)
        candidate_chunk_id_set = set(candidate_chunk_ids)
        ordered_chunk_id_set = set(ordered_chunk_ids)
        known_ids = [chunk_id for chunk_id in ordered_chunk_ids if chunk_id in candidate_chunk_id_set]
        unknown_ids = [chunk_id for chunk_id in candidate_chunk_ids if chunk_id not in ordered_chunk_id_set]
        ordered_candidate_chunk_ids = known_ids + unknown_ids

        candidate_chunks = {}
        evidence_blocks = []
        for chunk_id in ordered_candidate_chunk_ids:
            sentence = id_to_sentence.get(chunk_id)
            if not isinstance(sentence, str) or not sentence.strip():
                continue
            candidate_chunks[chunk_id] = sentence.strip()
            meta = source_meta.get(chunk_id, {"source_tags": {"context"}, "similarity": None, "rank": None})
            source_tags = sorted([tag for tag in meta.get("source_tags", set()) if tag])
            evidence_blocks.append(
                {
                    "chunk_id": chunk_id,
                    "source": "+".join(source_tags) if source_tags else "context",
                    "similarity": meta.get("similarity"),
                    "rank": meta.get("rank"),
                    "text": sentence.strip(),
                }
            )

        return {
            "retriever_context": retriever_context,
            "candidate_chunk_ids": ordered_candidate_chunk_ids,
            "candidate_chunks": candidate_chunks,
            "evidence_blocks": evidence_blocks,
            "chunk_text": self._compose_context_block_text(evidence_blocks),
        }

    def _build_relation_context_legacy(
        self,
        entity,
        entity_chunk_ids,
        id_to_sentence,
        sentences,
        sentence_to_id,
        vectors,
    ):
        chunk_text_list = self.get_sentences_for_entity(
            entity_dic={"_legacy_entity": entity},
            entity_id="_legacy_entity",
            id_to_sentence=id_to_sentence,
        )
        query = str((entity or {}).get("name", "")).strip()
        context = self.get_retriever_context(
            query=query,
            sentences=sentences,
            sentence_to_id=sentence_to_id,
            vectors=vectors,
            top_k=5,
            min_similarity=0.0,
            dedupe_threshold=1.0,
            use_mmr=False,
            entity_terms=[],
            entity_description="",
            entity_constraint_min_similarity=1.0,
        )
        retrieved_sentences = [item[0] for item in context]
        retrieved_chunk_ids = self._normalize_chunk_ids(
            [item[2] for item in context if isinstance(item, (list, tuple)) and len(item) >= 3]
        )
        candidate_chunk_ids = self._dedupe_preserve_order(entity_chunk_ids + retrieved_chunk_ids)
        candidate_chunks = {}
        for chunk_id in candidate_chunk_ids:
            sentence = id_to_sentence.get(chunk_id)
            if isinstance(sentence, str) and sentence.strip():
                candidate_chunks[chunk_id] = sentence.strip()
        unique_sentences = self._dedupe_preserve_order(chunk_text_list + retrieved_sentences)
        return {
            "candidate_chunk_ids": candidate_chunk_ids,
            "candidate_chunks": candidate_chunks,
            "evidence_blocks": [],
            "chunk_text": ", ".join(unique_sentences),
        }

    def _prepare_relation_request(
        self,
        entity_dic,
        entity_id,
        id_to_sentence,
        sentences,
        sentence_to_id,
        vectors,
        related_kg=None,
    ):
        entity_chunk_ids = self._normalize_chunk_ids(entity_dic[entity_id].get("chunkid", []))
        if self._is_feature_enabled("RAKG_ENABLE_EVIDENCE_CONTEXT", True):
            relation_context = self._build_relation_context(
                entity=entity_dic[entity_id],
                entity_chunk_ids=entity_chunk_ids,
                id_to_sentence=id_to_sentence,
                sentences=sentences,
                sentence_to_id=sentence_to_id,
                vectors=vectors,
                top_k=5,
                min_similarity=0.2,
                dedupe_threshold=0.97,
                neighbor_window=2,
            )
        else:
            relation_context = self._build_relation_context_legacy(
                entity=entity_dic[entity_id],
                entity_chunk_ids=entity_chunk_ids,
                id_to_sentence=id_to_sentence,
                sentences=sentences,
                sentence_to_id=sentence_to_id,
                vectors=vectors,
            )

        related_kg_payload = "none"
        if related_kg:
            related_kg_payload = json.dumps(related_kg, ensure_ascii=False)

        return {
            "entity_id": entity_id,
            "entity": entity_dic[entity_id],
            "entity_chunk_ids": entity_chunk_ids,
            "candidate_chunk_ids": relation_context.get("candidate_chunk_ids", []),
            "candidate_chunks": relation_context.get("candidate_chunks", {}),
            "evidence_blocks": relation_context.get("evidence_blocks", []),
            "chunk_text": relation_context.get("chunk_text", ""),
            "payload": {
                "text": relation_context.get("chunk_text", ""),
                "target_entity": entity_dic[entity_id].get("name"),
                "related_kg": related_kg_payload,
            },
            "related_kg_payload": related_kg_payload,
        }

    def _invoke_entity_centric_kg_payload(self, payload):
        prompt = ChatPromptTemplate.from_template(get_prompt("entity_centric_kg"))
        chain = prompt | self.model
        return self._invoke_with_partial_fallback(chain, payload)

    def _parse_relation_response(self, request, result):
        debug_logger.debug("-get_target_kg_single-")
        debug_logger.debug(
            "text=%s, target_entity=%s, related_kg=%s, result=%s",
            request["chunk_text"],
            request["entity"].get("name"),
            request["related_kg_payload"],
            result,
        )

        raw_text = result.content if hasattr(result, "content") else str(result)
        try:
            result_json = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(
                "KG output is truncated/non-JSON. Returning fallback with raw text. len=%d",
                len(raw_text),
            )
            result_json = {
                "central_entity": {
                    "name": request["entity"].get("name", ""),
                    "type": request["entity"].get("type", "Unknown"),
                    "description": "",
                    "attributes": [],
                    "relationships": [],
                    "provenance": {"chunk_ids": request["entity_chunk_ids"]},
                },
                "_truncated_raw_text": raw_text,
            }

        if isinstance(result_json, dict):
            provenance_meta = result_json.get("_provenance", {})
            if not isinstance(provenance_meta, dict):
                provenance_meta = {}
            provenance_meta["entity_chunk_ids"] = request["entity_chunk_ids"]
            provenance_meta["candidate_chunk_ids"] = request["candidate_chunk_ids"]
            provenance_meta["candidate_chunks"] = request["candidate_chunks"]
            provenance_meta["evidence_blocks"] = request["evidence_blocks"]
            result_json["_provenance"] = provenance_meta

            central_entity = result_json.get("central_entity", {})
            if isinstance(central_entity, dict):
                central_prov = central_entity.get("provenance", {})
                if not isinstance(central_prov, dict):
                    central_prov = {}
                central_ids = self._normalize_chunk_ids(central_prov.get("chunk_ids", []))
                central_prov["chunk_ids"] = self._dedupe_preserve_order(central_ids + request["entity_chunk_ids"])
                central_entity["provenance"] = central_prov

        return result_json

    def _append_relation_record(self, output_file, request, result_json):
        self._append_jsonl(
            output_file,
            {
                "chunk_text": request["chunk_text"],
                "entity": request["entity"],
                "kg": result_json,
                "candidate_chunk_ids": request["candidate_chunk_ids"],
                "candidate_chunks": request["candidate_chunks"],
            },
        )

    def _build_relation_task(self, request):
        return LLMTask(
            kind="relation",
            payload=request["payload"],
            invoke_fn=self._invoke_entity_centric_kg_payload,
            parser=lambda result, relation_request=request: self._parse_relation_response(relation_request, result),
            metadata={"entity_id": request["entity_id"], "target_entity": request["entity"].get("name")},
        )

    def get_target_kg_single(
        self,
        entity_dic,
        entity_id,
        id_to_sentence,
        sentences,
        sentence_to_id,
        vectors,
        output_file,
        related_kg=None,
    ):
        """调用LLM提取关系"""
        request = self._prepare_relation_request(
            entity_dic,
            entity_id,
            id_to_sentence,
            sentences,
            sentence_to_id,
            vectors,
            related_kg=related_kg,
        )
        result_json = self._require_task_result(
            self._get_llm_executor().invoke(self._build_relation_task(request))
        )
        self._append_relation_record(output_file, request, result_json)
        return result_json

    def get_target_kg_all(
        self,
        entity_dic,
        id_to_sentence,
        sentences,
        sentence_to_id,
        vectors,
        output_file,
        related_kg_map=None,
    ):
        """调用LLM对多文本提取关系"""
        executor = self._get_llm_executor()
        results = {}
        related_kg_map = related_kg_map or {}
        requests = []
        for entity_id in entity_dic:
            requests.append(
                self._prepare_relation_request(
                    entity_dic,
                    entity_id,
                    id_to_sentence,
                    sentences,
                    sentence_to_id,
                    vectors,
                    related_kg=related_kg_map.get(entity_id),
                )
            )

        batch_results = executor.invoke_batch(
            [self._build_relation_task(request) for request in requests],
            progress_label="REL",
            progress_total=len(requests),
            progress_enabled=bool(requests),
        )
        for request, task_result in zip(requests, batch_results):
            result = self._require_task_result(task_result)
            self._append_relation_record(output_file, request, result)
            results[request["entity_id"]] = result
        return results
