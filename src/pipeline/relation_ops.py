"""NER抽取与实体中心关系抽取能力。"""

import json
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import extract_entiry_centric_kg_en_v2
from src.prompt import text2entity_en
from src.pipeline.shared import debug_logger
from src.pipeline.shared import logger

class PipelineRelationOpsMixin:
    """NER抽取与关系抽取方法集合。"""

    def _normalize_chunk_ids(self, chunk_ids):
        if chunk_ids is None:
            return []
        if isinstance(chunk_ids, list):
            values = chunk_ids
        else:
            text = str(chunk_ids).strip()
            if not text:
                return []
            values = text.split(";;;") if ";;;" in text else [text]
        normalized = []
        for item in values:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return self._dedupe_preserve_order(normalized)

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

    def extract_from_text_single(self, text_single, output_file):
        """调用LLM提取实体"""
        prompt = ChatPromptTemplate.from_template(text2entity_en)
        chain = prompt | self.model
        result = self._invoke_with_partial_fallback(chain, {"text": text_single})
        debug_logger.debug("-extract_from_text_single-")
        debug_logger.debug(f"text_single={text_single}, result={result}")

        raw_text = result.content if hasattr(result, "content") else str(result)
        try:
            result_json = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(
                "NER output is truncated/non-JSON. Returning fallback with raw text. len=%d",
                len(raw_text),
            )
            result_json = {"State": False, "_truncated_raw_text": raw_text}

        combined_data = {"text": text_single, "entities": result_json}
        self._append_jsonl(output_file, combined_data)
        return result_json

    def extract_from_text_multiply(self, text_list, sent_to_id, output_file):
        """调用LLM对多文本提取实体"""
        ner_result_for_all = {}
        entity_num = 1
        for text in text_list:
            chunkid = sent_to_id.get(text)
            if chunkid is None:
                logger.warning("Sentence not found in sentence_to_id mapping, skipping chunk.")
                continue

            ner_result = self.extract_from_text_single(text, output_file)
            if "State" in ner_result:
                continue

            ner_result_num = len(ner_result)
            ner_result = self.rewrite(ner_result, entity_num)

            entity_num += ner_result_num
            ner_result_with_chunkid = self.add_chunkid(ner_result, [chunkid])
            ner_result_for_all.update(ner_result_with_chunkid)
        return ner_result_for_all

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
        chunk_text_list = self.get_sentences_for_entity(entity_dic, entity_id, id_to_sentence)
        entity_chunk_ids = self._normalize_chunk_ids(entity_dic[entity_id].get("chunkid", []))
        query = entity_dic[entity_id].get("name", "")
        context = self.get_retriever_context(query, sentences, sentence_to_id, vectors, top_k=5)
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
        chunk_text = ", ".join(unique_sentences)
        related_kg_payload = "none"
        if related_kg:
            related_kg_payload = json.dumps(related_kg, ensure_ascii=False)

        prompt = ChatPromptTemplate.from_template(extract_entiry_centric_kg_en_v2)
        chain = prompt | self.model
        result = self._invoke_with_partial_fallback(
            chain,
            {
                "text": chunk_text,
                "target_entity": entity_dic[entity_id].get("name"),
                "related_kg": related_kg_payload,
            },
        )

        debug_logger.debug("-get_target_kg_single-")
        debug_logger.debug(
            "text=%s, target_entity=%s, related_kg=%s, result=%s",
            chunk_text,
            entity_dic[entity_id].get("name"),
            related_kg_payload,
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
                    "name": entity_dic[entity_id].get("name", ""),
                    "type": entity_dic[entity_id].get("type", "Unknown"),
                    "description": "",
                    "attributes": [],
                    "relationships": [],
                    "provenance": {"chunk_ids": entity_chunk_ids},
                },
                "_truncated_raw_text": raw_text,
            }

        if isinstance(result_json, dict):
            provenance_meta = result_json.get("_provenance", {})
            if not isinstance(provenance_meta, dict):
                provenance_meta = {}
            provenance_meta["entity_chunk_ids"] = entity_chunk_ids
            provenance_meta["candidate_chunk_ids"] = candidate_chunk_ids
            provenance_meta["candidate_chunks"] = candidate_chunks
            result_json["_provenance"] = provenance_meta

            central_entity = result_json.get("central_entity", {})
            if isinstance(central_entity, dict):
                central_prov = central_entity.get("provenance", {})
                if not isinstance(central_prov, dict):
                    central_prov = {}
                central_ids = self._normalize_chunk_ids(central_prov.get("chunk_ids", []))
                central_prov["chunk_ids"] = self._dedupe_preserve_order(central_ids + entity_chunk_ids)
                central_entity["provenance"] = central_prov

        combined_data = {
            "chunk_text": chunk_text,
            "entity": entity_dic[entity_id],
            "kg": result_json,
            "candidate_chunk_ids": candidate_chunk_ids,
            "candidate_chunks": candidate_chunks,
        }
        self._append_jsonl(output_file, combined_data)
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
        results = {}
        related_kg_map = related_kg_map or {}
        for entity_id in entity_dic:
            result = self.get_target_kg_single(
                entity_dic,
                entity_id,
                id_to_sentence,
                sentences,
                sentence_to_id,
                vectors,
                output_file,
                related_kg=related_kg_map.get(entity_id),
            )
            results[entity_id] = result
        return results
