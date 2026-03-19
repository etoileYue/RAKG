"""NER抽取与实体中心关系抽取能力。"""

import json
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import extract_entiry_centric_kg_en_v2
from src.prompt import text2entity_en
from src.pipeline.shared import debug_logger
from src.pipeline.shared import logger

class PipelineRelationOpsMixin:
    """NER抽取与关系抽取方法集合。"""

    def extract_from_text_single(self, text_single, output_file):
        """调用LLM提取实体"""
        prompt = ChatPromptTemplate.from_template(text2entity_en)
        chain = prompt | self.model
        result = chain.invoke({"text": text_single})
        debug_logger.debug("-extract_from_text_single-")
        debug_logger.debug(f"text_single={text_single}, result={result}")

        if hasattr(result, "content"):
            result_json = json.loads(result.content)
        else:
            result_json = json.loads(result)

        combined_data = {"text": text_single, "entities": result_json}
        self._append_jsonl(output_file, combined_data)
        return result_json

    def extract_from_text_multiply(self, text_list, sent_to_id, output_file):
        """调用LLM对多文本提取实体"""
        ner_result_for_all = {}
        entity_num = 1
        for text in text_list:
            ner_result = self.extract_from_text_single(text, output_file)
            if "State" in ner_result:
                continue

            ner_result_num = len(ner_result)
            ner_result = self.rewrite(ner_result, entity_num)

            chunkid = sent_to_id.get(text)
            if chunkid is None:
                logger.warning("Sentence not found in sentence_to_id mapping, skipping chunk.")
                continue

            entity_num += ner_result_num
            ner_result_with_chunkid = self.add_chunkid(ner_result, chunkid)
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
        query = entity_dic[entity_id].get("name", "")
        context = self.get_retriever_context(query, sentences, sentence_to_id, vectors, top_k=5)
        retrieved_sentences = [item[0] for item in context]
        unique_sentences = self._dedupe_preserve_order(chunk_text_list + retrieved_sentences)
        chunk_text = ", ".join(unique_sentences)
        related_kg_payload = "none"
        if related_kg:
            related_kg_payload = json.dumps(related_kg, ensure_ascii=False)

        prompt = ChatPromptTemplate.from_template(extract_entiry_centric_kg_en_v2)
        chain = prompt | self.model
        result = chain.invoke(
            {
                "text": chunk_text,
                "target_entity": entity_dic[entity_id].get("name"),
                "related_kg": related_kg_payload,
            }
        )

        debug_logger.debug("-get_target_kg_single-")
        debug_logger.debug(
            "text=%s, target_entity=%s, related_kg=%s, result=%s",
            chunk_text,
            entity_dic[entity_id].get("name"),
            related_kg_payload,
            result,
        )

        if hasattr(result, "content"):
            result_json = json.loads(result.content)
        else:
            result_json = json.loads(result)

        combined_data = {"chunk_text": chunk_text, "entity": entity_dic[entity_id], "kg": result_json}
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
