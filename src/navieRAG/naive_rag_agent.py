import json
import logging
import os
import traceback
from typing import Dict, List, Union

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_provider import LLMProvider
from src.logger import get_logger
from src.textProcess import TextProcessor
from src.utils import parse_json_like_response

LOG_NAME_ENV_KEY = "RAKG_LOGGER_NAME"
DEFAULT_LOGGER_NAME = "AgentLog"

LOG_FILE_ENV_KEY = "RAKG_LOGGER_FILE"
DEFAULT_LOGGER_FILE = "Agent.log"


logger = get_logger(
    name=os.getenv(LOG_NAME_ENV_KEY, DEFAULT_LOGGER_NAME),
    level=logging.INFO,
    log_file=os.getenv(LOG_FILE_ENV_KEY, DEFAULT_LOGGER_FILE),
)

naive_rag_answer_prompt_cn = """
你是一个普通 RAG 问答助手。请仅根据给定的检索上下文回答问题。

问题：
{question}

检索上下文：
{context}

请严格输出 JSON（不要 markdown）：
{{
  "answer": "最终答案",
  "evidence_sources": [
    {{
      "source": "句子ID",
      "quote": "证据片段"
    }}
  ]
}}

约束：
1. 只能使用上下文中的信息，不得编造。
2. evidence_sources 至少给 1 条，source 必须来自上下文中的 sentence_id。
3. 如果证据不足，answer 必须明确写出“根据现有检索证据不足以得出确定结论”。
"""


class NaiveRAGAgent:
    """普通 RAG baseline：文本向量检索 + LLM 生成。"""

    def __init__(self):
        self.llm_provider = LLMProvider()
        self.model = self.llm_provider.get_llm()
        self.embeddings = self.llm_provider.get_embedding_model()

    def process(self, topic_data: Dict, idx: int, total_topics: int, rag_output_dir: str) -> Dict:
        """把单条 topic 处理成普通 RAG 可检索索引。"""
        topic = topic_data.get("topic")
        text = topic_data.get("content")
        if not topic or not text:
            raise ValueError("topic_data must contain non-empty 'topic' and 'content'.")

        logger.info("[NaiveRAG] Processing topic %s/%s: %s", idx, total_topics, topic)

        processor = TextProcessor(text=text, name=topic)
        text_split = processor.process()

        index_data = {
            "index": idx,
            "topic": topic,
            "sentences": text_split["sentences"],
            "vectors": text_split["vectors"],
            "sentence_to_id": text_split["sentence_to_id"],
            "id_to_sentence": text_split["id_to_sentence"],
        }

        output_path = os.path.join(rag_output_dir, f"{idx}.json")
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(index_data, outfile, ensure_ascii=False, indent=2)

        logger.info("[NaiveRAG] Saved retrieval index for topic %s to %s", topic, output_path)
        return {
            "index": idx,
            "topic": topic,
            "output_path": output_path,
            "index_data": index_data,
        }

    def process_all_topics(self, json_path: str, output_dir: str, done_offset: int = 0) -> Dict:
        """批量构建普通 RAG 索引（用于与 KG-RAG 对照实验）。"""
        with open(json_path, "r", encoding="utf-8") as file:
            topics = json.load(file)

        if not output_dir:
            raise ValueError("output_dir cannot be empty.")

        rag_output_dir = os.path.join(output_dir, "naive_rag_index")
        os.makedirs(rag_output_dir, exist_ok=True)

        processed_count = 0
        failed_topics = []

        for idx, topic_data in enumerate(topics, start=1):
            if idx <= done_offset:
                continue

            try:
                self.process(
                    topic_data=topic_data,
                    idx=idx,
                    total_topics=len(topics),
                    rag_output_dir=rag_output_dir,
                )
                processed_count += 1
            except Exception as e:
                logger.error(
                    "[NaiveRAG] Error building index for topic %s: %s",
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
        summary_path = os.path.join(output_dir, "naive_rag_process_summary.json")
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)

        logger.info("[NaiveRAG] Processing summary saved to %s", summary_path)
        return summary

    def _normalize_index_data(self, index_input: Union[str, Dict]) -> Dict:
        data = index_input
        if isinstance(index_input, str):
            with open(index_input, "r", encoding="utf-8") as f:
                data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("index_input must be a dict or a valid JSON file path.")

        required_fields = ["sentences", "vectors", "sentence_to_id"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"index_input missing required field: {field}")

        return data

    def retrieve(self, question: str, index_input: Union[str, Dict], top_k: int = 5) -> List[Dict]:
        """普通向量检索：返回 Top-K 文本证据。"""
        index_data = self._normalize_index_data(index_input)

        sentences = index_data.get("sentences", [])
        vectors = index_data.get("vectors", [])
        sentence_to_id = index_data.get("sentence_to_id", {})

        if not sentences or not vectors:
            return []

        query_vector = self.embeddings.embed_query(question)
        sentence_vectors = np.array(vectors)

        try:
            similarities = cosine_similarity([query_vector], sentence_vectors)[0]
        except Exception:
            logger.error("[NaiveRAG] Failed to compute retrieval similarity: %s", traceback.format_exc())
            return []

        top_k = max(1, min(top_k, len(sentences)))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved_items = []
        for rank, sent_idx in enumerate(top_indices, start=1):
            sentence = sentences[sent_idx]
            sentence_id = sentence_to_id.get(sentence, "")
            retrieved_items.append(
                {
                    "rank": rank,
                    "sentence": sentence,
                    "sentence_id": sentence_id,
                    "score": float(similarities[sent_idx]),
                }
            )
        return retrieved_items

    def _build_context(self, retrieved_items: List[Dict]) -> str:
        lines = []
        for idx, item in enumerate(retrieved_items, start=1):
            lines.append(
                f"{idx}. [sentence_id={item.get('sentence_id', '')}] "
                f"{item.get('sentence', '')} (score={item.get('score', 0):.4f})"
            )
        return "\n".join(lines)

    def _normalize_evidence_output(self, evidence_sources, fallback_items: List[Dict]) -> List[Dict]:
        if not isinstance(evidence_sources, list):
            evidence_sources = []

        normalized = []
        for item in evidence_sources:
            if isinstance(item, dict):
                source = str(item.get("source", "")).strip()
                quote = str(item.get("quote", "")).strip()
                if source or quote:
                    normalized.append({"source": source, "quote": quote})
            elif isinstance(item, str) and item.strip():
                normalized.append({"source": "", "quote": item.strip()})

        if normalized:
            return normalized

        fallback = []
        for item in fallback_items[:3]:
            fallback.append(
                {
                    "source": item.get("sentence_id", ""),
                    "quote": item.get("sentence", ""),
                }
            )
        return fallback

    def _format_answer(self, answer: str, evidence_sources: List[Dict]) -> str:
        answer_text = (
            answer
            if isinstance(answer, str) and answer.strip()
            else "根据现有检索证据不足以得出确定结论"
        )

        lines = [f"答案：{answer_text}", "证据来源："]
        if evidence_sources:
            for idx, item in enumerate(evidence_sources, start=1):
                lines.append(f"{idx}. [{item.get('source', '')}] {item.get('quote', '')}")
        else:
            lines.append("1. 无可用证据")
        return "\n".join(lines)

    def answer_question(
        self,
        question: str,
        index_input: Union[str, Dict],
        top_k: int = 5,
    ) -> Dict:
        """普通 RAG 问答：检索 -> 组装上下文 -> 生成答案。"""
        retrieved_items = self.retrieve(question=question, index_input=index_input, top_k=top_k)
        context_text = self._build_context(retrieved_items)

        prompt = ChatPromptTemplate.from_template(naive_rag_answer_prompt_cn)
        chain = prompt | self.model

        llm_result = chain.invoke(
            {
                "question": question,
                "context": context_text,
            }
        )

        parsed = parse_json_like_response(llm_result)
        if isinstance(parsed, dict):
            answer = parsed.get("answer", "")
            evidence_sources = self._normalize_evidence_output(
                parsed.get("evidence_sources", []),
                retrieved_items,
            )
        else:
            answer = llm_result.content if hasattr(llm_result, "content") else str(llm_result)
            evidence_sources = self._normalize_evidence_output([], retrieved_items)

        formatted_answer = self._format_answer(answer, evidence_sources)

        return {
            "question": question,
            "retrieval": {
                "top_k": top_k,
                "items": retrieved_items,
                "context_text": context_text,
            },
            "llm_output_raw": llm_result.content if hasattr(llm_result, "content") else str(llm_result),
            "answer": answer,
            "evidence_sources": evidence_sources,
            "formatted_answer": formatted_answer,
        }
