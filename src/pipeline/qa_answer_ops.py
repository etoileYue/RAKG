"""QA answer generation and output normalization."""

from langchain_core.prompts import ChatPromptTemplate

from src.prompt import get_prompt
from src.utils import parse_json_like_response


class PipelineQAAnswerOpsMixin:
    """Answer generation and formatting for graph-based QA."""

    def _normalize_evidence_output(self, evidence_sources, fallback_evidence):
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
        for item in fallback_evidence[:3]:
            fallback.append({"source": item.get("source", ""), "quote": item.get("text", "")})
        return fallback

    def _format_qa_output(self, answer, evidence_sources, graph_paths):
        answer_text = (
            answer
            if isinstance(answer, str) and answer.strip()
            else "根据现有图谱证据不足以得出确定结论"
        )
        lines = [f"答案：{answer_text}", "证据来源："]
        if evidence_sources:
            for idx, item in enumerate(evidence_sources, start=1):
                source = item.get("source", "")
                quote = item.get("quote", "")
                lines.append(f"{idx}. [{source}] {quote}")
        else:
            lines.append("1. 无可用证据")

        lines.append("图谱路径：")
        if graph_paths:
            for idx, path in enumerate(graph_paths, start=1):
                lines.append(f"{idx}. {path}")
        else:
            lines.append("1. 无可用路径")
        return "\n".join(lines)

    def answer_question_with_kg(
        self,
        question,
        knowledge_graph=None,
        max_hop=2,
        seed_top_k=5,
        max_context_items=30,
        cache_key=None,
        graph_index=None,
    ):
        retrieval = self.build_qa_retrieval_context(
            question=question,
            knowledge_graph=knowledge_graph,
            max_hop=max_hop,
            seed_top_k=seed_top_k,
            max_context_items=max_context_items,
            cache_key=cache_key,
            graph_index=graph_index,
        )

        graph_paths_text = "\n".join([f"- {path}" for path in retrieval["graph_paths"][:10]])
        prompt = ChatPromptTemplate.from_template(get_prompt("qa_answer"))
        chain = prompt | self.model
        llm_result = chain.invoke(
            {
                "question": question,
                "context": retrieval["context_text"],
                "graph_paths": graph_paths_text,
            }
        )

        parsed = parse_json_like_response(llm_result)
        if isinstance(parsed, dict):
            answer = parsed.get("answer", "")
            evidence_sources = self._normalize_evidence_output(
                parsed.get("evidence_sources", []),
                retrieval["evidence_items"],
            )
            graph_paths = parsed.get("graph_paths", [])
            if not isinstance(graph_paths, list):
                graph_paths = []
        else:
            answer = llm_result.content if hasattr(llm_result, "content") else str(llm_result)
            evidence_sources = self._normalize_evidence_output([], retrieval["evidence_items"])
            graph_paths = retrieval["graph_paths"][:3]

        if not graph_paths:
            graph_paths = retrieval["graph_paths"][:3]

        formatted_answer = self._format_qa_output(answer, evidence_sources, graph_paths)
        return {
            "question": question,
            "retrieval": retrieval,
            "llm_output_raw": llm_result.content if hasattr(llm_result, "content") else str(llm_result),
            "answer": answer,
            "evidence_sources": evidence_sources,
            "graph_paths": graph_paths,
            "formatted_answer": formatted_answer,
        }
