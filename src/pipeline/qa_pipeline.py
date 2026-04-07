"""Composable QA pipeline entry that keeps backward compatibility."""

from src.pipeline.qa_answer_ops import PipelineQAAnswerOpsMixin
from src.pipeline.qa_context_ops import PipelineQAContextOpsMixin
from src.pipeline.qa_graph_ops import PipelineQAGraphOpsMixin
from src.pipeline.qa_match_ops import PipelineQAMatchOpsMixin
from src.utils import parse_json_like_response


class KnowledgeGraphQA(
    PipelineQAGraphOpsMixin,
    PipelineQAMatchOpsMixin,
    PipelineQAContextOpsMixin,
    PipelineQAAnswerOpsMixin,
):
    """Backward-compatible QA class composed from split QA mixins."""

    def _parse_json_like_response(self, response):
        return parse_json_like_response(response)
