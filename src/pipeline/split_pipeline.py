"""Composable split pipeline class built from mixins under src.pipeline."""

from src.pipeline.entity_ops import PipelineEntityOpsMixin
from src.pipeline.graph_ops import PipelineGraphOpsMixin
from src.pipeline.io_ops import PipelineIOOpsMixin
from src.pipeline.relation_ops import PipelineRelationOpsMixin
from src.pipeline.retrieval_ops import PipelineRetrievalOpsMixin
from src.pipeline.similarity_ops import PipelineSimilarityOpsMixin


class SplitNERPipeline(
    PipelineIOOpsMixin,
    PipelineEntityOpsMixin,
    PipelineSimilarityOpsMixin,
    PipelineRetrievalOpsMixin,
    PipelineRelationOpsMixin,
    PipelineGraphOpsMixin,
):
    """Split-by-responsibility pipeline composition.

    This class mirrors methods from NERPipeline but is not wired into callers yet.
    """

    pass
