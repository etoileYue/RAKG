"""I/O and small utility adapters used by the pipeline."""

from src.utils import add_chunkid_to_entities
from src.utils import append_jsonl
from src.utils import dedupe_preserve_order
from src.utils import ensure_parent_dir
from src.utils import renumber_entities

class PipelineIOOpsMixin:
    """I/O and small utility adapters used by the pipeline."""

    def _ensure_parent_dir(self, output_file):
        ensure_parent_dir(output_file)

    def _append_jsonl(self, output_file, data):
        append_jsonl(output_file, data)

    def _dedupe_preserve_order(self, items):
        return dedupe_preserve_order(items)

    def _is_boundary_candidate(self, similarity_score, threshold, gray_margin):
        return similarity_score <= (threshold + gray_margin)

    def add_chunkid(self, ner_result, chunkid):
        return add_chunkid_to_entities(ner_result, chunkid)

    def rewrite(self, ner_result, entity_num):
        return renumber_entities(ner_result, entity_num)
