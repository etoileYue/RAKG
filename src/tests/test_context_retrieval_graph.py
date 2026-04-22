import unittest

from src.pipeline.entity_ops import PipelineEntityOpsMixin
from src.pipeline.graph_ops import PipelineGraphOpsMixin
from src.pipeline.io_ops import PipelineIOOpsMixin
from src.pipeline.relation_ops import PipelineRelationOpsMixin
from src.pipeline.retrieval_ops import PipelineRetrievalOpsMixin


class DummyEmbeddings:
    def embed_query(self, text):
        lowered = str(text or "").lower()
        if "alpha" in lowered or "alice" in lowered:
            return [1.0, 0.0]
        return [0.0, 1.0]


class DummyRelationPipeline(PipelineIOOpsMixin, PipelineRetrievalOpsMixin, PipelineRelationOpsMixin):
    def __init__(self):
        self.embeddings = DummyEmbeddings()


class DummyGraphPipeline(PipelineIOOpsMixin, PipelineEntityOpsMixin, PipelineGraphOpsMixin):
    pass


class ContextRetrievalGraphTests(unittest.TestCase):
    def setUp(self):
        self.relation_pipeline = DummyRelationPipeline()
        self.graph_pipeline = DummyGraphPipeline()

    def test_retriever_context_filters_low_similarity_and_redundancy(self):
        sentences = [
            "Alpha founded the company.",
            "Alpha established the company.",
            "Completely unrelated weather report.",
            "This sentence discusses founders without naming anyone.",
        ]
        sentence_to_id = {
            sentences[0]: "c1",
            sentences[1]: "c2",
            sentences[2]: "c3",
            sentences[3]: "c4",
        }
        vectors = [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.95, 0.05],
        ]

        context = self.relation_pipeline.get_retriever_context(
            query="Alpha",
            sentences=sentences,
            sentence_to_id=sentence_to_id,
            vectors=vectors,
            top_k=3,
            min_similarity=0.3,
            dedupe_threshold=0.995,
            entity_terms=["Alpha"],
            entity_constraint_min_similarity=1.01,
        )

        self.assertEqual(len(context), 1)
        self.assertEqual(context[0][2], "c1")

    def test_relation_context_builder_orders_by_document_and_expands_neighbors(self):
        id_to_sentence = {
            "d1": "Background sentence.",
            "d2": "Another background sentence.",
            "d3": "Alpha appears here.",
            "d4": "Transition sentence.",
            "d5": "Alpha wins the event.",
            "d6": "Closing sentence.",
        }
        sentences = list(id_to_sentence.values())
        sentence_to_id = {sentence: chunk_id for chunk_id, sentence in id_to_sentence.items()}
        vectors = [
            [0.05, 0.95],
            [0.1, 0.9],
            [0.82, 0.18],
            [0.2, 0.8],
            [0.99, 0.01],
            [0.1, 0.9],
        ]
        entity = {"name": "Alpha", "aliases": ["A"], "description": "", "chunkid": ["d3"]}

        relation_context = self.relation_pipeline._build_relation_context(
            entity=entity,
            entity_chunk_ids=["d3"],
            id_to_sentence=id_to_sentence,
            sentences=sentences,
            sentence_to_id=sentence_to_id,
            vectors=vectors,
            top_k=2,
            min_similarity=0.2,
            dedupe_threshold=0.99,
            neighbor_window=2,
        )

        self.assertEqual(
            relation_context["candidate_chunk_ids"],
            ["d1", "d2", "d3", "d4", "d5", "d6"],
        )

        block_by_chunk = {item["chunk_id"]: item for item in relation_context["evidence_blocks"]}
        self.assertIn("entity_chunk", block_by_chunk["d3"]["source"])
        self.assertIn("retrieval", block_by_chunk["d5"]["source"])

    def test_convert_knowledge_graph_prefers_llm_relation_provenance(self):
        input_data = {
            "entity1": {
                "central_entity": {
                    "name": "Alice",
                    "type": "Person",
                    "description": "",
                    "attributes": [],
                    "provenance": {"chunk_ids": ["c1"]},
                    "relationships": [
                        {
                            "relation": "works_with",
                            "target_name": "Bob",
                            "target_type": "Person",
                            "relation_description": "Alice works with Bob.",
                            "provenance": {"chunk_ids": ["c2"], "confidence": 0.88},
                        }
                    ],
                },
                "_provenance": {
                    "entity_chunk_ids": ["c1"],
                    "candidate_chunk_ids": ["c1", "c2", "c3"],
                    "candidate_chunks": {
                        "c1": "Alice background.",
                        "c2": "Alice works with Bob.",
                        "c3": "Noise sentence.",
                    },
                },
            }
        }

        converted = self.graph_pipeline.convert_knowledge_graph(input_data)
        relation = converted["relations"][0]
        self.assertEqual(relation["provenance"]["chunk_ids"], ["c2"])
        self.assertEqual(relation["provenance"]["strategy"], "llm_relation_provenance")
        self.assertAlmostEqual(relation["provenance"]["confidence"], 0.88)

    def test_convert_knowledge_graph_fallbacks_when_relation_provenance_missing(self):
        input_data = {
            "entity1": {
                "central_entity": {
                    "name": "Alice",
                    "type": "Person",
                    "description": "",
                    "attributes": [],
                    "provenance": {"chunk_ids": ["c1"]},
                    "relationships": [
                        {
                            "relation": "works_with",
                            "target_name": "Bob",
                            "target_type": "Person",
                            "relation_description": "Alice works with Bob.",
                        }
                    ],
                },
                "_provenance": {
                    "entity_chunk_ids": ["c1"],
                    "candidate_chunk_ids": ["c1", "c2"],
                    "candidate_chunks": {
                        "c1": "Alice profile.",
                        "c2": "Alice works with Bob in the same team.",
                    },
                },
            }
        }

        converted = self.graph_pipeline.convert_knowledge_graph(input_data)
        relation = converted["relations"][0]
        self.assertEqual(relation["provenance"]["chunk_ids"], ["c2"])
        self.assertEqual(relation["provenance"]["strategy"], "heuristic_sentence_match")


if __name__ == "__main__":
    unittest.main()
