"""实体相似候选生成与两阶段判定能力。"""

from itertools import combinations
import traceback
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from src.prompt import judge_sim_entity_en
from src.utils import parse_similarity_response
from src.utils import retry
from src.pipeline.shared import debug_logger
from src.pipeline.shared import logger

class PipelineSimilarityOpsMixin:
    """实体相似候选生成与判定方法集合。"""

    def similarity_candidates(self, entities, threshold=0.60):
        """通过向量相似度判断entities列表中是否存在候选相似实体"""
        def get_embedding_vector(text):
            result = self.embeddings.embed_documents([text])
            if isinstance(result, list) and isinstance(result[0], list):
                return result[0]
            return result

        entity_texts = {k: f"{v['name']} {v['type']}" for k, v in entities.items()}
        vectors = {k: get_embedding_vector(text) for k, text in entity_texts.items()}

        keys = list(vectors.keys())
        sim_matrix = np.zeros((len(keys), len(keys)))

        for i, j in combinations(range(len(keys)), 2):
            sim = float(cosine_similarity([vectors[keys[i]]], [vectors[keys[j]]])[0][0])
            sim_matrix[i][j] = sim

        candidates = [
            (keys[i], keys[j], float(sim_matrix[i][j]))
            for i, j in zip(*np.where(sim_matrix > threshold))
        ]
        return candidates

    @retry()
    def similarity_llm_single(self, entity1, entity2):
        """调用LLM判断两个相似实体是否为同一实体"""
        prompt = ChatPromptTemplate.from_template(judge_sim_entity_en)
        chain = prompt | self.similarity_model
        result = chain.invoke({"entity1": str(entity1), "entity2": str(entity2)})
        debug_logger.debug("-similarity_llm_single-")
        debug_logger.debug(f"entity1={entity1}, entity2={entity2}, result={result}")
        return parse_similarity_response(result)

    def _run_two_pass_similarity_disambiguation(
        self,
        candidates,
        left_entities,
        right_entities=None,
        threshold=0.60,
        gray_margin=0.05,
        scope_label="entity",
    ):
        """
        二阶段消歧
        对候选实体对进行两轮相似度消歧，第一轮筛出明确匹配和灰区样本
        第二轮对灰区样本再次判定，从而得到最终正匹配结果,并保留完整的灰区复核记录
        """
        if right_entities is None:
            right_entities = left_entities

        positives = []
        gray_queue = []
        for left_id, right_id, score in candidates:
            entity1 = left_entities.get(left_id)
            entity2 = right_entities.get(right_id)
            try:
                result = self.similarity_llm_single(entity1, entity2)
                is_boundary = self._is_boundary_candidate(score, threshold, gray_margin)
                needs_review = result.get("needs_review", False) or is_boundary

                if needs_review:
                    gray_queue.append(
                        {
                            "pair": (left_id, right_id),
                            "similarity_score": score,
                            "reason": result.get("reason", "unknown"),
                            "parse_status": result.get("parse_status", "unknown"),
                            "first_pass_result": result,
                        }
                    )
                    continue

                if result.get("result", False):
                    positives.append((left_id, right_id, score))
            except Exception:
                logger.error(
                    "Error processing %s pair (%s, %s): %s",
                    scope_label,
                    left_id,
                    right_id,
                    traceback.format_exc(),
                )

        resolved_by_second_pass = 0
        for item in gray_queue:
            left_id, right_id = item["pair"]
            entity1 = left_entities.get(left_id)
            entity2 = right_entities.get(right_id)
            try:
                second_pass_result = self.similarity_llm_single(entity1, entity2)
                if second_pass_result.get("result", False):
                    positives.append((left_id, right_id, item["similarity_score"]))
                    resolved_by_second_pass += 1
                item["second_pass_result"] = second_pass_result
            except Exception:
                logger.error(
                    "Error in second-pass %s disambiguation for pair (%s, %s): %s",
                    scope_label,
                    left_id,
                    right_id,
                    traceback.format_exc(),
                )

        return positives, gray_queue, resolved_by_second_pass

    def similarity_result(self, entities, threshold=0.60, gray_margin=0.05)->list[tuple]:
        """返回相似实体，两两一组"""
        candidates = self.similarity_candidates(entities, threshold=threshold)
        positives, gray_queue, resolved_by_second_pass = self._run_two_pass_similarity_disambiguation(
            candidates=candidates,
            left_entities=entities,
            threshold=threshold,
            gray_margin=gray_margin,
            scope_label="entity",
        )
        candidates_result = [(left_id, right_id) for left_id, right_id, _ in positives]

        self.last_disambiguation_gray_queue = gray_queue
        logger.info(
            "Disambiguation summary: candidates=%s, gray_queue=%s, "
            "resolved_by_second_pass=%s, merged_pairs=%s",
            len(candidates),
            len(gray_queue),
            resolved_by_second_pass,
            len(candidates_result),
        )
        return candidates_result

    def similarity_candidates_cross(self, left_entities, right_entities, threshold=0.60)->list[tuple]:
        """根据向量相似度判断新实体与已有实体之间是否存在相似实体，两两一组"""
        if not left_entities or not right_entities:
            return []

        left_keys = list(left_entities.keys())
        right_keys = list(right_entities.keys())

        left_texts = [f"{left_entities[k].get('name', '')} {left_entities[k].get('type', '')}" for k in left_keys]
        right_texts = [
            f"{right_entities[k].get('name', '')} {right_entities[k].get('type', '')}" for k in right_keys
        ]

        left_vectors = np.array(self.embeddings.embed_documents(left_texts))
        right_vectors = np.array(self.embeddings.embed_documents(right_texts))
        sim_matrix = cosine_similarity(left_vectors, right_vectors)

        candidates = []
        matched = np.argwhere(sim_matrix > threshold)
        for left_idx, right_idx in matched:
            candidates.append(
                (
                    left_keys[int(left_idx)],
                    right_keys[int(right_idx)],
                    float(sim_matrix[int(left_idx)][int(right_idx)]),
                )
            )
        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    def cross_similarity_result(self, left_entities, right_entities, threshold=0.60, gray_margin=0.05):
        """在新旧实体之间找出高置信匹配，并为每个新实体选出一个最相似的已有实体作为对齐目标"""
        candidates = self.similarity_candidates_cross(
            left_entities=left_entities,
            right_entities=right_entities,
            threshold=threshold,
        )

        positives, gray_queue, _ = self._run_two_pass_similarity_disambiguation(
            candidates=candidates,
            left_entities=left_entities,
            right_entities=right_entities,
            threshold=threshold,
            gray_margin=gray_margin,
            scope_label="cross-graph entity",
        )

        best_match = {}
        for left_id, right_id, score in positives:
            current = best_match.get(left_id)
            if current is None or score > current["score"]:
                best_match[left_id] = {"target_id": right_id, "score": score}

        logger.info(
            "Cross-graph entity alignment summary: candidates=%s, gray_queue=%s, matched=%s",
            len(candidates),
            len(gray_queue),
            len(best_match),
        )
        return best_match
