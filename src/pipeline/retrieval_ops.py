"""句子检索与上下文构建能力。"""

import traceback
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.pipeline.shared import logger

class PipelineRetrievalOpsMixin:
    """句子检索与上下文构建方法集合。"""

    def get_sentences_for_entity(self, entity_dic, entity_id, id_to_sentence):
        """根据某个实体entity_id,从映射关系中找出对应的句子列表"""
        if entity_id not in entity_dic:
            raise ValueError(f"Entity '{entity_id}' not found in entity_dic.")

        chunkids = entity_dic[entity_id].get("chunkid", [])
        if not chunkids:
            return []

        if isinstance(chunkids, list):
            chunkid_list = [str(cid).strip() for cid in chunkids if str(cid).strip()]
        else:
            chunkid = str(chunkids).strip()
            chunkid_list = [chunkid] if chunkid else []

        sentences = []
        for chunkid in chunkid_list:
            if chunkid in id_to_sentence:
                sentences.append(id_to_sentence[chunkid])
            else:
                logger.warning("Chunk ID '%s' not found in id_to_sentence.", chunkid)
        return sentences

    @staticmethod
    def _normalize_text(text):
        return " ".join(str(text or "").strip().lower().split())

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _passes_entity_constraint(
        self,
        sentence,
        sentence_vector,
        entity_terms,
        entity_constraint_vector,
        entity_constraint_min_similarity,
    ):
        if not entity_terms and entity_constraint_vector is None:
            return True

        sentence_text = self._normalize_text(sentence)
        if entity_terms:
            for term in entity_terms:
                term_text = self._normalize_text(term)
                if term_text and term_text in sentence_text:
                    return True

        if entity_constraint_vector is not None:
            sim = self._cosine_similarity(sentence_vector, entity_constraint_vector)
            if sim >= entity_constraint_min_similarity:
                return True

        return False

    def _select_with_mmr(self, candidates, sentence_vectors, top_k, mmr_lambda):
        if not candidates:
            return []

        selected = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            best_candidate = None
            best_score = float("-inf")
            for candidate in remaining:
                idx = candidate["idx"]
                relevance = float(candidate["similarity"])
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(
                        self._cosine_similarity(sentence_vectors[idx], sentence_vectors[item["idx"]])
                        for item in selected
                    )
                mmr_score = mmr_lambda * relevance - (1.0 - mmr_lambda) * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
            selected.append(best_candidate)
            remaining = [item for item in remaining if item["idx"] != best_candidate["idx"]]
        return selected

    def get_retriever_context(
        self,
        query,
        sentences,
        sentence_to_id,
        vectors,
        top_k=5,
        min_similarity=0.0,
        dedupe_threshold=0.97,
        use_mmr=False,
        mmr_lambda=0.75,
        entity_terms=None,
        entity_description="",
        entity_constraint_min_similarity=0.2,
    ):
        """
        根据query从一组句子中找出最相似的 Top-K 句子，并返回相关信息句子 + 相似度 + ID
        Args:
            query: 用户查询文本
            sentences: 候选句子列表
            sentence_to_id: 句子 → ID 映射
            vectors: 每个句子的向量表示embedding
            top_k: 返回最相似的前 K 条
            min_similarity: 最低相似度阈值，低于阈值的句子将被过滤
            dedupe_threshold: 句间冗余阈值（余弦相似度），过高重复句将被去重
            use_mmr: 是否启用 MMR 去重
            mmr_lambda: MMR 中相关性与多样性的平衡系数
            entity_terms: 实体名/别名约束列表
            entity_description: 实体描述（用于语义约束）
            entity_constraint_min_similarity: 语义约束最小相似度
        Returns:
            str: 相关信息句子 + 相似度 + ID
        """
        if not sentences or not vectors:
            return []

        query_vector = self.embeddings.embed_query(query)
        sentence_vectors = np.array(vectors)

        try:
            similarities = cosine_similarity([query_vector], sentence_vectors)[0]
        except Exception:
            logger.error("Failed to calculate sentence similarity: %s", traceback.format_exc())
            return []

        entity_terms = entity_terms or []
        entity_constraint_vector = None
        constraint_text_parts = [str(item).strip() for item in entity_terms if str(item).strip()]
        if str(entity_description or "").strip():
            constraint_text_parts.append(str(entity_description).strip())
        constraint_text = " ".join(constraint_text_parts).strip()
        if constraint_text:
            try:
                entity_constraint_vector = np.array(self.embeddings.embed_query(constraint_text))
            except Exception:
                logger.warning("Failed to build entity constraint embedding, fallback to term matching.")

        ranked_indices = np.argsort(similarities)[::-1]
        candidates = []
        for idx in ranked_indices:
            sentence = sentences[idx]
            similarity = float(similarities[idx])
            if similarity < min_similarity:
                continue
            sentence_vector = sentence_vectors[idx]
            if not self._passes_entity_constraint(
                sentence=sentence,
                sentence_vector=sentence_vector,
                entity_terms=entity_terms,
                entity_constraint_vector=entity_constraint_vector,
                entity_constraint_min_similarity=entity_constraint_min_similarity,
            ):
                continue
            candidates.append({"idx": int(idx), "sentence": sentence, "similarity": similarity})

        if not candidates:
            return []

        if use_mmr:
            selected_candidates = self._select_with_mmr(
                candidates=candidates,
                sentence_vectors=sentence_vectors,
                top_k=top_k,
                mmr_lambda=mmr_lambda,
            )
        else:
            selected_candidates = []
            for candidate in candidates:
                if len(selected_candidates) >= top_k:
                    break
                idx = candidate["idx"]
                is_redundant = False
                for chosen in selected_candidates:
                    sim = self._cosine_similarity(sentence_vectors[idx], sentence_vectors[chosen["idx"]])
                    if sim >= dedupe_threshold:
                        is_redundant = True
                        break
                if not is_redundant:
                    selected_candidates.append(candidate)

        retriever_context = []
        for candidate in selected_candidates[:top_k]:
            sentence = candidate["sentence"]
            similarity = candidate["similarity"]
            sentence_id = sentence_to_id.get(sentence)
            if sentence_id is None:
                logger.warning("Sentence not found in sentence_to_id mapping, skip retrieval hit.")
                continue
            retriever_context.append((sentence, similarity, sentence_id))

        return retriever_context
