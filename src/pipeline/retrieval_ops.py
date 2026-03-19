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

    def get_retriever_context(self, query, sentences, sentence_to_id, vectors, top_k=5):
        """
        根据query从一组句子中找出最相似的 Top-K 句子，并返回相关信息句子 + 相似度 + ID
        Args:
            query: 用户查询文本
            sentences: 候选句子列表
            sentence_to_id: 句子 → ID 映射
            vectors: 每个句子的向量表示embedding
            top_k: 返回最相似的前 K 条
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

        top_indices = np.argsort(similarities)[::-1][:top_k]
        retriever_context = []
        for idx in top_indices:
            sentence = sentences[idx]
            similarity = similarities[idx]
            sentence_id = sentence_to_id[sentence]
            retriever_context.append((sentence, similarity, sentence_id))

        return retriever_context
