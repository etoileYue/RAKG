from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.config import (
    OLLAMA_BASE_URL, DEFAULT_MODEL, EMBEDDING_MODEL, SIMILARITY_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, OPENAI_SIMILARITY_MODEL,
    USE_OPENAI, base_url
)

class LLMProvider:
    def __init__(self):
        if USE_OPENAI:
            # 使用 ChatOpenAI 并启用 JSON 模式
            self.llm = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=base_url,
                temperature=0,
            ).bind(response_format={"type": "json_object"})
            self.embedding_model = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=base_url
            )
            self.similarity_model = ChatOpenAI(
                model=OPENAI_SIMILARITY_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=base_url,
                temperature=0,
            ).bind(response_format={"type": "json_object"})
        else:
            # 使用 Ollama 模型
            self.llm = OllamaLLM(
                model=DEFAULT_MODEL,
                base_url=OLLAMA_BASE_URL,
                format='json',
                temperature=0
            )
            self.embedding_model = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            self.similarity_model = OllamaLLM(
                model=SIMILARITY_MODEL,
                base_url=OLLAMA_BASE_URL,
                format='json',
                temperature=0
            )

    def get_llm(self):
        return self.llm

    def get_embedding_model(self):
        return self.embedding_model

    def get_similarity_model(self):
        return self.similarity_model