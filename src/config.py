# Global configuration variables
OLLAMA_BASE_URL = "http://localhost:11434"  # Change this to your Ollama server URL
DEFAULT_MODEL = "qwen2.5:72b"
EMBEDDING_MODEL = "bge-m3:latest"   
SIMILARITY_MODEL = "qwen2:7b"



# OpenAI Configuration
base_url="https://api.siliconflow.cn/v1" # https://api.siliconflow.cn/v1 for siliconflow
OPENAI_API_KEY = "sk-mmqulwgspimiaevtxtikfywqmhwtqwmwladvxialnjkacobk"  # Set your OpenAI API key here
OPENAI_MODEL = "Qwen/Qwen3-32B"  # Default model
# OPENAI_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OPENAI_EMBEDDING_MODEL = "BAAI/bge-m3"  # Default embedding model
# OPENAI_SIMILARITY_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # Model for similarity checks
OPENAI_SIMILARITY_MODEL = "Qwen/Qwen3-32B"
OPENAI_MAX_TOKENS = 8192  # Max completion tokens for extraction calls
# OPENAI_SIMILARITY_MAX_TOKENS = 1024  # Similarity calls do not need long outputs


# Model Provider Selection
USE_OPENAI = True  # Set to True to use OpenAI, False to use Ollama

# Prompt language used by the main runtime pipeline.
# Supported values: "zh", "en"
PROMPT_LANGUAGE = "zh"

# Unified LLM batch execution controls.
LLM_PARALLEL_ENABLED = False
LLM_PARALLEL_MAX_WORKERS = 4
