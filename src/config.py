# Global configuration variables
OLLAMA_BASE_URL = "http://localhost:11434"  # Change this to your Ollama server URL
DEFAULT_MODEL = "qwen2.5:72b"
EMBEDDING_MODEL = "bge-m3:latest"   
SIMILARITY_MODEL = "qwen2:7b"



# OpenAI Configuration
base_url="https://api.siliconflow.cn/v1" # https://api.siliconflow.cn/v1 for siliconflow
OPENAI_API_KEY = "your_api_key"  # Set your OpenAI API key here
OPENAI_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # Default model
OPENAI_EMBEDDING_MODEL = "BAAI/bge-m3"  # Default embedding model
OPENAI_SIMILARITY_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # Model for similarity checks


# Model Provider Selection
USE_OPENAI = True  # Set to True to use OpenAI, False to use Ollama