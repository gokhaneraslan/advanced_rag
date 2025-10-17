from pathlib import Path

# File Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
COLLECTION_NAME = "my_rag_collection"

# Model Names
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-large"

# LLM Provider Configuration
LLM_PROVIDER = "gemini" # "openai", "gemini", "groq", "ollama"
LLM_MODEL = "gemini-2.5-flash"