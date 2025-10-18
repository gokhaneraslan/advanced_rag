import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# File Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
MEMORY_DB_PATH = BASE_DIR / "conversation_memory.db"
LOG_DIR = BASE_DIR / "logs"
COLLECTION_NAME = "my_rag_collection"

# Model Names
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-large"

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "openai", "gemini", "groq", "ollama"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# Memory Configuration
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "10"))
MEMORY_CLEANUP_DAYS = int(os.getenv("MEMORY_CLEANUP_DAYS", "30"))

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "100/minute")
ENABLE_CORS = os.getenv("ENABLE_CORS", "true").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "rag_api.log"

# Retrieval Configuration
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.95"))

# Text Splitting Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
SPLITTING_METHOD = os.getenv("SPLITTING_METHOD", "semantic")  # "recursive" or "semantic"


def validate_config():
    """
    Validate configuration and create necessary directories.
    Raises ValueError if critical configuration is missing.
    """
    # Create necessary directories
    for directory in [DATA_DIR, VECTOR_STORE_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
    
    # Validate LLM provider
    valid_providers = ["openai", "gemini", "groq", "ollama"]
    if LLM_PROVIDER not in valid_providers:
        raise ValueError(
            f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. "
            f"Must be one of {valid_providers}"
        )
    
    # Validate API keys based on provider
    if LLM_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
    
    if LLM_PROVIDER == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini provider")
    
    if LLM_PROVIDER == "groq" and not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable is required for Groq provider")
    
    # Validate numeric ranges
    if MAX_MEMORY_MESSAGES < 1:
        raise ValueError("MAX_MEMORY_MESSAGES must be at least 1")
    
    if RETRIEVAL_TOP_K < 1:
        raise ValueError("RETRIEVAL_TOP_K must be at least 1")
    
    if RERANKER_TOP_N < 1 or RERANKER_TOP_N > RETRIEVAL_TOP_K:
        raise ValueError(f"RERANKER_TOP_N must be between 1 and {RETRIEVAL_TOP_K}")
    
    if not 0 <= SIMILARITY_THRESHOLD <= 1:
        raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
    
    if not 0 <= LLM_TEMPERATURE <= 2:
        raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
    
    # Validate splitting method
    if SPLITTING_METHOD not in ["recursive", "semantic"]:
        raise ValueError("SPLITTING_METHOD must be 'recursive' or 'semantic'")
    
    logger.info("Configuration validation passed")
    logger.info(f"LLM Provider: {LLM_PROVIDER} ({LLM_MODEL})")
    logger.info(f"Memory limit: {MAX_MEMORY_MESSAGES} messages")
    logger.info(f"Retrieval: top_k={RETRIEVAL_TOP_K}, reranker_top_n={RERANKER_TOP_N}")


def get_config_summary() -> dict:
    """
    Get a summary of current configuration.
    
    Returns:
        Dictionary with configuration values (API keys masked)
    """
    return {
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "max_memory_messages": MAX_MEMORY_MESSAGES,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "reranker_top_n": RERANKER_TOP_N,
        "splitting_method": SPLITTING_METHOD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "api_host": API_HOST,
        "api_port": API_PORT,
        "cors_enabled": ENABLE_CORS,
        "log_level": LOG_LEVEL,
    }