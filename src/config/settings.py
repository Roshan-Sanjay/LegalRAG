"""Environment-backed configuration for LegalRAG."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # .env loading is optional; real environment variables still work.
    pass

ROOT_DIR = Path(__file__).resolve().parents[2]

def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else ROOT_DIR / value

CHUNKS_PATH = _path("CHUNKS_PATH", "data/chunks/chunked_judgments.csv")
EMBEDDINGS_PATH = _path("EMBEDDINGS_PATH", "data/embeddings/legalbert_embeddings.joblib")
DENSE_INDEX_PATH = _path("DENSE_INDEX_PATH", "vector_db/nearest_neighbors.joblib")
TFIDF_INDEX_PATH = _path("TFIDF_INDEX_PATH", "vector_db/tfidf_store.joblib")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
BACKEND_URL = os.getenv("BACKEND_URL", f"http://{API_HOST}:{API_PORT}").rstrip("/")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nlpaueb/legal-bert-base-uncased")
TOP_K = int(os.getenv("TOP_K", "20"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.65"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.35"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "4000"))
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
MAX_GENERATION_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "1024"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() in {"1", "true", "yes"}
ENABLE_QUERY_EXPANSION = False
