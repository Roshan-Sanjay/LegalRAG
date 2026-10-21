"""
Central configuration file for LegalRAG.
Modify values here instead of changing them
throughout the project.
"""

# ==========================
# Retrieval Settings
# ==========================

TOP_K = 20

DENSE_WEIGHT = 0.65
SPARSE_WEIGHT = 0.35

# ==========================
# Context Builder
# ==========================

MAX_CONTEXT_CHARS = 6000

# ==========================
# LLM Settings
# ==========================

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "llama3:8b"

TEMPERATURE = 0.1

MAX_GENERATION_TOKENS = 300

# ==========================
# Logging
# ==========================

LOG_LEVEL = "INFO"

# ==========================
# Future Settings
# ==========================

ENABLE_RERANKING = False

ENABLE_QUERY_EXPANSION = False