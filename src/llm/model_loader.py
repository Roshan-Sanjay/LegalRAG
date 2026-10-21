import logging

from src.config.settings import OLLAMA_MODEL


logger = logging.getLogger(__name__)


class OllamaConfig:

    MODEL_NAME = OLLAMA_MODEL

    TEMPERATURE = 0.1

    MAX_TOKENS = 1024



