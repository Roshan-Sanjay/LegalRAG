import logging

from src.config.settings import (
    MAX_GENERATION_TOKENS,
    OLLAMA_MODEL,
    TEMPERATURE as CONFIGURED_TEMPERATURE,
)


logger = logging.getLogger(__name__)


class OllamaConfig:

    MODEL_NAME = OLLAMA_MODEL

    TEMPERATURE = CONFIGURED_TEMPERATURE

    MAX_TOKENS = MAX_GENERATION_TOKENS
