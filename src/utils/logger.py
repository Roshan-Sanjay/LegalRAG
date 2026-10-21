import logging

from src.config.settings import (
    LOG_LEVEL
)

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL.upper()
    ),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "LegalRAG"
)