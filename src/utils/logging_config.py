"""Logging configuration for the project."""

import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Console handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# File handler
logger.add(
    LOG_DIR / "app.log",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)

# Predictions log (for drift monitoring)
predictions_logger = logger.bind(name="predictions")
predictions_logger.add(
    LOG_DIR / "predictions.jsonl",
    rotation="50 MB",
    retention="90 days",
    format="{message}",
    filter=lambda record: record["extra"].get("name") == "predictions",
    level="INFO",
)
