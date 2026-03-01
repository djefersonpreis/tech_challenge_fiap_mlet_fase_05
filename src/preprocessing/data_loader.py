"""Data loading utilities."""

import pandas as pd
from loguru import logger

from src.utils.constants import RAW_DATA_FILE


def load_data(filepath: str | None = None) -> pd.DataFrame:
    """Load the raw dataset from Excel file.

    Args:
        filepath: Path to the Excel file. If None, uses the default path.

    Returns:
        Raw DataFrame with all columns.
    """
    path = filepath or str(RAW_DATA_FILE)
    logger.info(f"Loading data from {path}")
    df = pd.read_excel(path)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df
