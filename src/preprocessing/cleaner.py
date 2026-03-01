"""Data cleaning module."""

import pandas as pd
from loguru import logger


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data: handle missing values, fix types, drop unusable rows.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Starting data cleaning")
    df = df.copy()
    initial_rows = len(df)

    # Drop rows where both Matem and Portug are missing (only 2 rows)
    df = df.dropna(subset=["Matem", "Portug"], how="all")
    logger.info(f"Dropped {initial_rows - len(df)} rows with missing Matem/Portug")

    # Fill remaining individual NaN in Matem/Portug with median
    for col in ["Matem", "Portug"]:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(f"Filled {col} NaN with median={median_val:.2f}")

    # Inglês has 577/860 nulls — too sparse, exclude from features
    # Pedra 20 has 537 nulls, Pedra 21 has 398 nulls — exclude from features
    # Avaliador columns are identifiers, not features — exclude

    # Standardize text columns
    df["Gênero"] = df["Gênero"].str.strip()
    df["Instituição de ensino"] = df["Instituição de ensino"].str.strip()
    df["Pedra 22"] = df["Pedra 22"].str.strip()
    df["Atingiu PV"] = df["Atingiu PV"].str.strip()
    df["Indicado"] = df["Indicado"].str.strip()
    df["Rec Psicologia"] = df["Rec Psicologia"].str.strip()

    logger.info(f"Cleaning done. {len(df)} rows remaining from {initial_rows}")
    return df
