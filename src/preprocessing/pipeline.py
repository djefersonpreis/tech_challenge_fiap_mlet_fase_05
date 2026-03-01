"""Preprocessing pipeline module - orchestrates full data preparation."""

import pandas as pd
from loguru import logger

from src.preprocessing.cleaner import clean_data
from src.preprocessing.data_loader import load_data
from src.preprocessing.feature_engineering import engineer_features
from src.utils.constants import ALL_FEATURES, TARGET_COL


def build_pipeline(filepath: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Run the full preprocessing pipeline: load -> clean -> engineer features.

    Args:
        filepath: Optional path to raw data file.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    logger.info("Running full preprocessing pipeline")

    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    logger.info(
        f"Pipeline complete. X shape: {X.shape}, "
        f"y distribution: {y.value_counts().to_dict()}"
    )
    return X, y


def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess a single input record for prediction (API usage).

    Applies the same feature engineering transformations used during training
    to a raw input dictionary.

    Args:
        data: Dictionary with raw feature values.

    Returns:
        DataFrame with one row, ready for model prediction.
    """
    from src.utils.constants import (
        DESTAQUE_MAPPING,
        GENDER_MAPPING,
        INSTITUTION_MAPPING,
        PEDRA_ORDER,
        REC_PSICOLOGIA_MAPPING,
        YES_NO_MAPPING,
    )

    df = pd.DataFrame([data])

    # Derive Anos_no_programa
    if "Ano ingresso" in df.columns and "Anos_no_programa" not in df.columns:
        df["Anos_no_programa"] = 2022 - df["Ano ingresso"]

    # Encode categoricals if still in string form
    if df["Pedra 22"].dtype == object:
        df["Pedra 22"] = df["Pedra 22"].map(PEDRA_ORDER)
    if df["Gênero"].dtype == object:
        df["Gênero"] = df["Gênero"].map(GENDER_MAPPING)
    if df["Instituição de ensino"].dtype == object:
        df["Instituição de ensino"] = df["Instituição de ensino"].map(
            INSTITUTION_MAPPING
        )
    if "Atingiu PV" in df.columns and df["Atingiu PV"].dtype == object:
        df["Atingiu PV"] = df["Atingiu PV"].map(YES_NO_MAPPING)
    if "Indicado" in df.columns and df["Indicado"].dtype == object:
        df["Indicado"] = df["Indicado"].map(YES_NO_MAPPING)
    if "Rec Psicologia" in df.columns and df["Rec Psicologia"].dtype == object:
        df["Rec Psicologia"] = df["Rec Psicologia"].map(REC_PSICOLOGIA_MAPPING)

    # Destaque columns
    for col_base in ["Destaque IEG", "Destaque IDA", "Destaque IPV"]:
        bin_col = f"{col_base}_bin"
        if col_base in df.columns and bin_col not in df.columns:
            df[bin_col] = df[col_base].apply(
                lambda v: next(
                    (val for key, val in DESTAQUE_MAPPING.items()
                     if str(v).startswith(key)),
                    0,
                )
                if pd.notna(v)
                else 0
            )

    # Select only the features the model expects
    result = df[ALL_FEATURES]
    return result
