"""Feature engineering module."""

import pandas as pd
from loguru import logger

from src.utils.constants import (
    DESTAQUE_MAPPING,
    GENDER_MAPPING,
    INSTITUTION_MAPPING,
    PEDRA_ORDER,
    REC_PSICOLOGIA_MAPPING,
    TARGET_COL,
    YES_NO_MAPPING,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create and transform features for modeling.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with engineered features ready for modeling.
    """
    logger.info("Starting feature engineering")
    df = df.copy()

    # --- Target variable ---
    # Defas: negative = behind ideal (defasagem), 0 = on track, positive = ahead
    # Binary target: 1 = at risk (Defas < 0), 0 = on track or ahead (Defas >= 0)
    df[TARGET_COL] = (df["Defas"] < 0).astype(int)
    logger.info(
        f"Target '{TARGET_COL}' created: "
        f"{df[TARGET_COL].value_counts().to_dict()}"
    )

    # --- Derived numeric features ---
    # Years in program
    df["Anos_no_programa"] = 2022 - df["Ano ingresso"]

    # --- Categorical encoding ---
    # Destaque columns: extract Destaque/Melhorar prefix as binary
    for col in ["Destaque IEG", "Destaque IDA", "Destaque IPV"]:
        bin_col = f"{col}_bin"
        df[bin_col] = df[col].apply(_extract_destaque)
        logger.debug(f"Encoded {col} -> {bin_col}")

    # Pedra 22: ordinal encoding
    df["Pedra 22"] = df["Pedra 22"].map(PEDRA_ORDER)

    # Gender
    df["Gênero"] = df["Gênero"].map(GENDER_MAPPING)

    # Institution
    df["Instituição de ensino"] = df["Instituição de ensino"].map(INSTITUTION_MAPPING)

    # Atingiu PV / Indicado
    df["Atingiu PV"] = df["Atingiu PV"].map(YES_NO_MAPPING)
    df["Indicado"] = df["Indicado"].map(YES_NO_MAPPING)

    # Rec Psicologia
    df["Rec Psicologia"] = df["Rec Psicologia"].map(REC_PSICOLOGIA_MAPPING)

    logger.info(f"Feature engineering done. Shape: {df.shape}")
    return df


def _extract_destaque(value: str) -> int:
    """Extract binary destaque/melhorar from text.

    Args:
        value: Text value starting with 'Destaque:' or 'Melhorar:'.

    Returns:
        1 for destaque, 0 for melhorar.
    """
    if pd.isna(value):
        return 0
    for key, val in DESTAQUE_MAPPING.items():
        if str(value).startswith(key):
            return val
    return 0
