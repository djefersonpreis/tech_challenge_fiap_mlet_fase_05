"""Constants and configuration values for the project."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATATHON_DIR = PROJECT_ROOT / "DATATHON"

# Data file
RAW_DATA_FILE = DATATHON_DIR / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

# Model settings
MODEL_FILENAME = "model_v1.joblib"
PIPELINE_FILENAME = "pipeline_v1.joblib"
REFERENCE_DATA_FILENAME = "reference_data.parquet"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
PIPELINE_PATH = MODELS_DIR / PIPELINE_FILENAME
REFERENCE_DATA_PATH = MODELS_DIR / REFERENCE_DATA_FILENAME

# Random seed
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.2

# Target column
TARGET_COL = "Defas_bin"

# Original columns from dataset
ORIGINAL_TARGET = "Defas"

# Numeric features used for modeling
# NOTE: IAN (Indicador de Adequação de Nível) is excluded because it directly
# encodes level adequacy (corr=-0.98 with target), causing data leakage.
# INDE 22 is also excluded because it's a composite index that includes IAN.
# Fase is excluded as combined with Idade 22 it can reconstruct the target.
NUMERIC_FEATURES = [
    "IAA",
    "IEG",
    "IPS",
    "IDA",
    "IPV",
    "Matem",
    "Portug",
    "Idade 22",
    "Anos_no_programa",
]

# Categorical features used for modeling
CATEGORICAL_FEATURES = [
    "Gênero",
    "Instituição de ensino",
    "Pedra 22",
    "Atingiu PV",
    "Indicado",
    "Rec Psicologia",
    "Destaque IEG_bin",
    "Destaque IDA_bin",
    "Destaque IPV_bin",
]

# All features
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Pedra ordering (performance tiers, low to high)
PEDRA_ORDER = {"Quartzo": 0, "Ágata": 1, "Ametista": 2, "Topázio": 3}

# Binary mapping for destaque columns
DESTAQUE_MAPPING = {"Destaque": 1, "Melhorar": 0}

# Binary yes/no mapping
YES_NO_MAPPING = {"Sim": 1, "Não": 0}

# Gender mapping
GENDER_MAPPING = {"Menina": 0, "Menino": 1}

# Institution mapping
INSTITUTION_MAPPING = {"Escola Pública": 0, "Rede Decisão": 1, "Escola JP II": 2}

# Rec Psicologia mapping (ordinal: severity)
REC_PSICOLOGIA_MAPPING = {
    "Sem limitações": 0,
    "Não atendido": 1,
    "Não indicado": 2,
    "Não avaliado": 3,
    "Requer avaliação": 4,
}
