"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest

from src.utils.constants import ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def sample_raw_df():
    """Create a sample raw DataFrame mimicking the XLSX structure."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "RA": [f"RA-{i}" for i in range(n)],
        "Fase": np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], n),
        "Turma": np.random.choice(["A", "B", "C", "D"], n),
        "Nome": [f"Aluno-{i}" for i in range(n)],
        "Ano nasc": np.random.choice(range(2003, 2015), n),
        "Idade 22": np.random.choice(range(8, 20), n),
        "Gênero": np.random.choice(["Menina", "Menino"], n),
        "Ano ingresso": np.random.choice(range(2016, 2023), n),
        "Instituição de ensino": np.random.choice(
            ["Escola Pública", "Rede Decisão", "Escola JP II"], n, p=[0.87, 0.12, 0.01]
        ),
        "Pedra 20": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio", None], n),
        "Pedra 21": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio", None], n),
        "Pedra 22": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
        "INDE 22": np.random.uniform(3.0, 9.5, n).round(3),
        "Cg": np.random.randint(1, 863, n),
        "Cf": np.random.randint(1, 193, n),
        "Ct": np.random.randint(1, 19, n),
        "Nº Av": np.random.choice([3, 4], n),
        "Avaliador1": [f"Avaliador-{np.random.randint(1, 30)}" for _ in range(n)],
        "Rec Av1": np.random.choice(
            ["Mantido na Fase atual", "Promovido de Fase", "Promovido de Fase + Bolsa"], n
        ),
        "Avaliador2": [f"Avaliador-{np.random.randint(1, 30)}" for _ in range(n)],
        "Rec Av2": np.random.choice(
            ["Mantido na Fase atual", "Promovido de Fase", "Promovido de Fase + Bolsa"], n
        ),
        "Avaliador3": [f"Avaliador-{np.random.randint(1, 30)}" for _ in range(n)],
        "Rec Av3": np.random.choice(
            ["Mantido na Fase atual", "Promovido de Fase"], n
        ),
        "Avaliador4": [None] * n,
        "Rec Av4": [None] * n,
        "IAA": np.random.uniform(0, 10, n).round(1),
        "IEG": np.random.uniform(0, 10, n).round(1),
        "IPS": np.random.uniform(2.5, 10, n).round(1),
        "Rec Psicologia": np.random.choice(
            ["Não atendido", "Sem limitações", "Requer avaliação", "Não indicado", "Não avaliado"], n
        ),
        "IDA": np.random.uniform(0, 10, n).round(1),
        "Matem": np.random.uniform(0, 10, n).round(1),
        "Portug": np.random.uniform(0, 10, n).round(1),
        "Inglês": np.random.choice([None, *np.random.uniform(0, 10, 10).round(1).tolist()], n),
        "Indicado": np.random.choice(["Sim", "Não"], n, p=[0.15, 0.85]),
        "Atingiu PV": np.random.choice(["Sim", "Não"], n, p=[0.13, 0.87]),
        "IPV": np.random.uniform(0, 10, n).round(3),
        "IAN": np.random.uniform(0, 10, n).round(1),
        "Fase ideal": np.random.choice(
            ["ALFA  (2º e 3º ano)", "Fase 1 (4º ano)", "Fase 2 (5º e 6º ano)",
             "Fase 3 (7º e 8º ano)", "Fase 4 (9º ano)"], n
        ),
        "Defas": np.random.choice([-3, -2, -1, 0, 1], n, p=[0.05, 0.2, 0.45, 0.25, 0.05]),
        "Destaque IEG": np.random.choice([
            "Destaque: A sua boa entrega das lições de casa.",
            "Melhorar: Melhorar a sua entrega de lições de casa.",
        ], n),
        "Destaque IDA": np.random.choice([
            "Destaque: As suas boas notas na Passos Mágicos.",
            "Melhorar: Empenhar-se mais nas aulas e avaliações.",
        ], n),
        "Destaque IPV": np.random.choice([
            "Destaque: A sua boa integração aos Princípios Passos Mágicos.",
            "Melhorar: Integrar-se mais aos Princípios Passos Mágicos.",
        ], n),
    })


@pytest.fixture
def sample_cleaned_df(sample_raw_df):
    """Create a cleaned DataFrame."""
    from src.preprocessing.cleaner import clean_data
    return clean_data(sample_raw_df)


@pytest.fixture
def sample_engineered_df(sample_cleaned_df):
    """Create a feature-engineered DataFrame."""
    from src.preprocessing.feature_engineering import engineer_features
    return engineer_features(sample_cleaned_df)


@pytest.fixture
def sample_features_target(sample_engineered_df):
    """Create X, y pair ready for training."""
    from src.utils.constants import ALL_FEATURES, TARGET_COL
    X = sample_engineered_df[ALL_FEATURES]
    y = sample_engineered_df[TARGET_COL]
    return X, y


@pytest.fixture
def sample_prediction_input():
    """Create a sample prediction request dict."""
    return {
        "IAA": 8.5,
        "IEG": 7.0,
        "IPS": 6.5,
        "IDA": 6.0,
        "IPV": 7.3,
        "Matem": 6.5,
        "Portug": 7.0,
        "Idade 22": 14,
        "Ano ingresso": 2020,
        "Gênero": "Menina",
        "Instituição de ensino": "Escola Pública",
        "Pedra 22": "Ametista",
        "Atingiu PV": "Não",
        "Indicado": "Não",
        "Rec Psicologia": "Sem limitações",
        "Destaque IEG": "Destaque: A sua boa entrega das lições de casa.",
        "Destaque IDA": "Destaque: As suas boas notas na Passos Mágicos.",
        "Destaque IPV": "Destaque: A sua boa integração aos Princípios Passos Mágicos.",
    }
