"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_engineering import engineer_features, _extract_destaque
from src.utils.constants import TARGET_COL, PEDRA_ORDER


class TestEngineerFeatures:
    """Tests for engineer_features function."""

    def test_returns_dataframe(self, sample_cleaned_df):
        """Test that engineer_features returns a DataFrame."""
        result = engineer_features(sample_cleaned_df)
        assert isinstance(result, pd.DataFrame)

    def test_creates_target_column(self, sample_cleaned_df):
        """Test that binary target column is created."""
        result = engineer_features(sample_cleaned_df)
        assert TARGET_COL in result.columns
        assert set(result[TARGET_COL].unique()).issubset({0, 1})

    def test_target_logic(self, sample_cleaned_df):
        """Test that Defas < 0 maps to 1 (at risk) and >= 0 maps to 0."""
        result = engineer_features(sample_cleaned_df)
        for idx, row in result.iterrows():
            original_defas = sample_cleaned_df.loc[idx, "Defas"]
            expected = 1 if original_defas < 0 else 0
            assert row[TARGET_COL] == expected, f"Row {idx}: Defas={original_defas}, expected={expected}"

    def test_anos_no_programa(self, sample_cleaned_df):
        """Test Anos_no_programa calculation."""
        result = engineer_features(sample_cleaned_df)
        assert "Anos_no_programa" in result.columns
        for idx, row in result.iterrows():
            expected = 2022 - sample_cleaned_df.loc[idx, "Ano ingresso"]
            assert row["Anos_no_programa"] == expected

    def test_pedra_encoding(self, sample_cleaned_df):
        """Test Pedra 22 is ordinal encoded."""
        result = engineer_features(sample_cleaned_df)
        valid_values = set(PEDRA_ORDER.values())
        for val in result["Pedra 22"].dropna():
            assert val in valid_values

    def test_gender_encoding(self, sample_cleaned_df):
        """Test gender is binary encoded."""
        result = engineer_features(sample_cleaned_df)
        assert set(result["Gênero"].unique()).issubset({0, 1})

    def test_institution_encoding(self, sample_cleaned_df):
        """Test institution is encoded."""
        result = engineer_features(sample_cleaned_df)
        assert result["Instituição de ensino"].dtype in [np.int64, np.float64, int]

    def test_destaque_binary_columns(self, sample_cleaned_df):
        """Test that Destaque binary columns are created."""
        result = engineer_features(sample_cleaned_df)
        for col in ["Destaque IEG_bin", "Destaque IDA_bin", "Destaque IPV_bin"]:
            assert col in result.columns
            assert set(result[col].unique()).issubset({0, 1})

    def test_atingiu_pv_encoding(self, sample_cleaned_df):
        """Test Atingiu PV is binary encoded."""
        result = engineer_features(sample_cleaned_df)
        assert set(result["Atingiu PV"].unique()).issubset({0, 1})

    def test_does_not_modify_original(self, sample_cleaned_df):
        """Test that original DataFrame is not modified."""
        original_cols = set(sample_cleaned_df.columns)
        engineer_features(sample_cleaned_df)
        assert set(sample_cleaned_df.columns) == original_cols


class TestExtractDestaque:
    """Tests for _extract_destaque helper."""

    def test_destaque_returns_1(self):
        assert _extract_destaque("Destaque: A sua boa entrega") == 1

    def test_melhorar_returns_0(self):
        assert _extract_destaque("Melhorar: Melhorar a sua entrega") == 0

    def test_none_returns_0(self):
        assert _extract_destaque(None) == 0

    def test_unknown_returns_0(self):
        assert _extract_destaque("Unknown value") == 0
