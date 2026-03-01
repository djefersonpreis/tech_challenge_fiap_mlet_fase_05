"""Tests for data cleaning module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import clean_data


class TestCleanData:
    """Tests for clean_data function."""

    def test_returns_dataframe(self, sample_raw_df):
        """Test that clean_data returns a DataFrame."""
        result = clean_data(sample_raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_original(self, sample_raw_df):
        """Test that cleaning does not modify the original DataFrame."""
        original_shape = sample_raw_df.shape
        clean_data(sample_raw_df)
        assert sample_raw_df.shape == original_shape

    def test_no_null_in_matem_portug(self, sample_raw_df):
        """Test that Matem and Portug have no nulls after cleaning."""
        result = clean_data(sample_raw_df)
        assert result["Matem"].isna().sum() == 0
        assert result["Portug"].isna().sum() == 0

    def test_strips_whitespace(self, sample_raw_df):
        """Test that text columns are stripped."""
        sample_raw_df.loc[0, "Gênero"] = "  Menina  "
        result = clean_data(sample_raw_df)
        assert result.loc[0, "Gênero"] == "Menina"

    def test_handles_empty_dataframe(self):
        """Test cleaning an empty DataFrame."""
        empty_df = pd.DataFrame(columns=["Matem", "Portug", "Gênero",
                                          "Instituição de ensino", "Pedra 22",
                                          "Atingiu PV", "Indicado", "Rec Psicologia"])
        result = clean_data(empty_df)
        assert len(result) == 0

    def test_drops_rows_missing_both_matem_portug(self):
        """Test that rows with both Matem and Portug NaN are dropped."""
        df = pd.DataFrame({
            "Matem": [5.0, np.nan, 7.0],
            "Portug": [6.0, np.nan, 8.0],
            "Gênero": ["Menina", "Menino", "Menina"],
            "Instituição de ensino": ["Escola Pública"] * 3,
            "Pedra 22": ["Quartzo"] * 3,
            "Atingiu PV": ["Sim"] * 3,
            "Indicado": ["Não"] * 3,
            "Rec Psicologia": ["Sem limitações"] * 3,
        })
        result = clean_data(df)
        assert len(result) == 2

    def test_fills_single_null_with_median(self):
        """Test that a single null in Matem is filled with median."""
        df = pd.DataFrame({
            "Matem": [4.0, np.nan, 8.0],
            "Portug": [6.0, 7.0, 8.0],
            "Gênero": ["Menina", "Menino", "Menina"],
            "Instituição de ensino": ["Escola Pública"] * 3,
            "Pedra 22": ["Quartzo"] * 3,
            "Atingiu PV": ["Sim"] * 3,
            "Indicado": ["Não"] * 3,
            "Rec Psicologia": ["Sem limitações"] * 3,
        })
        result = clean_data(df)
        assert result["Matem"].isna().sum() == 0
        assert result.loc[1, "Matem"] == 6.0  # median of [4, 8]
