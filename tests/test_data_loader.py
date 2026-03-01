"""Tests for data loading module."""

import pandas as pd
import pytest

from src.preprocessing.data_loader import load_data
from src.utils.constants import RAW_DATA_FILE


class TestLoadData:
    """Tests for load_data function."""

    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a DataFrame."""
        df = load_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_data_has_expected_shape(self):
        """Test the dataset has expected number of columns."""
        df = load_data()
        assert df.shape[0] > 0
        assert df.shape[1] == 42

    def test_load_data_has_key_columns(self):
        """Test that key columns exist."""
        df = load_data()
        expected_cols = ["Defas", "INDE 22", "IAA", "IEG", "IPS", "IDA", "IPV", "IAN",
                         "Fase", "Pedra 22", "Gênero", "Idade 22"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_data_invalid_path_raises(self):
        """Test that invalid path raises an error."""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path.xlsx")

    def test_load_data_target_column_exists(self):
        """Test that the target column 'Defas' exists and is numeric."""
        df = load_data()
        assert "Defas" in df.columns
        assert df["Defas"].dtype in ["int64", "float64"]
