"""Tests for preprocessing pipeline module."""

import pandas as pd
import pytest

from src.preprocessing.pipeline import build_pipeline, preprocess_input
from src.utils.constants import ALL_FEATURES, TARGET_COL


class TestBuildPipeline:
    """Tests for build_pipeline function."""

    def test_returns_tuple(self):
        """Test that build_pipeline returns (X, y) tuple."""
        X, y = build_pipeline()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_features_match_expected(self):
        """Test that output features match ALL_FEATURES."""
        X, y = build_pipeline()
        assert list(X.columns) == ALL_FEATURES

    def test_no_nulls_in_features(self):
        """Test that feature matrix has no nulls."""
        X, y = build_pipeline()
        assert X.isna().sum().sum() == 0

    def test_target_is_binary(self):
        """Test that target is binary."""
        X, y = build_pipeline()
        assert set(y.unique()).issubset({0, 1})

    def test_shapes_consistent(self):
        """Test that X and y have the same number of rows."""
        X, y = build_pipeline()
        assert len(X) == len(y)


class TestPreprocessInput:
    """Tests for preprocess_input function."""

    def test_returns_dataframe(self, sample_prediction_input):
        """Test that preprocess_input returns a DataFrame."""
        result = preprocess_input(sample_prediction_input)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_correct_columns(self, sample_prediction_input):
        """Test that output has exactly the expected feature columns."""
        result = preprocess_input(sample_prediction_input)
        assert list(result.columns) == ALL_FEATURES

    def test_output_single_row(self, sample_prediction_input):
        """Test that output has exactly one row."""
        result = preprocess_input(sample_prediction_input)
        assert len(result) == 1

    def test_encodes_categorical_to_numeric(self, sample_prediction_input):
        """Test that categorical features are encoded to numeric."""
        result = preprocess_input(sample_prediction_input)
        for col in result.columns:
            assert result[col].dtype in ["int64", "float64", int, float], \
                f"Column {col} is not numeric: {result[col].dtype}"

    def test_derives_anos_no_programa(self, sample_prediction_input):
        """Test that Anos_no_programa is correctly derived."""
        result = preprocess_input(sample_prediction_input)
        expected = 2022 - sample_prediction_input["Ano ingresso"]
        assert result["Anos_no_programa"].iloc[0] == expected
