"""Tests for run_training CLI module."""

import pytest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np


class TestRunTrainingMain:
    """Tests for the main training CLI function."""

    @patch("src.training.run_training.evaluate_model")
    @patch("src.training.run_training.train_final_model")
    @patch("src.training.run_training.select_best_model")
    @patch("src.training.run_training.cross_validate_models")
    @patch("src.training.run_training.build_pipeline")
    def test_main_runs_end_to_end(
        self, mock_pipeline, mock_cv, mock_select, mock_train, mock_eval
    ):
        """Test that main function runs the complete pipeline."""
        from src.training.run_training import main

        # Setup mocks
        n = 50
        X = pd.DataFrame(np.random.rand(n, 12), columns=[f"f{i}" for i in range(12)])
        y = pd.Series(np.random.choice([0, 1], n))
        mock_pipeline.return_value = (X, y)
        mock_cv.return_value = {"Model": {"f1": np.array([0.8])}}
        mock_select.return_value = "Model"
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_train.return_value = (mock_model, mock_scaler, "Model")
        mock_eval.return_value = {
            "f1": 0.85, "recall": 0.8, "roc_auc": 0.9, "accuracy": 0.82
        }

        model, scaler, metrics = main()
        assert model is not None
        assert metrics["f1"] == 0.85
        mock_pipeline.assert_called_once()
        mock_cv.assert_called_once()
        mock_train.assert_called_once()
        mock_eval.assert_called_once()
