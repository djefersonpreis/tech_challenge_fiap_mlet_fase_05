"""Tests for training module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.training.train import (
    cross_validate_models,
    get_candidate_models,
    select_best_model,
    train_final_model,
)


class TestGetCandidateModels:
    """Tests for get_candidate_models function."""

    def test_returns_dict(self):
        """Test that it returns a dictionary."""
        models = get_candidate_models()
        assert isinstance(models, dict)

    def test_has_expected_models(self):
        """Test that expected model names are present."""
        models = get_candidate_models()
        assert "LogisticRegression" in models
        assert "RandomForest" in models
        assert "GradientBoosting" in models

    def test_models_are_sklearn_estimators(self):
        """Test that values are sklearn estimators."""
        models = get_candidate_models()
        assert isinstance(models["LogisticRegression"], LogisticRegression)
        assert isinstance(models["RandomForest"], RandomForestClassifier)
        assert isinstance(models["GradientBoosting"], GradientBoostingClassifier)


class TestCrossValidateModels:
    """Tests for cross_validate_models function."""

    def test_returns_dict(self, sample_features_target):
        """Test that cross_validate_models returns a dict."""
        X, y = sample_features_target
        results = cross_validate_models(X, y, cv_folds=3)
        assert isinstance(results, dict)

    def test_has_metrics_for_each_model(self, sample_features_target):
        """Test that results contain metrics for each model."""
        X, y = sample_features_target
        results = cross_validate_models(X, y, cv_folds=3)
        for model_name in results:
            assert "f1" in results[model_name]
            assert "recall" in results[model_name]
            assert "roc_auc" in results[model_name]

    def test_metric_values_are_arrays(self, sample_features_target):
        """Test that metric values are numpy arrays."""
        X, y = sample_features_target
        results = cross_validate_models(X, y, cv_folds=3)
        for model_name in results:
            for metric in results[model_name]:
                assert isinstance(results[model_name][metric], np.ndarray)


class TestSelectBestModel:
    """Tests for select_best_model function."""

    def test_returns_string(self):
        """Test that select_best_model returns a string."""
        cv_results = {
            "ModelA": {"f1": np.array([0.8, 0.82, 0.79])},
            "ModelB": {"f1": np.array([0.9, 0.88, 0.91])},
        }
        result = select_best_model(cv_results, metric="f1")
        assert isinstance(result, str)

    def test_selects_highest_mean(self):
        """Test that the model with highest mean metric is selected."""
        cv_results = {
            "ModelA": {"f1": np.array([0.8, 0.82, 0.79])},
            "ModelB": {"f1": np.array([0.9, 0.88, 0.91])},
        }
        result = select_best_model(cv_results, metric="f1")
        assert result == "ModelB"


class TestTrainFinalModel:
    """Tests for train_final_model function."""

    def test_returns_tuple(self, sample_features_target, tmp_path, monkeypatch):
        """Test that train_final_model returns (model, scaler, name) tuple."""
        X, y = sample_features_target
        monkeypatch.setattr("src.training.train.MODELS_DIR", tmp_path)
        monkeypatch.setattr("src.training.train.MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr("src.training.train.PIPELINE_PATH", tmp_path / "scaler.joblib")
        monkeypatch.setattr("src.training.train.REFERENCE_DATA_PATH", tmp_path / "ref.parquet")

        model, scaler, name = train_final_model(X, y, model_name="LogisticRegression")
        assert model is not None
        assert scaler is not None
        assert isinstance(name, str)

    def test_saves_artifacts(self, sample_features_target, tmp_path, monkeypatch):
        """Test that model artifacts are saved to disk."""
        X, y = sample_features_target
        model_path = tmp_path / "model.joblib"
        scaler_path = tmp_path / "scaler.joblib"
        ref_path = tmp_path / "ref.parquet"

        monkeypatch.setattr("src.training.train.MODELS_DIR", tmp_path)
        monkeypatch.setattr("src.training.train.MODEL_PATH", model_path)
        monkeypatch.setattr("src.training.train.PIPELINE_PATH", scaler_path)
        monkeypatch.setattr("src.training.train.REFERENCE_DATA_PATH", ref_path)

        train_final_model(X, y, model_name="LogisticRegression")

        assert model_path.exists()
        assert scaler_path.exists()
        assert ref_path.exists()

    def test_model_can_predict(self, sample_features_target, tmp_path, monkeypatch):
        """Test that trained model can make predictions."""
        X, y = sample_features_target
        monkeypatch.setattr("src.training.train.MODELS_DIR", tmp_path)
        monkeypatch.setattr("src.training.train.MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr("src.training.train.PIPELINE_PATH", tmp_path / "scaler.joblib")
        monkeypatch.setattr("src.training.train.REFERENCE_DATA_PATH", tmp_path / "ref.parquet")

        model, scaler, _ = train_final_model(X, y, model_name="LogisticRegression")

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
