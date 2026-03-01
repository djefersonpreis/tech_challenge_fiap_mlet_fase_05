"""Tests for evaluation module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.evaluate import compute_metrics, evaluate_model


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_returns_dict(self):
        """Test that compute_metrics returns a dict."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = compute_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test that result has expected metric keys."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = compute_metrics(y_true, y_pred)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "confusion_matrix" in result

    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        result = compute_metrics(y_true, y_pred)
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_with_probabilities(self):
        """Test that roc_auc is computed when probabilities provided."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2])
        result = compute_metrics(y_true, y_pred, y_proba)
        assert "roc_auc" in result
        assert 0 <= result["roc_auc"] <= 1

    def test_without_probabilities(self):
        """Test that roc_auc is not computed without probabilities."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        result = compute_metrics(y_true, y_pred, None)
        assert "roc_auc" not in result

    def test_confusion_matrix_shape(self):
        """Test that confusion matrix is 2x2 list."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        result = compute_metrics(y_true, y_pred)
        assert len(result["confusion_matrix"]) == 2
        assert len(result["confusion_matrix"][0]) == 2

    def test_metrics_values_in_range(self):
        """Test that metric values are between 0 and 1."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        result = compute_metrics(y_true, y_pred)
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert 0 <= result[metric] <= 1


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_evaluate_with_scaler(self, sample_features_target):
        """Test evaluate_model with a scaler."""
        X, y = sample_features_target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        metrics = evaluate_model(model, X, y, scaler=scaler)
        assert isinstance(metrics, dict)
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_without_scaler(self, sample_features_target):
        """Test evaluate_model without a scaler."""
        X, y = sample_features_target
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y, scaler=None)
        assert isinstance(metrics, dict)
        assert "f1" in metrics
