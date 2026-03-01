"""Tests for the FastAPI application."""

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.api.app import app, _model, _scaler
import src.api.app as app_module


@pytest.fixture
def trained_model_artifacts(sample_features_target, tmp_path):
    """Create trained model artifacts for API testing."""
    X, y = sample_features_target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_scaled, y)

    model_path = tmp_path / "model.joblib"
    scaler_path = tmp_path / "scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler, model_path, scaler_path


@pytest.fixture
def client_with_model(trained_model_artifacts):
    """Create a test client with model loaded."""
    model, scaler, _, _ = trained_model_artifacts

    # Inject model and scaler into module globals
    app_module._model = model
    app_module._scaler = scaler

    client = TestClient(app)
    yield client

    # Cleanup
    app_module._model = None
    app_module._scaler = None


@pytest.fixture
def client_without_model():
    """Create a test client without model loaded."""
    app_module._model = None
    app_module._scaler = None
    client = TestClient(app)
    yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_with_model(self, client_with_model):
        """Test health check when model is loaded."""
        response = client_with_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_without_model(self, client_without_model):
        """Test health check when model is not loaded."""
        response = client_without_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_success(self, client_with_model, sample_prediction_input):
        """Test successful prediction."""
        response = client_with_model.post("/predict", json=sample_prediction_input)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert data["risk_level"] in ["Baixo", "Médio", "Alto"]

    def test_predict_no_model_returns_503(self, client_without_model, sample_prediction_input):
        """Test prediction without model returns 503."""
        response = client_without_model.post("/predict", json=sample_prediction_input)
        assert response.status_code == 503

    def test_predict_invalid_input(self, client_with_model):
        """Test prediction with invalid input returns 422."""
        response = client_with_model.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422

    def test_predict_response_structure(self, client_with_model, sample_prediction_input):
        """Test that response has the expected structure."""
        response = client_with_model.post("/predict", json=sample_prediction_input)
        data = response.json()
        assert set(data.keys()) == {"prediction", "probability", "risk_level", "message"}

    def test_predict_risk_levels(self, client_with_model, sample_prediction_input):
        """Test that risk level corresponds to probability."""
        response = client_with_model.post("/predict", json=sample_prediction_input)
        data = response.json()
        prob = data["probability"]
        if prob < 0.3:
            assert data["risk_level"] == "Baixo"
        elif prob < 0.7:
            assert data["risk_level"] == "Médio"
        else:
            assert data["risk_level"] == "Alto"


class TestLoadModelArtifacts:
    """Tests for load_model_artifacts."""

    def test_load_with_missing_files(self, monkeypatch):
        """Test that load_model_artifacts handles missing files gracefully."""
        from pathlib import Path
        monkeypatch.setattr("src.api.app.MODEL_PATH", Path("/nonexistent/model.joblib"))
        monkeypatch.setattr("src.api.app.PIPELINE_PATH", Path("/nonexistent/scaler.joblib"))

        from src.api.app import load_model_artifacts
        load_model_artifacts()
        # Should not raise, just log warning
        assert app_module._model is None

    def test_get_model_raises_when_none(self):
        """Test get_model raises HTTPException when model is None."""
        from fastapi import HTTPException
        from src.api.app import get_model
        app_module._model = None
        with pytest.raises(HTTPException) as exc:
            get_model()
        assert exc.value.status_code == 503

    def test_get_scaler_raises_when_none(self):
        """Test get_scaler raises HTTPException when scaler is None."""
        from fastapi import HTTPException
        from src.api.app import get_scaler
        app_module._scaler = None
        with pytest.raises(HTTPException) as exc:
            get_scaler()
        assert exc.value.status_code == 503
