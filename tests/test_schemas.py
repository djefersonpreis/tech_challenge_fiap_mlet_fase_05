"""Tests for API schemas."""

import pytest
from pydantic import ValidationError

from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    def test_valid_request(self, sample_prediction_input):
        """Test creating a valid prediction request."""
        req = PredictionRequest(**sample_prediction_input)
        assert req.iaa == 8.5
        assert req.genero == "Menina"

    def test_missing_required_field(self):
        """Test that missing required field raises error."""
        with pytest.raises(ValidationError):
            PredictionRequest()

    def test_model_dump_uses_aliases(self, sample_prediction_input):
        """Test that model_dump with by_alias uses original names."""
        req = PredictionRequest(**sample_prediction_input)
        dumped = req.model_dump(by_alias=True)
        assert "IAA" in dumped
        assert "Idade 22" in dumped

    def test_accepts_alias_names(self):
        """Test that the schema accepts alias field names."""
        data = {
            "IAA": 8.0, "IEG": 7.0, "IPS": 6.0,
            "IDA": 5.0, "IPV": 6.0, "Matem": 6.0,
            "Portug": 7.0, "Idade 22": 13,
            "Ano ingresso": 2020, "Gênero": "Menino",
            "Instituição de ensino": "Escola Pública",
            "Pedra 22": "Quartzo", "Atingiu PV": "Não",
            "Indicado": "Não", "Rec Psicologia": "Sem limitações",
            "Destaque IEG": "Destaque: test", "Destaque IDA": "Melhorar: test",
            "Destaque IPV": "Destaque: test",
        }
        req = PredictionRequest(**data)
        assert req.iaa == 8.0


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self):
        """Test creating a valid prediction response."""
        resp = PredictionResponse(
            prediction=1, probability=0.85,
            risk_level="Alto", message="Risco alto"
        )
        assert resp.prediction == 1
        assert resp.risk_level == "Alto"


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_defaults(self):
        """Test default values."""
        resp = HealthResponse()
        assert resp.status == "healthy"
        assert resp.model_loaded is False
        assert resp.version == "1.0.0"
