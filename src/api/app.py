"""FastAPI application for the Passos Mágicos prediction model."""

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.preprocessing.pipeline import preprocess_input
from src.utils.constants import MODEL_PATH, PIPELINE_PATH

# Global model and scaler
_model = None
_scaler = None


def load_model_artifacts():
    """Load the trained model and scaler from disk."""
    global _model, _scaler
    try:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(PIPELINE_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
        logger.info(f"Scaler loaded from {PIPELINE_PATH}")
    except FileNotFoundError as e:
        logger.warning(f"Model artifacts not found: {e}. Run training first.")


def get_model():
    """Get the loaded model, raising error if not available."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first.",
        )
    return _model


def get_scaler():
    """Get the loaded scaler, raising error if not available."""
    if _scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Scaler not loaded. Run training first.",
        )
    return _scaler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — load model on startup."""
    load_model_artifacts()
    yield


app = FastAPI(
    title="Passos Mágicos - Predição de Defasagem Escolar",
    description=(
        "API para predição do risco de defasagem escolar de estudantes "
        "da Associação Passos Mágicos. Utiliza modelo de Machine Learning "
        "treinado com dados do PEDE 2024 (período 2022-2024)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predição"])
async def predict(request: PredictionRequest):
    """Predict the risk of educational lag for a student.

    Receives student data and returns risk prediction with probability.
    """
    start_time = time.time()
    model = get_model()
    scaler = get_scaler()

    try:
        # Convert request to dict using aliases
        input_data = request.model_dump(by_alias=True)
        logger.debug(f"Input data: {input_data}")

        # Preprocess input
        X = preprocess_input(input_data)

        # Scale
        X_scaled = pd.DataFrame(
            scaler.transform(X), columns=X.columns, index=X.index
        )

        # Predict
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        # Risk level
        if probability < 0.3:
            risk_level = "Baixo"
            message = "O estudante apresenta baixo risco de defasagem escolar."
        elif probability < 0.7:
            risk_level = "Médio"
            message = "O estudante apresenta risco moderado de defasagem escolar. Recomenda-se acompanhamento."
        else:
            risk_level = "Alto"
            message = "O estudante apresenta alto risco de defasagem escolar. Intervenção recomendada."

        response = PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            message=message,
        )

        # Log prediction for drift monitoring
        latency_ms = (time.time() - start_time) * 1000
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": input_data,
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "latency_ms": round(latency_ms, 2),
        }
        logger.bind(name="predictions").info(json.dumps(log_entry))

        logger.info(
            f"Prediction: {prediction} (prob={probability:.4f}, "
            f"risk={risk_level}) in {latency_ms:.1f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
