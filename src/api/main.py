"""
FastAPI serving layer for the churn prediction model.

Loads the model + label encoders promoted by src/train.py and exposes
/predict, /health, and /model-info endpoints.

Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

MODELS_DIR = Path("models")

_state = {"model": None, "encoders": None, "feature_columns": None, "info": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts telecom customer churn probability from account features.",
    version="1.0.0",
    lifespan=lifespan,
)


def load_artifacts() -> None:
    model_path = MODELS_DIR / "production_model.joblib"
    encoders_path = MODELS_DIR / "label_encoders.joblib"
    columns_path = MODELS_DIR / "feature_columns.json"
    info_path = MODELS_DIR / "production_model_info.json"

    if not model_path.exists():
        logger.warning("No production model found at %s — /predict will 503.", model_path)
        return

    _state["model"] = joblib.load(model_path)
    _state["encoders"] = joblib.load(encoders_path)
    with open(columns_path) as f:
        _state["feature_columns"] = json.load(f)
    with open(info_path) as f:
        _state["info"] = json.load(f)

    logger.info("Loaded model '%s' (run_id=%s)",
                _state["info"]["model_name"], _state["info"]["run_id"])


def _encode_input(payload: CustomerFeatures) -> pd.DataFrame:
    row = payload.model_dump()
    df = pd.DataFrame([row])

    encoders = _state["encoders"]
    for col, encoder in encoders.items():
        # Guard against unseen categories rather than raising a raw
        # sklearn ValueError back to the client.
        known = set(encoder.classes_)
        if df.at[0, col] not in known:
            raise HTTPException(
                status_code=422,
                detail=f"Unrecognized value '{df.at[0, col]}' for field '{col}'. "
                       f"Expected one of {sorted(known)}.",
            )
        df[col] = encoder.transform(df[col].astype(str))

    return df[_state["feature_columns"]]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _state["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=_state["info"]["model_name"] if loaded else None,
    )


@app.get("/model-info")
def model_info() -> dict:
    if _state["info"] is None:
        raise HTTPException(status_code=503, detail="No model loaded.")
    return _state["info"]


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    X = _encode_input(payload)
    proba = float(_state["model"].predict_proba(X)[0, 1])
    prediction = "Yes" if proba >= 0.5 else "No"

    logger.info("Prediction=%s proba=%.4f", prediction, proba)

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(proba, 4),
        model_name=_state["info"]["model_name"],
        model_version_run_id=_state["info"]["run_id"],
    )
