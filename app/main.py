"""FastAPI prediction service for MP022 (hostel mess food waste).

POST /predict  -- one meal's features in, predicted waste_kg out.
GET  /health   -- {"status": "ok"}.

The client's two rules (PROBLEM.md section 5) live in the pydantic schema, so
a bad request is rejected with 422 and never reaches the model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.logging_config import configure_logging
from mess_waste.config import load_params
from mess_waste.data_checks import normalise_label

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

PARAMS = load_params()
NUMERIC_FEATURES = PARAMS["data"]["numeric_features"]
CATEGORICAL_FEATURES = PARAMS["data"]["categorical_features"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

logger = configure_logging()

app = FastAPI(
    title="MP022 Hostel Mess Food Waste API",
    description="Predicts waste_kg for one meal at one hostel.",
    version="1.0",
)


def _load_model() -> Any:
    """Load the champion model exported by mess_waste.export_model."""
    if not MODEL_PATH.exists():
        logger.error("model file not found", extra={"fields": {"path": str(MODEL_PATH)}})
        return None
    model = joblib.load(MODEL_PATH)
    logger.info("model loaded", extra={"fields": {"path": str(MODEL_PATH)}})
    return model


MODEL = _load_model()


class PredictRequest(BaseModel):
    """Exactly the eight fields the client sends.

    Rule 1: temp_c above 75 is impossible      -> le=75.
    Rule 2: cooked_kg below 25 is a typo       -> ge=25.
    """

    model_config = ConfigDict(extra="forbid")

    students_registered: int = Field(..., ge=0, description="students registered for the mess")
    attendance_pct: float = Field(..., ge=0, le=100, description="share of registered students who ate")
    menu_rating: float = Field(..., ge=0, le=5, description="average rating of the dish, in stars")
    temp_c: float = Field(..., le=75, description="temperature at meal time in deg C; above 75 is rejected")
    cooked_kg: float = Field(..., ge=25, description="food cooked in kg; below 25 is rejected")
    plate_size_cm: float = Field(..., gt=0, description="plate diameter in cm")
    meal: str = Field(..., min_length=1, description="breakfast, lunch or dinner")
    hostel: str = Field(..., min_length=1, description="Boys Hostel 1, Boys Hostel 2 or Girls Hostel")


class PredictResponse(BaseModel):
    """The predicted waste in kg."""

    prediction: float
    waste_kg: float
    unit: str = "kg"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def to_frame(request: PredictRequest) -> pd.DataFrame:
    """One request -> a one-row frame with the columns the model was trained on.

    The model was trained on cleaned data whose text labels went through
    normalise_label ('Boys Hostel 2' -> 'boys hostel 2'), so the same
    normalisation is applied here. Without it the one-hot encoder sees an
    unknown category and silently drops the signal.
    """
    row = request.model_dump()
    for column in CATEGORICAL_FEATURES:
        row[column] = normalise_label(row[column])
    return pd.DataFrame([row], columns=FEATURE_ORDER)


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    """Liveness and readiness probe."""
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    """Predict waste_kg for one meal."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    started = time.perf_counter()
    frame = to_frame(request)
    prediction = float(MODEL.predict(frame)[0])
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

    # One JSON line per prediction, on stdout. No personal details exist in
    # this payload: the features describe a meal, not a student.
    logger.info(
        "prediction",
        extra={
            "fields": {
                "path": "/predict",
                "prediction_kg": round(prediction, 4),
                "latency_ms": elapsed_ms,
                "features": request.model_dump(),
            }
        },
    )

    return {"prediction": prediction, "waste_kg": prediction, "unit": "kg"}
