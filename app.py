"""FastAPI demo using a trained model from a public medical benchmark dataset."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.json"

try:
    MODEL: dict[str, Any] = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:  # fail fast if deployment artifact is missing/corrupt
    raise RuntimeError(f"Unable to load model artifact at {MODEL_PATH}: {exc}") from exc

FEATURE_NAMES: list[str] = MODEL["dataset"]["feature_names"]
INFERENCE = MODEL["inference"]
FEATURE_MEAN = INFERENCE["feature_mean"]
FEATURE_SCALE = INFERENCE["feature_scale"]
COEFFICIENTS = INFERENCE["coefficient_benign"]
INTERCEPT = INFERENCE["intercept_benign"]

app = FastAPI(
    title="Fast API / Flask App — Medical ML Demo",
    description=(
        "Educational API for a logistic-regression model trained on the public "
        "Wisconsin Diagnostic Breast Cancer benchmark. NOT for diagnosis or treatment."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    features: Annotated[
        list[float],
        Field(
            min_length=30,
            max_length=30,
            description=(
                "Exactly 30 positive numeric measurements in the order returned by GET /features. "
                "Use public benchmark/test samples only; do not submit personal patient records."
            ),
            examples=[[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                       1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                       25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]],
        ),
    ]

    @field_validator("features")
    @classmethod
    def valid_measurements(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(v) for v in values):
            raise ValueError("All feature values must be finite numbers")
        if not all(v > 0 for v in values):
            raise ValueError("All feature values must be positive measurements")
        return values


def _sigmoid(logit: float) -> float:
    """Numerically stable logistic sigmoid."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def predict_probabilities(features: list[float]) -> tuple[float, float]:
    scaled = [(value - mean) / scale for value, mean, scale in zip(features, FEATURE_MEAN, FEATURE_SCALE)]
    logit_benign = INTERCEPT + sum(coef * value for coef, value in zip(COEFFICIENTS, scaled))
    probability_benign = _sigmoid(logit_benign)
    return (1.0 - probability_benign, probability_benign)


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    return {
        "name": app.title,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "feature_order": "/features",
        "warning": "Educational demo only; not a diagnostic or treatment tool.",
    }


@app.get("/health", tags=["service"])
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": bool(MODEL), "model_version": MODEL["model_version"]}


@app.get("/model-info", tags=["model"])
def model_info() -> dict[str, Any]:
    """Expose dataset source, training approach, and holdout metrics."""
    return {
        "model_name": MODEL["model_name"],
        "model_version": MODEL["model_version"],
        "dataset": MODEL["dataset"],
        "training": MODEL["training"],
        "disclaimer": (
            "For software education and demonstration only. Holdout metrics describe this benchmark; "
            "they do not establish clinical safety, diagnostic accuracy for an individual, or suitability for care."
        ),
    }


@app.get("/features", tags=["model"])
def feature_order() -> dict[str, Any]:
    return {
        "count": len(FEATURE_NAMES),
        "features_in_required_order": FEATURE_NAMES,
        "units": "dataset-provided numerical measurements; consult the dataset documentation",
    }


@app.post("/predict", tags=["prediction"])
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Return a benchmark class and model scores, not a clinical diagnosis."""
    if len(request.features) != len(FEATURE_NAMES):
        raise HTTPException(status_code=422, detail=f"Exactly {len(FEATURE_NAMES)} values are required")
    malignant, benign = predict_probabilities(request.features)
    predicted_class = "benign" if benign >= malignant else "malignant"
    return {
        "predicted_benchmark_class": predicted_class,
        "model_probability_scores": {
            "malignant": round(malignant, 6),
            "benign": round(benign, 6),
        },
        "feature_count": len(request.features),
        "model_version": MODEL["model_version"],
        "interpretation": (
            "These are outputs from a model trained on a public research dataset, not personal medical risk estimates."
        ),
        "disclaimer": "Do not use for screening, diagnosis, treatment, or other medical decisions.",
    }
