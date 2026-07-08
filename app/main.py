from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.data import load_ecg5000
from src.inference import (
    DEFAULT_MODEL_DIR,
    EXPECTED_SERIES_LENGTH,
    find_latest_model,
    label_name,
    load_model_artifact,
    predict_one,
)


class PredictionRequest(BaseModel):
    values: list[float] = Field(
        ...,
        description="One ECG sequence with exactly 140 numeric values.",
    )


class PredictionResponse(BaseModel):
    predicted_label: int
    predicted_name: str
    probabilities: dict[str, float] | None = None
    task: str
    model_path: str
    source: str
    true_original_label: int | None = None
    true_binary_label: int | None = None
    true_name: str | None = None


class ModelState:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.model_path: Path | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.model_path is not None


state = ModelState()
app = FastAPI(
    title="ECG5000 Anomaly Detection API",
    version="0.1.0",
    description="Small FastAPI wrapper around the local ECG5000 baseline model.",
)


def resolve_model_path() -> Path:
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        return Path(configured_path)
    return find_latest_model(DEFAULT_MODEL_DIR)


def load_model() -> None:
    model_path = resolve_model_path()
    model, metadata = load_model_artifact(model_path)
    state.model = model
    state.metadata = metadata
    state.model_path = model_path


def get_loaded_model() -> tuple[Any, dict[str, Any], Path]:
    if not state.is_loaded:
        try:
            load_model()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    assert state.model is not None
    assert state.model_path is not None
    return state.model, state.metadata, state.model_path


def validate_sequence(values: list[float]) -> np.ndarray:
    sequence = np.asarray(values, dtype=np.float32)
    if sequence.shape != (EXPECTED_SERIES_LENGTH,):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected exactly {EXPECTED_SERIES_LENGTH} ECG values, "
                f"got {sequence.shape[0] if sequence.ndim == 1 else sequence.shape}."
            ),
        )
    return sequence


def attach_metadata(
    result: dict[str, Any],
    *,
    model_path: Path,
    task: str,
    source: str,
) -> PredictionResponse:
    payload = {
        **result,
        "task": task,
        "model_path": str(model_path),
        "source": source,
    }
    return PredictionResponse(**payload)


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_model()
    except FileNotFoundError:
        # The API can still start before training; /health will report unloaded.
        pass


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": state.is_loaded,
        "model_path": str(state.model_path) if state.model_path else None,
    }


@app.get("/model")
def model_info() -> dict[str, Any]:
    _, metadata, model_path = get_loaded_model()
    return {
        "model_path": str(model_path),
        "metadata": metadata,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model, metadata, model_path = get_loaded_model()
    task = metadata.get("task", "binary")
    sequence = validate_sequence(request.values)
    result = predict_one(model, sequence, task=task)
    return attach_metadata(
        result,
        model_path=model_path,
        task=task,
        source="request_body",
    )


@app.get("/sample", response_model=PredictionResponse)
def predict_sample(
    split: str = Query(default="test", pattern="^(train|test)$"),
    sample_index: int = Query(default=0, ge=0),
) -> PredictionResponse:
    model, metadata, model_path = get_loaded_model()
    task = metadata.get("task", "binary")
    normal_label = int(metadata.get("normal_label", 1))

    dataset = load_ecg5000()
    if split == "train":
        X, y = dataset.X_train, dataset.y_train
    else:
        X, y = dataset.X_test, dataset.y_test

    if sample_index >= len(X):
        raise HTTPException(
            status_code=404,
            detail=f"sample_index must be between 0 and {len(X) - 1}.",
        )

    sequence = X[sample_index]
    true_original_label = int(y[sample_index])
    result = predict_one(model, sequence, task=task)

    result["true_original_label"] = true_original_label
    if task == "binary":
        true_binary_label = int(true_original_label != normal_label)
        result["true_binary_label"] = true_binary_label
        result["true_name"] = label_name(true_binary_label, task)
    else:
        result["true_name"] = label_name(true_original_label, task)

    return attach_metadata(
        result,
        model_path=model_path,
        task=task,
        source=f"{split}[{sample_index}]",
    )
