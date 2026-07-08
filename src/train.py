from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.data import PROJECT_ROOT, load_ecg5000
except ModuleNotFoundError:
    from data import PROJECT_ROOT, load_ecg5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fast CPU baseline for ECG5000."
    )
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        default="binary",
        help="Use binary anomaly detection or original 5-class labels.",
    )
    parser.add_argument(
        "--normal-label",
        type=int,
        default=1,
        help="Label treated as normal for the binary task.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum Logistic Regression iterations.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models",
        help="Directory for saved model artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "reports",
        help="Directory for saved evaluation reports.",
    )
    parser.add_argument(
        "--experiment-name",
        default="ecg5000_baseline",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging.",
    )
    return parser.parse_args()


def make_labels(
    y: np.ndarray,
    *,
    task: str,
    normal_label: int,
) -> np.ndarray:
    if task == "binary":
        return (y != normal_label).astype(np.int64)
    return y.astype(np.int64)


def build_model(max_iter: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    *,
    task: str,
) -> dict[str, float]:
    average = "binary" if task == "binary" else "macro"
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if task == "binary" and y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return {key: float(value) for key, value in metrics.items()}


def predict_scores(model: Pipeline, X: np.ndarray, *, task: str) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X)
    if task == "binary" and probabilities.shape[1] == 2:
        return probabilities[:, 1]
    return None


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_to_mlflow(
    *,
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Pipeline,
    report_path: Path,
    model_path: Path,
) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("MLflow is not installed. Skipping MLflow logging.")
        return

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(model, artifact_path="model")


def main() -> None:
    args = parse_args()

    dataset = load_ecg5000()
    X_train, X_test = dataset.X_train, dataset.X_test
    y_train = make_labels(
        dataset.y_train,
        task=args.task,
        normal_label=args.normal_label,
    )
    y_test = make_labels(
        dataset.y_test,
        task=args.task,
        normal_label=args.normal_label,
    )

    model = build_model(max_iter=args.max_iter, random_state=args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test, task=args.task)
    metrics = compute_metrics(y_test, y_pred, y_score, task=args.task)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / f"ecg5000_{args.task}_logreg_{timestamp}.pkl"
    report_path = args.report_dir / f"ecg5000_{args.task}_report_{timestamp}.json"

    model_payload = {
        "model": model,
        "metadata": {
            "task": args.task,
            "normal_label": args.normal_label,
            "features": X_train.shape[1],
            "created_at": timestamp,
        },
    }
    with model_path.open("wb") as f:
        pickle.dump(model_payload, f)

    report = {
        "task": args.task,
        "normal_label": args.normal_label,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }
    save_json(report, report_path)

    params = {
        "task": args.task,
        "normal_label": args.normal_label,
        "model_type": "LogisticRegression",
        "class_weight": "balanced",
        "max_iter": args.max_iter,
        "random_state": args.random_state,
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "series_length": X_train.shape[1],
    }

    if not args.no_mlflow:
        log_to_mlflow(
            experiment_name=args.experiment_name,
            params=params,
            metrics=metrics,
            model=model,
            report_path=report_path,
            model_path=model_path,
        )

    print("Training complete.")
    print(f"Task: {args.task}")
    print(f"Model artifact: {model_path}")
    print(f"Report artifact: {report_path}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
