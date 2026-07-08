from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

try:
    from src.data import PROJECT_ROOT, load_ecg5000
    from src.train import compute_metrics, make_labels, predict_scores
except ModuleNotFoundError:
    from data import PROJECT_ROOT, load_ecg5000
    from train import compute_metrics, make_labels, predict_scores


DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "artifacts" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ECG5000 baseline model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a saved .pkl model artifact. Defaults to the latest model.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory used when searching for the latest saved model.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for evaluation JSON reports.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory for confusion matrix figures.",
    )
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        default=None,
        help="Override the task saved in model metadata.",
    )
    parser.add_argument(
        "--normal-label",
        type=int,
        default=None,
        help="Override the normal class label saved in model metadata.",
    )
    return parser.parse_args()


def find_latest_model(model_dir: Path) -> Path:
    candidates = sorted(
        model_dir.glob("*.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .pkl model artifacts found in {model_dir}. Run src/train.py first."
        )
    return candidates[0]


def load_model_artifact(model_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    with model_path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata", {})

    return payload, {}


def select_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_ecg5000()
    if split == "train":
        return dataset.X_train, dataset.y_train
    if split == "test":
        return dataset.X_test, dataset.y_test
    return dataset.X_all, dataset.y_all


def class_names(task: str) -> list[str]:
    if task == "binary":
        return ["normal", "abnormal"]
    return ["1", "2", "3", "4", "5"]


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=color,
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    model_path = args.model_path or find_latest_model(args.model_dir)
    model, metadata = load_model_artifact(model_path)

    task = args.task or metadata.get("task", "binary")
    normal_label = (
        args.normal_label
        if args.normal_label is not None
        else int(metadata.get("normal_label", 1))
    )

    X, y_raw = select_split(args.split)
    y_true = make_labels(y_raw, task=task, normal_label=normal_label)

    y_pred = model.predict(X)
    y_score = predict_scores(model, X, task=task)

    metrics = compute_metrics(y_true, y_pred, y_score, task=task)
    labels = class_names(task)
    matrix = confusion_matrix(y_true, y_pred)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = (
        args.report_dir / f"ecg5000_{task}_{args.split}_evaluation_{timestamp}.json"
    )
    figure_path = (
        args.figure_dir / f"ecg5000_{task}_{args.split}_confusion_matrix_{timestamp}.png"
    )

    plot_confusion_matrix(matrix, labels, figure_path)

    report = {
        "model_path": str(model_path),
        "model_metadata": metadata,
        "task": task,
        "normal_label": normal_label,
        "split": args.split,
        "shape": list(X.shape),
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix_figure": str(figure_path),
    }
    save_json(report, report_path)

    print("Evaluation complete.")
    print(f"Model artifact: {model_path}")
    print(f"Split: {args.split}")
    print(f"Report artifact: {report_path}")
    print(f"Confusion matrix figure: {figure_path}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
