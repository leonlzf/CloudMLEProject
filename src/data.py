from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset"
EXPECTED_SERIES_LENGTH = 140


@dataclass(frozen=True)
class ECG5000Data:
    """Container for the ECG5000 train/test split."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def X_all(self) -> np.ndarray:
        return np.vstack([self.X_train, self.X_test])

    @property
    def y_all(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_test])


def load_ecg5000_txt(
    path: str | Path,
    *,
    zero_based_labels: bool = False,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one ECG5000 .txt file.

    The first column is the class label and the remaining 140 columns are the
    ECG time-series values.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ECG5000 file not found: {path}")

    data = np.loadtxt(path, dtype=dtype)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array from {path}, got shape {data.shape}")

    expected_columns = EXPECTED_SERIES_LENGTH + 1
    if data.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in {path}, got {data.shape[1]}"
        )

    if np.isnan(data).any():
        raise ValueError(f"Found missing values in {path}")

    y = data[:, 0].astype(np.int64)
    X = data[:, 1:].astype(dtype, copy=False)

    if zero_based_labels:
        y = y - 1

    return X, y


def load_ecg5000(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    zero_based_labels: bool = False,
    dtype: np.dtype = np.float32,
) -> ECG5000Data:
    """Load the ECG5000 train/test split from the local dataset directory."""

    data_dir = Path(data_dir)
    train_path = data_dir / "ECG5000_TRAIN.txt"
    test_path = data_dir / "ECG5000_TEST.txt"

    X_train, y_train = load_ecg5000_txt(
        train_path,
        zero_based_labels=zero_based_labels,
        dtype=dtype,
    )
    X_test, y_test = load_ecg5000_txt(
        test_path,
        zero_based_labels=zero_based_labels,
        dtype=dtype,
    )

    return ECG5000Data(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def summarize_split(name: str, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Return basic shape and value statistics for one data split."""

    return {
        "split": name,
        "samples": int(len(y)),
        "series_length": int(X.shape[1]),
        "classes": sorted(np.unique(y).astype(int).tolist()),
        "missing_values": int(np.isnan(X).sum()),
        "min": float(np.min(X)),
        "max": float(np.max(X)),
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
    }


def class_distribution(y: np.ndarray) -> dict[int, int]:
    """Return class counts as a plain dictionary."""

    labels, counts = np.unique(y, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(labels.tolist(), counts.tolist())
    }


def print_dataset_summary(dataset: ECG5000Data) -> None:
    """Print a small human-readable summary for quick local checks."""

    splits = [
        ("train", dataset.X_train, dataset.y_train),
        ("test", dataset.X_test, dataset.y_test),
        ("all", dataset.X_all, dataset.y_all),
    ]

    for name, X, y in splits:
        summary = summarize_split(name, X, y)
        print(f"{name}:")
        print(f"  shape: X={X.shape}, y={y.shape}")
        print(f"  classes: {summary['classes']}")
        print(f"  missing_values: {summary['missing_values']}")
        print(
            "  values: "
            f"min={summary['min']:.4f}, "
            f"max={summary['max']:.4f}, "
            f"mean={summary['mean']:.4f}, "
            f"std={summary['std']:.4f}"
        )
        print(f"  class_distribution: {class_distribution(y)}")


if __name__ == "__main__":
    ecg5000 = load_ecg5000()
    print_dataset_summary(ecg5000)
