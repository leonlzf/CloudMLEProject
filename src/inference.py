from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

try:
    from src.data import PROJECT_ROOT, load_ecg5000
except ModuleNotFoundError:
    from data import PROJECT_ROOT, load_ecg5000


DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
EXPECTED_SERIES_LENGTH = 140


# 解析命令行参数，支持指定模型路径、数据 split、样本 index 或外部 JSON 输入。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local inference with a saved ECG5000 model."
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
        choices=["train", "test"],
        default="test",
        help="Dataset split to sample from when no input file is provided.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index used from the selected split.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Optional JSON file containing one 140-value ECG sequence.",
    )
    return parser.parse_args()


# 在模型目录中查找最近修改的 .pkl 文件，作为默认推理模型。
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


# 从 pickle artifact 中加载 sklearn pipeline 和训练时保存的 metadata。
def load_model_artifact(model_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    with model_path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata", {})

    return payload, {}


# 从 JSON 文件中读取一条 140 维 ECG 序列，并校验输入长度。
def load_sequence_from_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        values = payload.get("values") or payload.get("ecg") or payload.get("signal")
    else:
        values = payload

    sequence = np.asarray(values, dtype=np.float32)
    if sequence.shape != (EXPECTED_SERIES_LENGTH,):
        raise ValueError(
            f"Expected one ECG sequence with shape ({EXPECTED_SERIES_LENGTH},), "
            f"got {sequence.shape}"
        )
    return sequence


# 从本地 ECG5000 train/test split 中取出一条样本及其原始标签。
def load_sequence_from_dataset(split: str, sample_index: int) -> tuple[np.ndarray, int]:
    dataset = load_ecg5000()
    if split == "train":
        X, y = dataset.X_train, dataset.y_train
    else:
        X, y = dataset.X_test, dataset.y_test

    if sample_index < 0 or sample_index >= len(X):
        raise IndexError(
            f"sample_index must be between 0 and {len(X) - 1}, got {sample_index}"
        )

    return X[sample_index], int(y[sample_index])


# 将模型输出的数字标签转换成更易读的类别名称。
def label_name(label: int, task: str) -> str:
    if task == "binary":
        return "abnormal" if label == 1 else "normal"
    return str(label)


# 对单条 ECG 序列执行推理，并返回预测标签、类别名称和概率。
def predict_one(
    model: Pipeline,
    sequence: np.ndarray,
    *,
    task: str,
) -> dict[str, Any]:
    X = sequence.reshape(1, -1)
    predicted_label = int(model.predict(X)[0])

    result: dict[str, Any] = {
        "predicted_label": predicted_label,
        "predicted_name": label_name(predicted_label, task),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = [int(value) for value in model.classes_.tolist()]
        result["probabilities"] = {
            label_name(label, task): float(probability)
            for label, probability in zip(classes, probabilities.tolist())
        }

    return result


# 命令行入口：加载模型、读取输入样本、执行预测并打印 JSON 结果。
def main() -> None:
    args = parse_args()

    model_path = args.model_path or find_latest_model(args.model_dir)
    model, metadata = load_model_artifact(model_path)
    task = metadata.get("task", "binary")

    true_label = None
    if args.input_json:
        sequence = load_sequence_from_json(args.input_json)
        source = str(args.input_json)
    else:
        sequence, true_label = load_sequence_from_dataset(args.split, args.sample_index)
        source = f"{args.split}[{args.sample_index}]"

    result = predict_one(model, sequence, task=task)
    result.update(
        {
            "model_path": str(model_path),
            "source": source,
            "task": task,
        }
    )

    if true_label is not None:
        result["true_original_label"] = true_label
        if task == "binary":
            result["true_binary_label"] = int(true_label != metadata.get("normal_label", 1))
            result["true_name"] = label_name(result["true_binary_label"], task)
        else:
            result["true_name"] = label_name(true_label, task)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
