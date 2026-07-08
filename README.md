# ECG Anomaly Detection MLOps Mini-Reproduction

一个面向练手的 ECG 时间序列异常检测小项目。项目重点不是追求模型指标或医学可用性，而是通过一个轻量数据集完整走一遍机器学习工程流程：数据读取、预处理、模型训练、MLflow 实验追踪、模型制品管理，以及基础 AWS 部署/存储实践。

## Resume-Style Summary

**ECG Anomaly Detection MLOps Practice | AWS + MLflow Mini Project**

- Reproduced a lightweight ECG time-series classification pipeline on the ECG5000 dataset, using 140-point single-lead signals as a compact practice case for anomaly detection and multi-class classification.
- Built a reproducible experiment workflow around data loading, normalization checks, train/test evaluation, metric logging, artifact tracking, and model version comparison with MLflow.
- Practiced cloud-oriented ML lifecycle management by organizing dataset/model artifacts for S3-style storage and preparing the project for AWS-based experiment tracking and deployment workflows.
- Treated model quality as secondary to engineering fluency, focusing on end-to-end MLOps familiarity: dataset inspection, baseline modeling, experiment logging, artifact management, and deployable service design.

## Project Goal

This repository is a small reproduction exercise inspired by ECG anomaly detection projects. The main goal is to become familiar with:

- ECG time-series dataset handling
- Basic anomaly/multi-class classification workflow
- MLflow tracking, runs, metrics, parameters, and artifacts
- AWS-oriented project structure, especially S3 artifact storage and deployable model packaging
- A practical end-to-end machine learning lifecycle

This is not intended to be a production medical model. The dataset is small, the training goal is intentionally modest, and the final model quality is not the main evaluation criterion.

## Dataset Analysis

The dataset folder contains the ECG5000 train/test split in three equivalent formats:

- `ECG5000_TRAIN.txt` / `ECG5000_TEST.txt`
- `ECG5000_TRAIN.arff` / `ECG5000_TEST.arff`
- `ECG5000_TRAIN.ts` / `ECG5000_TEST.ts`

The `.txt` files are the simplest format for this project. Each row contains:

- column 1: class label, from `1` to `5`
- columns 2-141: one ECG time series with `140` numeric time steps

The included metadata states that the dataset is univariate, equal-length, has no missing values, and uses five class labels. The local provenance note says: `ECG5000 provenance not determined yet`.

Raw dataset files are not committed to this repository. Place the expected ECG5000 files under `dataset/` before running the local pipeline.

### Split Summary

| Split | Samples | Features per Sample | Shape |
| --- | ---: | ---: | --- |
| Train | 500 | 140 | `(500, 141)` |
| Test | 4,500 | 140 | `(4500, 141)` |
| Total | 5,000 | 140 | `(5000, 141)` |

### Class Distribution

| Class | Train | Test | Total | Total % |
| --- | ---: | ---: | ---: | ---: |
| 1 | 292 | 2,627 | 2,919 | 58.38% |
| 2 | 177 | 1,590 | 1,767 | 35.34% |
| 3 | 10 | 86 | 96 | 1.92% |
| 4 | 19 | 175 | 194 | 3.88% |
| 5 | 2 | 22 | 24 | 0.48% |

The data is highly imbalanced. Classes `1` and `2` dominate the dataset, while classes `3`, `4`, and especially `5` are rare. For a real modeling task, this would require careful handling with class weights, stratified validation, threshold tuning, or anomaly-oriented evaluation. For this mini project, the imbalance is mainly useful because it gives a realistic reason to log per-class metrics and confusion matrices in MLflow.

### Value Statistics

| Split | Missing Values | Min | Max | Mean | Std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0 | -5.7976 | 4.0581 | 0.0000 | 0.9964 |
| Test | 0 | -7.0904 | 7.4021 | -0.0000 | 0.9964 |
| Total | 0 | -7.0904 | 7.4021 | -0.0000 | 0.9964 |

The signals appear already normalized, with near-zero mean and near-unit standard deviation. This makes the dataset convenient for fast baseline experiments.

## Suggested Workflow

1. Load `ECG5000_TRAIN.txt` and `ECG5000_TEST.txt`.
2. Split labels and time-series features.
3. Train a simple baseline model, such as Logistic Regression, 1D-CNN, LSTM, or a small PyTorch MLP.
4. Log each run with MLflow:
   - parameters: model type, learning rate, batch size, epochs
   - metrics: accuracy, precision, recall, F1, per-class F1
   - artifacts: confusion matrix, classification report, model file
5. Store model artifacts in an S3-compatible location.
6. Optionally deploy a small FastAPI inference service that loads the selected model artifact.

## Data Loading

The first reusable project module is implemented in `src/data.py`. It loads the local ECG5000 `.txt` files, validates the expected 140-point sequence length, checks for missing values, and returns train/test arrays.

```python
from src.data import load_ecg5000

dataset = load_ecg5000()

X_train, y_train = dataset.X_train, dataset.y_train
X_test, y_test = dataset.X_test, dataset.y_test
X_all, y_all = dataset.X_all, dataset.y_all
```

For quick verification from the command line:

```bash
python src/data.py
```

## Baseline Training

`src/train.py` implements a small CPU-friendly baseline for this practice project. By default, it converts the original 5-class labels into a binary anomaly task:

- `0`: normal ECG, original class `1`
- `1`: abnormal ECG, original classes `2`, `3`, `4`, and `5`

The model is intentionally simple: `StandardScaler` + `LogisticRegression(class_weight="balanced")`. This keeps training fast on a normal laptop and is enough for practicing the full MLOps flow.

Run a local training job without MLflow:

```bash
python src/train.py --no-mlflow
```

Run with MLflow logging enabled after installing the MLflow dependency:

```bash
python src/train.py
```

If MLflow is not installed yet, the script still trains the model and skips MLflow logging with a console message.

Training outputs are written under `artifacts/`:

- `artifacts/models/`: saved model pickle files
- `artifacts/reports/`: JSON metrics, classification report, and confusion matrix

These local artifacts are ignored by git and can be regenerated at any time.

## Evaluation

`src/evaluate.py` evaluates a saved model artifact independently from training. By default, it loads the latest `.pkl` file from `artifacts/models/` and evaluates it on the test split.

```bash
python src/evaluate.py
```

To evaluate a specific model:

```bash
python src/evaluate.py --model-path artifacts/models/<model-file>.pkl
```

To evaluate another split:

```bash
python src/evaluate.py --split train
python src/evaluate.py --split all
```

Evaluation outputs are written under:

- `artifacts/reports/`: evaluation metrics, classification report, and confusion matrix values
- `artifacts/figures/`: confusion matrix PNG files

## Local Inference

`src/inference.py` loads a saved model and predicts one ECG sequence. By default, it loads the latest model artifact and predicts `test[0]`.

```bash
python src/inference.py
```

To predict another sample from the local dataset:

```bash
python src/inference.py --split test --sample-index 2627
```

To use a specific model artifact:

```bash
python src/inference.py --model-path artifacts/models/<model-file>.pkl
```

The script prints a JSON response with the predicted label, readable class name, probabilities, model path, and true label when the sample comes from the local dataset.

## FastAPI Serving

`app/main.py` wraps the local inference logic in a small FastAPI service. It loads the latest model artifact from `artifacts/models/` by default. You can override the model path with the `MODEL_PATH` environment variable.

Start the API locally:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

- `GET /health`: check service and model loading status
- `GET /model`: show loaded model metadata
- `GET /sample?split=test&sample_index=0`: predict one sample from the local dataset
- `POST /predict`: predict one request body with exactly 140 ECG values

Prediction body shape:

```text
{
  "values": [v1, v2, ..., v140]
}
```

The real request must include exactly 140 values.

## AWS and MLflow Practice Scope

The project can be used to practice the following cloud MLOps components:

- **MLflow Tracking**: record experiments, parameters, metrics, artifacts, and model versions
- **S3**: store datasets, trained models, plots, and evaluation reports
- **EC2**: host a simple training job, MLflow server, or FastAPI inference service
- **RDS/PostgreSQL**: use as an MLflow backend store if practicing a more realistic setup
- **Docker**: package the training/inference environment for repeatable execution
- **CloudWatch**: optionally inspect logs for the deployed service

These components are intentionally kept small. The goal is to understand the workflow rather than build a large-scale production platform.

## Repository Structure

```text
.
+-- dataset/
|   +-- ECG5000.txt
|   +-- ECG5000_TRAIN.txt
|   +-- ECG5000_TEST.txt
|   +-- ECG5000_TRAIN.arff
|   +-- ECG5000_TEST.arff
|   +-- ECG5000_TRAIN.ts
|   +-- ECG5000_TEST.ts
+-- notebooks/
|   +-- 01_data_exploration.ipynb
+-- src/
|   +-- data.py
|   +-- train.py
|   +-- evaluate.py
|   +-- inference.py
+-- app/
|   +-- main.py
+-- requirements.txt
+-- .gitignore
+-- README.md
```

### Planned File Responsibilities

| Path | Purpose |
| --- | --- |
| `notebooks/01_data_exploration.ipynb` | Explore ECG waveforms, class distribution, and simple baseline ideas. |
| `src/data.py` | Load ECG5000 files and prepare features/labels. |
| `src/train.py` | Train baseline models and log experiments with MLflow. |
| `src/evaluate.py` | Generate metrics, classification reports, and confusion matrices. |
| `src/inference.py` | Load a saved model artifact and run local predictions. |
| `app/main.py` | Optional FastAPI inference service for AWS deployment practice. |
| `requirements.txt` | Python dependencies for local training, MLflow, and optional serving. |
| `.gitignore` | Ignore local runtime artifacts such as virtual environments and MLflow outputs. |

## Notes

- This repository currently focuses on the dataset and MLOps project framing.
- Reported model metrics should be treated as learning artifacts, not as claims of clinical performance.
- Because the minority classes are very small, aggregate accuracy alone is not a meaningful metric.
- The most valuable output of this project is a reproducible MLflow/AWS workflow, not a high-performing ECG classifier.
