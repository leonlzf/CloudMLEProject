# ECG Anomaly Detection MLOps Mini-Reproduction

一个面向练手的 ECG 时间序列异常检测小项目。项目重点不是追求模型指标或医学可用性，而是通过一个轻量数据集完整走一遍机器学习工程流程：数据读取、预处理、模型训练、MLflow 实验追踪、模型制品管理，以及基础 AWS 部署/存储实践。

## 简历式项目描述

**ECG Anomaly Detection MLOps Practice | AWS + MLflow Mini Project**

- 基于 ECG5000 数据集复现了一个轻量级 ECG 时间序列分类流程，将 140 个时间点的单导联 ECG 信号作为 anomaly detection 和 binary classification 的练习场景。
- 搭建了可复现的本地实验流程，覆盖数据读取、基础数据校验、train/test evaluation、metric logging、artifact 保存和模型版本对比等环节。
- 围绕 AWS 风格的 MLOps 流程组织项目结构，为后续接入 S3 model artifacts、EC2 deployment、MLflow Tracking 和 FastAPI inference service 做准备。
- 项目不以模型质量为核心目标，而是更关注端到端工程流程：dataset inspection、baseline modeling、evaluation report、local inference、API serving 和后续 cloud deployment。

## 项目目标

这个仓库是一个小型复现项目，用来练习 ECG anomaly detection 的完整机器学习工程流程。主要目标是熟悉：

- ECG time-series dataset 的读取和处理
- 基础 anomaly detection / binary classification 流程
- MLflow tracking、runs、metrics、parameters 和 artifacts
- 面向 AWS 的项目组织方式，尤其是 S3 artifact storage 和可部署的 model serving 结构
- 从数据到模型、从模型到服务的端到端 machine learning lifecycle

这个项目不是 production medical model。数据集较小，训练目标也刻意保持简单，最终模型分数不是主要评价标准。

## 数据集分析

`dataset/` 目录用于放置 ECG5000 的 train/test split。原始数据包含三种等价格式：

- `ECG5000_TRAIN.txt` / `ECG5000_TEST.txt`
- `ECG5000_TRAIN.arff` / `ECG5000_TEST.arff`
- `ECG5000_TRAIN.ts` / `ECG5000_TEST.ts`

本项目默认使用 `.txt` 文件，因为格式最直接。每一行表示一条 ECG 样本：

- 第 1 列：class label，取值为 `1` 到 `5`
- 第 2-141 列：长度为 `140` 的 ECG time series

数据集 metadata 显示它是 univariate、equal-length、无 missing values，并且包含 5 个类别。本地说明文件中写着：`ECG5000 provenance not determined yet`。

原始数据文件没有提交到 GitHub。运行本地 pipeline 前，需要把 ECG5000 文件放到 `dataset/` 目录下。

### 数据规模

| Split | Samples | Features per Sample | Shape |
| --- | ---: | ---: | --- |
| Train | 500 | 140 | `(500, 141)` |
| Test | 4,500 | 140 | `(4500, 141)` |
| Total | 5,000 | 140 | `(5000, 141)` |

### 类别分布

| Class | Train | Test | Total | Total % |
| --- | ---: | ---: | ---: | ---: |
| 1 | 292 | 2,627 | 2,919 | 58.38% |
| 2 | 177 | 1,590 | 1,767 | 35.34% |
| 3 | 10 | 86 | 96 | 1.92% |
| 4 | 19 | 175 | 194 | 3.88% |
| 5 | 2 | 22 | 24 | 0.48% |

数据类别非常不均衡。Class `1` 和 class `2` 占据绝大多数样本，而 class `3`、class `4`，尤其是 class `5` 样本很少。真实建模时需要考虑 class weight、stratified validation、threshold tuning 或 anomaly-oriented evaluation。这个 mini project 里，这种不均衡主要用来练习记录 per-class metrics 和 confusion matrix。

### 数值统计

| Split | Missing Values | Min | Max | Mean | Std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0 | -5.7976 | 4.0581 | 0.0000 | 0.9964 |
| Test | 0 | -7.0904 | 7.4021 | -0.0000 | 0.9964 |
| Total | 0 | -7.0904 | 7.4021 | -0.0000 | 0.9964 |

从统计结果看，信号基本已经接近标准化，整体均值接近 0，标准差接近 1。因此这个数据集很适合快速跑 baseline experiment。

## 建议流程

1. 读取 `ECG5000_TRAIN.txt` 和 `ECG5000_TEST.txt`。
2. 拆分 labels 和 time-series features。
3. 先训练一个简单 baseline model，例如 Logistic Regression、1D-CNN、LSTM 或小型 PyTorch MLP。
4. 用 MLflow 记录每次实验：
   - parameters: model type、learning rate、batch size、epochs
   - metrics: accuracy、precision、recall、F1、per-class F1
   - artifacts: confusion matrix、classification report、model file
5. 将 model artifacts 保存到 S3-compatible location。
6. 可选：部署一个小型 FastAPI inference service，加载选定的 model artifact 做预测。

## Data Loading

第一个可复用模块是 `src/data.py`。它负责读取本地 ECG5000 `.txt` 文件，校验每条样本是否为 140 个时间点，检查 missing values，并返回 train/test arrays。

```python
from src.data import load_ecg5000

dataset = load_ecg5000()

X_train, y_train = dataset.X_train, dataset.y_train
X_test, y_test = dataset.X_test, dataset.y_test
X_all, y_all = dataset.X_all, dataset.y_all
```

命令行快速检查：

```bash
python src/data.py
```

## Baseline Training

`src/train.py` 实现了一个适合 CPU 快速运行的 baseline。默认会把原始 5 分类标签转换成 binary anomaly task：

- `0`: normal ECG，对应原始 class `1`
- `1`: abnormal ECG，对应原始 class `2`、`3`、`4`、`5`

模型刻意保持简单：`StandardScaler` + `LogisticRegression(class_weight="balanced")`。这样普通 laptop 也可以很快训练完成，同时足够支撑后面的 MLOps 全流程练习。

不启用 MLflow 的本地训练：

```bash
python src/train.py --no-mlflow
```

安装 MLflow 后启用 MLflow logging：

```bash
python src/train.py
```

如果当前环境还没有安装 MLflow，脚本仍然会完成模型训练，只是会在控制台提示并跳过 MLflow logging。

训练产物会写入 `artifacts/`：

- `artifacts/models/`: 保存的 model pickle files
- `artifacts/reports/`: JSON metrics、classification report 和 confusion matrix

这些本地 artifacts 已经被 `.gitignore` 忽略，可以随时重新生成。

## Evaluation

`src/evaluate.py` 用来独立评估已经保存的 model artifact。默认会从 `artifacts/models/` 中加载最新的 `.pkl` 文件，并在 test split 上评估。

```bash
python src/evaluate.py
```

指定某一个模型：

```bash
python src/evaluate.py --model-path artifacts/models/<model-file>.pkl
```

评估其他 split：

```bash
python src/evaluate.py --split train
python src/evaluate.py --split all
```

评估产物会写入：

- `artifacts/reports/`: evaluation metrics、classification report 和 confusion matrix values
- `artifacts/figures/`: confusion matrix PNG files

## Local Inference

`src/inference.py` 用来加载保存好的模型，并对单条 ECG sequence 做本地预测。默认加载最新 model artifact，并预测 `test[0]`。

```bash
python src/inference.py
```

预测本地数据集里的其他样本：

```bash
python src/inference.py --split test --sample-index 2627
```

指定某一个 model artifact：

```bash
python src/inference.py --model-path artifacts/models/<model-file>.pkl
```

脚本会输出 JSON response，包含 predicted label、readable class name、probabilities、model path，以及来自本地数据集时的 true label。

## FastAPI Serving

`app/main.py` 将本地 inference 逻辑包装成一个小型 FastAPI service。默认从 `artifacts/models/` 加载最新 model artifact，也可以通过 `MODEL_PATH` environment variable 指定模型路径。

本地启动 API：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

关闭本地 API：

```bash
Ctrl+C
```

如果服务是用后台进程启动的，可以先找到 PID，再关闭：

```powershell
Get-Process | Where-Object { $_.ProcessName -like "*python*" }
Stop-Process -Id <PID>
```

主要 endpoints：

- `GET /health`: 检查 service 状态和 model loading status
- `GET /model`: 查看当前加载的 model metadata
- `GET /sample?split=test&sample_index=0`: 从本地数据集中取一个样本并预测
- `POST /predict`: 对请求体中的一条 140 维 ECG sequence 做预测

Prediction body shape:

```text
{
  "values": [v1, v2, ..., v140]
}
```

真实请求必须包含正好 140 个数值。

## Docker Serving

项目提供了一个轻量 `Dockerfile`，用于把 FastAPI service 打包成可部署的 container image。镜像只包含代码和 Python dependencies，不内置本地 `dataset/` 或 `artifacts/`。

Build image:

```bash
docker build -t ecg5000-api .
```

本地运行 API，并挂载已有的模型和数据目录：

```bash
docker run --rm -p 8000:8000 -v ${PWD}/artifacts:/app/artifacts:ro -v ${PWD}/dataset:/app/dataset:ro ecg5000-api
```

启动后可以访问：

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/sample?split=test&sample_index=2627"
```

如果只想使用 `POST /predict`，容器只需要能加载 model artifact；`/sample` endpoint 额外依赖挂载后的 `dataset/`。

## AWS S3 + EC2 Deployment

本项目已经跑通一个最小 AWS deployment 闭环：

```text
Local training
-> MLflow tracking
-> model/report artifacts
-> S3 artifact storage
-> EC2 pulls code from GitHub
-> EC2 downloads model from S3
-> Dockerized FastAPI serving
-> public API inference
```

实际测试使用的区域：

```text
ca-central-1
```

### 1. S3 Artifact Storage

本地训练完成后，模型文件位于：

```text
artifacts/models/ecg5000_binary_logreg_20260708_150506.pkl
```

训练报告位于：

```text
artifacts/reports/ecg5000_binary_report_20260708_150506.json
```

创建 S3 bucket：

```powershell
aws s3 mb s3://cloudmle-ecg5000-artifacts-lzf-ca --region ca-central-1
```

上传 model artifact：

```powershell
aws s3 cp artifacts\models\ecg5000_binary_logreg_20260708_150506.pkl s3://cloudmle-ecg5000-artifacts-lzf-ca/models/ --region ca-central-1
```

上传 report artifact：

```powershell
aws s3 cp artifacts\reports\ecg5000_binary_report_20260708_150506.json s3://cloudmle-ecg5000-artifacts-lzf-ca/reports/ --region ca-central-1
```

确认 S3 中的 artifacts：

```powershell
aws s3 ls s3://cloudmle-ecg5000-artifacts-lzf-ca/models/ --region ca-central-1
aws s3 ls s3://cloudmle-ecg5000-artifacts-lzf-ca/reports/ --region ca-central-1
```

如果使用 AWS SSO profile，每条 `aws` 命令后追加：

```powershell
--profile <profile-name>
```

### 2. EC2 Setup

EC2 配置：

```text
AMI: Amazon Linux 2023
Instance type: t2.micro
Storage: 18 GiB gp3
Region: ca-central-1
Security group: cloudmle-ecg-api-sg
```

Security group inbound rules：

```text
SSH         TCP  22    My IP
Custom TCP  TCP  8000  My IP
```

Outbound rule 保持默认即可：

```text
All traffic -> 0.0.0.0/0
```

建议给 EC2 绑定 IAM role：

```text
Role name: cloudmle-ec2-s3-readonly-role
Permission: AmazonS3ReadOnlyAccess
Use case: EC2
```

这样 EC2 可以从 S3 下载模型，而不需要在服务器上保存 AWS access key。

### 3. Install Runtime on EC2

SSH 进入 EC2 后，更新系统并安装依赖：

```bash
sudo dnf update -y
sudo dnf install -y git docker awscli
```

启动 Docker：

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

将 `ec2-user` 加入 Docker group：

```bash
sudo usermod -aG docker ec2-user
```

执行后退出并重新 SSH，让 group 权限生效：

```bash
exit
```

重新登录后验证：

```bash
git --version
docker --version
aws --version
docker ps
```

### 4. Clone Repo and Download Model

在 EC2 上 clone GitHub repo：

```bash
git clone https://github.com/leonlzf/CloudMLEProject.git
cd CloudMLEProject
```

创建模型目录：

```bash
mkdir -p artifacts/models
```

确认 EC2 可以读取 S3：

```bash
aws s3 ls s3://cloudmle-ecg5000-artifacts-lzf-ca/models/ --region ca-central-1
```

从 S3 下载模型：

```bash
aws s3 cp s3://cloudmle-ecg5000-artifacts-lzf-ca/models/ecg5000_binary_logreg_20260708_150506.pkl artifacts/models/ --region ca-central-1
```

确认模型文件存在：

```bash
ls artifacts/models
```

### 5. Build and Run FastAPI Container

在 EC2 的项目目录中 build Docker image：

```bash
docker build -t ecg5000-api .
```

运行 container，并把模型目录挂载到容器内：

```bash
docker run -d --name ecg5000-api -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts:ro ecg5000-api
```

检查 container：

```bash
docker ps
docker logs ecg5000-api
```

在 EC2 内部测试：

```bash
curl http://127.0.0.1:8000/health
```

预期返回：

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/artifacts/models/ecg5000_binary_logreg_20260708_150506.pkl"
}
```

### 6. Test Public API

从本地浏览器或 Postman 访问 EC2 public IP：

```text
http://<EC2_PUBLIC_IP>:8000/health
```

测试 inference endpoint：

```text
POST http://<EC2_PUBLIC_IP>:8000/predict
```

Request body：

```json
{
  "values": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

返回 `200 OK` 且包含 `predicted_name`、`probabilities`、`model_path`，即表示 public API inference 成功。

如果 EC2 内部 `curl` 成功，但本地访问失败，优先检查：

- 是否使用 EC2 **Public IPv4 address**
- Security group 是否开放 `Custom TCP 8000` 给当前 My IP
- Docker container 是否仍在运行

### 7. Stop and Cleanup

停止并删除 container：

```bash
docker stop ecg5000-api
docker rm ecg5000-api
```

如果只是暂时不用服务，在 AWS Console 中 stop EC2 instance：

```text
EC2 -> Instances -> Instance state -> Stop instance
```

不要随手 terminate，除非确定以后不再需要该 instance。

### 8. Automated EC2 Deployment Script

项目中提供了一个简单的 EC2 deployment script：

```text
scripts/deploy_ec2.sh
```

这个脚本用于在 Amazon Linux 2023 EC2 上自动复现上面的部署流程。它会执行：

- 安装 `git`、`docker`、`awscli`
- 启动 Docker service
- clone 或 pull GitHub repo
- 从 S3 下载指定 model artifact
- build `ecg5000-api` Docker image
- 停止并删除旧的 `ecg5000-api` container
- 启动新的 FastAPI container
- 调用 `http://127.0.0.1:8000/health` 做健康检查

脚本默认配置：

```bash
REGION="ca-central-1"
BUCKET="cloudmle-ecg5000-artifacts-lzf-ca"
MODEL_FILE="ecg5000_binary_logreg_20260708_150506.pkl"
REPO_URL="https://github.com/leonlzf/CloudMLEProject.git"
```

在 EC2 上运行：

```bash
cd ~/CloudMLEProject
git pull
bash scripts/deploy_ec2.sh
```

如果 EC2 上还没有 clone repo，可以先执行：

```bash
git clone https://github.com/leonlzf/CloudMLEProject.git
cd CloudMLEProject
bash scripts/deploy_ec2.sh
```

运行前需要确认：

- EC2 使用 Amazon Linux 2023
- EC2 已绑定可以读取 S3 的 IAM role
- S3 中存在脚本配置的 `MODEL_FILE`
- Security group 已开放 `8000` 给当前 My IP

## AWS and MLflow Practice Scope

这个项目后续可以继续练习以下 cloud MLOps 组件：

- **MLflow Tracking**: 记录 experiments、parameters、metrics、artifacts 和 model versions
- **S3**: 存储 datasets、trained models、plots 和 evaluation reports
- **EC2**: 托管 training job、MLflow server 或 FastAPI inference service
- **RDS/PostgreSQL**: 作为 MLflow backend store，练习更接近真实环境的实验管理
- **Docker**: 打包 training/inference environment，保证可重复运行
- **CloudWatch**: 查看 deployed service 的基础 logs

这些组件会保持小而清晰。这个项目的目标是理解 workflow，而不是搭建大型 production platform。

## Repository Structure

```text
.
+-- dataset/
|   +-- README.md
|   +-- ECG5000_TRAIN.txt
|   +-- ECG5000_TEST.txt
|   +-- ...
+-- notebooks/
|   +-- 01_data_exploration.ipynb
+-- src/
|   +-- data.py
|   +-- train.py
|   +-- evaluate.py
|   +-- inference.py
+-- app/
|   +-- main.py
+-- Dockerfile
+-- .dockerignore
+-- requirements.txt
+-- .gitignore
+-- README.md
```

### 文件职责

| Path | Purpose |
| --- | --- |
| `notebooks/01_data_exploration.ipynb` | 探索 ECG waveform、class distribution 和基础 baseline 思路。 |
| `src/data.py` | 读取 ECG5000 文件并准备 features/labels。 |
| `src/train.py` | 训练 baseline model，并可选接入 MLflow logging。 |
| `src/evaluate.py` | 生成 metrics、classification report 和 confusion matrix。 |
| `src/inference.py` | 加载保存好的 model artifact 并执行本地单条预测。 |
| `app/main.py` | FastAPI inference service，用于后续 AWS deployment practice。 |
| `Dockerfile` | 构建 FastAPI service 的 container image。 |
| `.dockerignore` | 控制 Docker build context，排除 dataset、artifacts 和本地缓存。 |
| `requirements.txt` | 本地 training、MLflow 和 serving 所需 Python dependencies。 |
| `.gitignore` | 忽略本地 runtime artifacts，例如 virtual environment、dataset、MLflow outputs 和 model artifacts。 |

## Notes

- 这个仓库当前聚焦本地 dataset、baseline model 和 MLOps project framing。
- 模型指标只应被视为学习过程中的 artifacts，不代表任何 clinical performance。
- 少数类样本非常少，因此只看 accuracy 并不可靠。
- 这个项目最有价值的产出是可复现的 MLflow/AWS workflow，而不是高性能 ECG classifier。
