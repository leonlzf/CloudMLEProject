#!/usr/bin/env bash
set -euo pipefail

# Fixed project configuration.
REGION="ca-central-1"
BUCKET="cloudmle-ecg5000-artifacts-lzf-ca"
MODEL_FILE="ecg5000_binary_logreg_20260708_150506.pkl"

REPO_URL="https://github.com/leonlzf/CloudMLEProject.git"
APP_DIR="$HOME/CloudMLEProject"

IMAGE_NAME="ecg5000-api"
CONTAINER_NAME="ecg5000-api"

echo "Installing runtime dependencies..."
sudo dnf update -y
sudo dnf install -y git docker awscli

echo "Starting Docker..."
sudo systemctl enable docker
sudo systemctl start docker

# Pull the latest code if the repository already exists; otherwise clone it.
if [ -d "$APP_DIR/.git" ]; then
  echo "Pulling latest code..."
  cd "$APP_DIR"
  git pull
else
  echo "Cloning repository..."
  rm -rf "$APP_DIR"
  git clone "$REPO_URL" "$APP_DIR"
fi

# Download the model artifact from S3 into the local artifacts directory.
echo "Downloading model artifact from S3..."
mkdir -p "$APP_DIR/artifacts/models"

aws s3 cp \
  "s3://$BUCKET/models/$MODEL_FILE" \
  "$APP_DIR/artifacts/models/$MODEL_FILE" \
  --region "$REGION"

cd "$APP_DIR"

echo "Building Docker image..."
sudo docker build -t "$IMAGE_NAME" .

# Remove any existing container with the same name before starting a new one.
echo "Restarting container..."
sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

sudo docker run -d \
  --name "$CONTAINER_NAME" \
  -p 8000:8000 \
  -v "$APP_DIR/artifacts:/app/artifacts:ro" \
  "$IMAGE_NAME"

echo "Testing health endpoint..."
sleep 3
curl http://127.0.0.1:8000/health

echo
echo "Deployment complete."
