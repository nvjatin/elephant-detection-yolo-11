#!/bin/bash
# Setup script for Elephant Detection on Apple Silicon
# Ensures all operations use external SSD

set -e

# Configuration
SSD_PATH="/Volumes/Extended Storage/Elephant-Detection"

echo "=============================================="
echo "Elephant Detection - Apple Silicon Setup"
echo "=============================================="

# Check if SSD is mounted
if [ ! -d "$SSD_PATH" ]; then
    echo "ERROR: External SSD not mounted at: $SSD_PATH"
    echo "Please connect your external storage and try again."
    exit 1
fi

echo "✓ External SSD found at: $SSD_PATH"

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p "$SSD_PATH/datasets/kaggle"
mkdir -p "$SSD_PATH/datasets/roboflow"
mkdir -p "$SSD_PATH/datasets/openimages"
mkdir -p "$SSD_PATH/datasets/custom"
mkdir -p "$SSD_PATH/models/pretrained"
mkdir -p "$SSD_PATH/models/trained"
mkdir -p "$SSD_PATH/outputs/videos"
mkdir -p "$SSD_PATH/outputs/detections"
mkdir -p "$SSD_PATH/outputs/training"
mkdir -p "$SSD_PATH/logs"
mkdir -p "$SSD_PATH/config"
mkdir -p "$SSD_PATH/scripts"
mkdir -p "$SSD_PATH/backend"
mkdir -p "$SSD_PATH/frontend"

echo "✓ Directory structure created"

# Navigate to project
cd "$SSD_PATH"

# Create Python virtual environment on SSD
if [ ! -d "$SSD_PATH/venv" ]; then
    echo ""
    echo "Creating virtual environment on SSD..."
    python3 -m venv "$SSD_PATH/venv"
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source "$SSD_PATH/venv/bin/activate"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify MPS support
echo ""
echo "Verifying Apple Silicon (MPS) support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✓ MPS (Metal Performance Shaders) is ready!')
else:
    print('✗ MPS not available - will use CPU')
"

# Download YOLOv8n to SSD
echo ""
echo "Downloading YOLOv8n model to SSD..."
python3 -c "
from ultralytics import YOLO
from pathlib import Path
import shutil

model = YOLO('yolov8n.pt')
dest = Path('$SSD_PATH/models/pretrained/yolov8n.pt')
dest.parent.mkdir(parents=True, exist_ok=True)

# Find and copy the downloaded model
import ultralytics
model_path = Path(ultralytics.__file__).parent / 'yolov8n.pt'
if not model_path.exists():
    # Check common download locations
    from pathlib import Path
    home = Path.home()
    possible_paths = [
        home / '.cache' / 'ultralytics' / 'yolov8n.pt',
        Path('yolov8n.pt'),
    ]
    for p in possible_paths:
        if p.exists():
            model_path = p
            break

if model_path.exists() and not dest.exists():
    shutil.copy2(model_path, dest)
    print(f'✓ Model saved to: {dest}')
elif dest.exists():
    print(f'✓ Model already exists: {dest}')
else:
    print('Model will be downloaded on first use')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Project location: $SSD_PATH"
echo ""
echo "Activate environment:"
echo "  source \"$SSD_PATH/venv/bin/activate\""
echo ""
echo "Run detection:"
echo "  python detect.py --mode webcam"
echo "  python detect.py --mode video --input video.mp4"
echo ""
echo "Download datasets:"
echo "  python scripts/download_dataset.py --source kaggle --dataset <name>"
echo ""
echo "Train model:"
echo "  python scripts/train.py --dataset <name> --epochs 100"
echo ""
