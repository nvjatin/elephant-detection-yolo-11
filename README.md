# Elephant Detection

A YOLOv11-based computer vision project for detecting elephants in images and video.  
This project supports training, validation, prediction, and export workflows using a custom elephant dataset.

## Features

- Train an elephant detector with YOLOv11m
- Validate model performance on the dataset
- Run inference on images, videos, or webcam input
- Use NVIDIA CUDA for GPU-accelerated training
- Export trained models to formats like ONNX or TorchScript

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support
- PyTorch
- Ultralytics
- OpenCV

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If you want CUDA support for NVIDIA GPUs:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Dataset Setup

The dataset config is in:

```text
datasets/data.yaml
```

Make sure the paths inside `data.yaml` point to your local dataset folders.

## Train

```bash
& "$env:APPDATA\Python\Python312\Scripts\yolo.exe" detect train data="datasets/data.yaml" model=yolo11m.pt epochs=50 imgsz=640 batch=8 device=0 workers=2 patience=20 cache=False project="models" name=elephant_model
```

## Validate

```bash
& "$env:APPDATA\Python\Python312\Scripts\yolo.exe" detect val model="models\elephant_model\weights\best.pt" data="datasets\data.yaml" device=0
```

## Predict

```bash
& "$env:APPDATA\Python\Python312\Scripts\yolo.exe" detect predict model="models\elephant_model\weights\best.pt" source="path\to\image.jpg" device=0
```

## Output

Training results are saved in:

```text
models\elephant_model\
```

## Project Structure

```text
.
├── backend/
├── config/
├── datasets/
├── detect.py
├── frontend/
├── logs/
├── models/
├── outputs/
├── runs/
├── scripts/
├── sri-lankan-wild-elephant-dataset/
├── training.py
└── requirements.txt
```


