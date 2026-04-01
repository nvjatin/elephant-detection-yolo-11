"""
Configuration for Elephant Detection System
All paths point to external SSD - NEVER uses internal Mac storage.
"""

import os
from pathlib import Path

# =============================================================================
# BASE PATHS (External SSD)
# =============================================================================

BASE_PATH = Path("/Volumes/Extended Storage/Elephant-Detection")
DATASET_PATH = BASE_PATH / "datasets"
MODEL_PATH = BASE_PATH / "models"
OUTPUT_PATH = BASE_PATH / "outputs"
LOG_PATH = BASE_PATH / "logs"
SCRIPTS_PATH = BASE_PATH / "scripts"
CONFIG_PATH = BASE_PATH / "config"

# =============================================================================
# DATASET PATHS
# =============================================================================

# Kaggle datasets
KAGGLE_DATASET_PATH = DATASET_PATH / "kaggle"

# Roboflow datasets
ROBOFLOW_DATASET_PATH = DATASET_PATH / "roboflow"

# Open Images datasets
OPENIMAGES_DATASET_PATH = DATASET_PATH / "openimages"

# Custom datasets
CUSTOM_DATASET_PATH = DATASET_PATH / "custom"

# =============================================================================
# MODEL PATHS
# =============================================================================

# Pre-trained models
PRETRAINED_MODEL_PATH = MODEL_PATH / "pretrained"

# Custom trained models
TRAINED_MODEL_PATH = MODEL_PATH / "trained"

# Best model for inference
BEST_MODEL_PATH = MODEL_PATH / "trained" / "best.pt"

# Default YOLO model
DEFAULT_YOLO_MODEL = "yolov8n.pt"

# =============================================================================
# OUTPUT PATHS
# =============================================================================

# Processed videos
VIDEO_OUTPUT_PATH = OUTPUT_PATH / "videos"

# Detection logs
DETECTION_LOG_PATH = OUTPUT_PATH / "detections"

# Training runs
TRAINING_OUTPUT_PATH = OUTPUT_PATH / "training"

# =============================================================================
# DEVICE CONFIGURATION (Apple Silicon)
# =============================================================================

import torch

def get_device() -> str:
    """Get optimal device for Apple Silicon."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = get_device()

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,  # Optimized for M3
    "imgsz": 640,
    "patience": 20,
    "workers": 4,  # Apple Silicon optimized
    "cache": False,  # Don't cache to RAM - stream from SSD
    "device": DEVICE,
    "project": str(TRAINING_OUTPUT_PATH),
    "exist_ok": True,
}

# Memory-optimized batch sizes for different M-series chips
BATCH_SIZE_MAP = {
    "M1": 8,
    "M1_Pro": 12,
    "M1_Max": 16,
    "M2": 12,
    "M2_Pro": 16,
    "M2_Max": 24,
    "M3": 16,
    "M3_Pro": 24,
    "M3_Max": 32,
}

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

INFERENCE_CONFIG = {
    "conf_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_det": 100,
    "device": DEVICE,
}

# COCO class ID for elephant
ELEPHANT_CLASS_ID = 22

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================

DEFAULT_CAMERAS = [
    {"source": 0, "location": "Village A"},
    {"source": "http://192.168.1.101:8080/video", "location": "Village B"},
    {"source": "http://192.168.1.102:8080/video", "location": "Forest Edge"},
    {"source": "rtsp://192.168.1.103:554/stream", "location": "Highway"},
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create all required directories on SSD."""
    directories = [
        DATASET_PATH,
        KAGGLE_DATASET_PATH,
        ROBOFLOW_DATASET_PATH,
        OPENIMAGES_DATASET_PATH,
        CUSTOM_DATASET_PATH,
        MODEL_PATH,
        PRETRAINED_MODEL_PATH,
        TRAINED_MODEL_PATH,
        OUTPUT_PATH,
        VIDEO_OUTPUT_PATH,
        DETECTION_LOG_PATH,
        TRAINING_OUTPUT_PATH,
        LOG_PATH,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"✓ All directories created on: {BASE_PATH}")


def get_model_path(model_name: str = None) -> Path:
    """Get path to model file on SSD."""
    if model_name is None:
        # Return best trained model if exists, else default
        if BEST_MODEL_PATH.exists():
            return BEST_MODEL_PATH
        return PRETRAINED_MODEL_PATH / DEFAULT_YOLO_MODEL
    return TRAINED_MODEL_PATH / model_name


def verify_ssd_mounted() -> bool:
    """Verify external SSD is mounted."""
    if not BASE_PATH.exists():
        raise RuntimeError(
            f"External SSD not mounted at: {BASE_PATH}\n"
            "Please connect your external storage and try again."
        )
    return True


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("ELEPHANT DETECTION - CONFIGURATION")
    print("=" * 60)
    print(f"Base Path:     {BASE_PATH}")
    print(f"Datasets:      {DATASET_PATH}")
    print(f"Models:        {MODEL_PATH}")
    print(f"Outputs:       {OUTPUT_PATH}")
    print(f"Device:        {DEVICE}")
    print(f"SSD Mounted:   {BASE_PATH.exists()}")
    print("=" * 60)


# Verify SSD on import
if __name__ != "__main__":
    try:
        verify_ssd_mounted()
    except RuntimeError as e:
        print(f"WARNING: {e}")
