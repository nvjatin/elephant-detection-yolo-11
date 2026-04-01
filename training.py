"""
YOLOv11 Training Script for Elephant Detection
Optimized for Apple M3 with MPS backend.

IMPORTANT: This script does NOT auto-execute training.
Use the train_model() function or CLI commands manually.

Usage:
    # Option 1: Import and call function
    from training import train_model
    train_model()
    
    # Option 2: Run as script with --run flag
    python training.py --run
    
    # Option 3: Use yolo CLI (recommended)
    See commands at bottom of this file
"""

import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION - ALL PATHS ON EXTERNAL SSD
# =============================================================================

# Base paths (External SSD)
BASE_PATH = Path("/Volumes/Extended Storage/Elephant-Detection")
DATASET_PATH = BASE_PATH / "datasets"
MODEL_PATH = BASE_PATH / "models"
OUTPUT_PATH = MODEL_PATH / "elephant_model"

# Dataset configuration
DATA_YAML = DATASET_PATH / "data.yaml"

# Model configuration
BASE_MODEL = "yolo11m.pt"  # YOLOv11 Medium - good balance of speed and accuracy

# Training hyperparameters (optimized for Apple M3)
TRAINING_CONFIG = {
    "epochs": 50,
    "batch": 8,           # Safe for M3 (8GB+ unified memory)
    "imgsz": 640,
    "workers": 2,         # Reduced to avoid memory pressure
    "device": "mps",      # Apple Metal Performance Shaders
    "patience": 20,       # Early stopping patience
    "cache": False,       # Don't cache images in RAM
    "amp": False,         # AMP can be unstable on MPS
    "verbose": True,
    "plots": True,
}


# =============================================================================
# TRAINING FUNCTION (Manual execution only)
# =============================================================================

def train_model(
    data_yaml: str = None,
    model: str = None,
    epochs: int = None,
    batch_size: int = None,
    img_size: int = None,
    device: str = None,
    project: str = None,
    name: str = None,
    resume: bool = False,
):
    """
    Train YOLOv8 model for elephant detection.
    
    This function does NOT auto-execute. Call it explicitly.
    
    Args:
        data_yaml: Path to data.yaml file
        model: Base model (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Number of training epochs
        batch_size: Batch size (8 recommended for M3)
        img_size: Image size (640 default)
        device: Device to use ('mps' for Apple Silicon)
        project: Output project directory
        name: Run name
        resume: Resume from last checkpoint
    
    Returns:
        Training results object
    """
    from ultralytics import YOLO
    
    # Verify SSD is mounted
    if not BASE_PATH.exists():
        raise RuntimeError(
            f"External SSD not mounted at: {BASE_PATH}\n"
            "Please connect your external storage."
        )
    
    # Use defaults if not specified
    data_yaml = data_yaml or str(DATA_YAML)
    model = model or BASE_MODEL
    epochs = epochs or TRAINING_CONFIG["epochs"]
    batch_size = batch_size or TRAINING_CONFIG["batch"]
    img_size = img_size or TRAINING_CONFIG["imgsz"]
    device = device or TRAINING_CONFIG["device"]
    project = project or str(MODEL_PATH)
    name = name or "elephant_model"
    
    # Verify data.yaml exists
    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}\n"
            "Please create data.yaml in your datasets folder."
        )
    
    # Print configuration
    print("=" * 60)
    print("ELEPHANT DETECTION - YOLOV8 TRAINING")
    print("=" * 60)
    print(f"Dataset:    {data_yaml}")
    print(f"Model:      {model}")
    print(f"Epochs:     {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Device:     {device}")
    print(f"Output:     {project}/{name}")
    print("=" * 60)
    
    # Load model
    yolo = YOLO(model)
    
    # Start training
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        resume=resume,
        workers=TRAINING_CONFIG["workers"],
        patience=TRAINING_CONFIG["patience"],
        cache=TRAINING_CONFIG["cache"],
        amp=TRAINING_CONFIG["amp"],
        verbose=TRAINING_CONFIG["verbose"],
        plots=TRAINING_CONFIG["plots"],
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: {project}/{name}/weights/best.pt")
    print(f"Last model: {project}/{name}/weights/last.pt")
    print("=" * 60)
    
    return results


def validate_model(
    model_path: str = None,
    data_yaml: str = None,
    device: str = "mps",
):
    """
    Validate trained model.
    
    Args:
        model_path: Path to trained model (best.pt)
        data_yaml: Path to data.yaml
        device: Device to use
    
    Returns:
        Validation metrics
    """
    from ultralytics import YOLO
    
    model_path = model_path or str(OUTPUT_PATH / "weights" / "best.pt")
    data_yaml = data_yaml or str(DATA_YAML)
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Validating: {model_path}")
    
    yolo = YOLO(model_path)
    results = yolo.val(data=data_yaml, device=device)
    
    return results


def export_model(
    model_path: str = None,
    format: str = "onnx",
):
    """
    Export model to different format.
    
    Args:
        model_path: Path to trained model
        format: Export format (onnx, torchscript, coreml, etc.)
    
    Returns:
        Path to exported model
    """
    from ultralytics import YOLO
    
    model_path = model_path or str(OUTPUT_PATH / "weights" / "best.pt")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Exporting: {model_path} -> {format}")
    
    yolo = YOLO(model_path)
    export_path = yolo.export(format=format)
    
    print(f"Exported to: {export_path}")
    return export_path


def print_commands():
    """Print CLI commands for manual execution."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    YOLOV8 TRAINING COMMANDS (COPY & PASTE)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. TRAINING COMMAND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

yolo detect train \\
    data="/Volumes/Extended Storage/Elephant-Detection/datasets/data.yaml" \\
    model=yolov8n.pt \\
    epochs=50 \\
    imgsz=640 \\
    batch=8 \\
    device=mps \\
    workers=2 \\
    patience=20 \\
    cache=False \\
    project="/Volumes/Extended Storage/Elephant-Detection/models" \\
    name=elephant_model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. VALIDATION COMMAND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

yolo detect val \\
    model="/Volumes/Extended Storage/Elephant-Detection/models/elephant_model/weights/best.pt" \\
    data="/Volumes/Extended Storage/Elephant-Detection/datasets/data.yaml" \\
    device=mps

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. EXPORT COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Export to ONNX
yolo export \\
    model="/Volumes/Extended Storage/Elephant-Detection/models/elephant_model/weights/best.pt" \\
    format=onnx

# Export to CoreML (Apple optimized)
yolo export \\
    model="/Volumes/Extended Storage/Elephant-Detection/models/elephant_model/weights/best.pt" \\
    format=coreml

# Export to TorchScript
yolo export \\
    model="/Volumes/Extended Storage/Elephant-Detection/models/elephant_model/weights/best.pt" \\
    format=torchscript

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. RESUME TRAINING (if interrupted)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

yolo detect train \\
    model="/Volumes/Extended Storage/Elephant-Detection/models/elephant_model/weights/last.pt" \\
    resume=True

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. INFERENCE/PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Predict on image
yolo detect predict \\
    model="/Volumes/Extended Storage/Elephant-Detection/models/elephant_model/weights/best.pt" \\
    source="/path/to/image.jpg" \\
    device=mps

# Predict on video
yolo detect predict \\
    model="/Volumes/Extended Storage/Elephant-Detection/models/elephant_model/weights/best.pt" \\
    source="/path/to/video.mp4" \\
    device=mps

# Predict from webcam
yolo detect predict \\
    model="/Volumes/Extended Storage/Elephant-Detection/models/elephant_model/weights/best.pt" \\
    source=0 \\
    device=mps

""")


# =============================================================================
# MAIN (Only runs if explicitly requested)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YOLOv8 Training for Elephant Detection",
        epilog="NOTE: Training does NOT auto-start. Use --run or --commands."
    )
    
    parser.add_argument("--run", action="store_true",
                        help="Actually run training (requires explicit flag)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation on trained model")
    parser.add_argument("--export", type=str, metavar="FORMAT",
                        help="Export model to format (onnx, coreml, torchscript)")
    parser.add_argument("--commands", action="store_true",
                        help="Print CLI commands for manual execution")
    
    # Training options
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--model", type=str, help="Base model")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    
    args = parser.parse_args()
    
    if args.commands or (not args.run and not args.validate and not args.export):
        print_commands()
    
    elif args.run:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch,
            model=args.model,
            resume=args.resume,
        )
    
    elif args.validate:
        validate_model()
    
    elif args.export:
        export_model(format=args.export)
