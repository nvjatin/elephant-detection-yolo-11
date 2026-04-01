"""
Training Script for Elephant Detection using YOLOv8
Optimized for Apple Silicon (M1/M2/M3) with MPS backend.
All data reads from and writes to external SSD.

Usage:
    python scripts/train.py --dataset elephant_data --epochs 100
    python scripts/train.py --dataset elephant_data --resume
    python scripts/train.py --dataset elephant_data --batch-size 8
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATASET_PATH,
    MODEL_PATH,
    PRETRAINED_MODEL_PATH,
    TRAINED_MODEL_PATH,
    TRAINING_OUTPUT_PATH,
    LOG_PATH,
    TRAINING_CONFIG,
    BATCH_SIZE_MAP,
    get_device,
    ensure_directories,
    verify_ssd_mounted,
)


def get_system_info() -> dict:
    """Get system information for optimization."""
    import platform
    import subprocess
    
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }
    
    # Try to get Mac chip info
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        )
        info["cpu"] = result.stdout.strip()
        
        # Detect M-series chip
        for chip in BATCH_SIZE_MAP.keys():
            if chip.replace("_", " ") in info["cpu"]:
                info["chip"] = chip
                info["recommended_batch"] = BATCH_SIZE_MAP[chip]
                break
    except Exception:
        pass
    
    return info


def create_dataset_yaml(dataset_path: Path, output_path: Path) -> Path:
    """Create or verify dataset YAML file for YOLOv8."""
    yaml_path = dataset_path / "data.yaml"
    
    if yaml_path.exists():
        print(f"✓ Using existing data.yaml: {yaml_path}")
        return yaml_path
    
    # Create a basic data.yaml
    yaml_content = f"""# Elephant Detection Dataset
# Auto-generated for training

path: {dataset_path}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['elephant']
"""
    
    yaml_path.write_text(yaml_content)
    print(f"✓ Created data.yaml: {yaml_path}")
    return yaml_path


def train(
    dataset_name: str,
    epochs: int = None,
    batch_size: int = None,
    imgsz: int = None,
    model: str = None,
    resume: bool = False,
    name: str = None,
):
    """
    Train YOLOv8 model on elephant dataset.
    
    Args:
        dataset_name: Name of dataset folder in DATASET_PATH
        epochs: Number of training epochs
        batch_size: Batch size (auto-tuned for chip if not specified)
        imgsz: Image size
        model: Base model to use
        resume: Resume from last checkpoint
        name: Training run name
    """
    from ultralytics import YOLO
    
    # Verify SSD and create directories
    verify_ssd_mounted()
    ensure_directories()
    
    # Get system info
    sys_info = get_system_info()
    device = get_device()
    
    print("=" * 60)
    print("ELEPHANT DETECTION - TRAINING")
    print("=" * 60)
    print(f"Device: {device.upper()}")
    print(f"PyTorch: {sys_info['torch']}")
    print(f"MPS Available: {sys_info['mps_available']}")
    if "chip" in sys_info:
        print(f"Chip: {sys_info['chip']}")
    print("=" * 60)
    
    # Setup paths
    dataset_path = DATASET_PATH / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Auto-tune batch size for Apple Silicon
    if batch_size is None:
        batch_size = sys_info.get("recommended_batch", TRAINING_CONFIG["batch_size"])
        print(f"Auto-selected batch size: {batch_size}")
    
    # Create/verify dataset YAML
    data_yaml = create_dataset_yaml(dataset_path, TRAINING_OUTPUT_PATH)
    
    # Setup model
    if model is None:
        model_path = PRETRAINED_MODEL_PATH / "yolov8n.pt"
        if not model_path.exists():
            # Download to SSD
            print(f"Downloading YOLOv8n to: {model_path}")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            yolo = YOLO("yolov8n.pt")
            # Move downloaded model to SSD
            import shutil
            default_path = Path.home() / ".cache" / "ultralytics"
            # The model will be auto-downloaded, we use it directly
        model = str(model_path) if model_path.exists() else "yolov8n.pt"
    
    # Training run name
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"elephant_{timestamp}"
    
    # Resume handling
    if resume:
        last_checkpoint = TRAINING_OUTPUT_PATH / name / "weights" / "last.pt"
        if last_checkpoint.exists():
            model = str(last_checkpoint)
            print(f"Resuming from: {last_checkpoint}")
        else:
            print("No checkpoint found, starting fresh")
            resume = False
    
    # Initialize model
    print(f"\nLoading model: {model}")
    yolo = YOLO(model)
    
    # Configure training parameters
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs or TRAINING_CONFIG["epochs"],
        "batch": batch_size,
        "imgsz": imgsz or TRAINING_CONFIG["imgsz"],
        "patience": TRAINING_CONFIG["patience"],
        "workers": TRAINING_CONFIG["workers"],
        "device": device,
        "project": str(TRAINING_OUTPUT_PATH),
        "name": name,
        "exist_ok": True,
        "cache": False,  # Stream from SSD, don't cache in RAM
        "resume": resume,
        "verbose": True,
        "plots": True,
        "save": True,
        "save_period": 10,  # Save checkpoint every 10 epochs
    }
    
    # MPS-specific optimizations
    if device == "mps":
        # Reduce memory pressure on Apple Silicon
        train_args.update({
            "amp": False,  # AMP can be unstable on MPS
            "rect": False,  # Rectangular training can cause issues
        })
    
    print("\nTraining Configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print()
    
    # Start training
    print("Starting training...")
    print("-" * 60)
    
    try:
        results = yolo.train(**train_args)
        
        # Copy best model to standard location
        best_model = TRAINING_OUTPUT_PATH / name / "weights" / "best.pt"
        if best_model.exists():
            import shutil
            dest = TRAINED_MODEL_PATH / "best.pt"
            shutil.copy2(best_model, dest)
            print(f"\n✓ Best model saved to: {dest}")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {TRAINING_OUTPUT_PATH / name}")
        print(f"Best model: {TRAINED_MODEL_PATH / 'best.pt'}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        
        # Save error log
        error_log = LOG_PATH / f"train_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        error_log.write_text(str(e))
        print(f"Error log saved to: {error_log}")
        
        raise


def validate(model_path: str = None, dataset_name: str = None):
    """Validate a trained model."""
    from ultralytics import YOLO
    
    verify_ssd_mounted()
    
    # Default to best model
    if model_path is None:
        model_path = TRAINED_MODEL_PATH / "best.pt"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Validating model: {model_path}")
    
    yolo = YOLO(str(model_path))
    
    if dataset_name:
        data_yaml = DATASET_PATH / dataset_name / "data.yaml"
        results = yolo.val(data=str(data_yaml), device=get_device())
    else:
        results = yolo.val(device=get_device())
    
    return results


def export_model(model_path: str = None, format: str = "onnx"):
    """Export model to different formats."""
    from ultralytics import YOLO
    
    verify_ssd_mounted()
    
    if model_path is None:
        model_path = TRAINED_MODEL_PATH / "best.pt"
    
    print(f"Exporting model: {model_path}")
    print(f"Format: {format}")
    
    yolo = YOLO(str(model_path))
    
    export_path = yolo.export(format=format)
    
    # Move to models folder
    import shutil
    dest = TRAINED_MODEL_PATH / Path(export_path).name
    shutil.move(export_path, dest)
    
    print(f"✓ Exported to: {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Elephant Detection (Apple Silicon optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train.py --dataset elephant_data

  # Custom parameters
  python train.py --dataset elephant_data --epochs 50 --batch-size 8

  # Resume training
  python train.py --dataset elephant_data --resume --name my_run

  # Validate model
  python train.py --validate --model models/trained/best.pt

  # Export model
  python train.py --export --format onnx
        """
    )
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", default=True,
                      help="Train model (default)")
    mode.add_argument("--validate", action="store_true",
                      help="Validate model")
    mode.add_argument("--export", action="store_true",
                      help="Export model")
    
    # Training arguments
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset folder name in datasets/")
    parser.add_argument("--epochs", "-e", type=int,
                        help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int,
                        help="Batch size (auto-tuned if not specified)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--model", "-m", type=str,
                        help="Base model path")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--name", "-n", type=str,
                        help="Training run name")
    
    # Export arguments
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "torchscript", "coreml", "tflite"],
                        help="Export format")
    
    args = parser.parse_args()
    
    try:
        if args.validate:
            validate(args.model, args.dataset)
        elif args.export:
            export_model(args.model, args.format)
        else:
            if not args.dataset:
                parser.error("--dataset required for training")
            train(
                dataset_name=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                model=args.model,
                resume=args.resume,
                name=args.name,
            )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
