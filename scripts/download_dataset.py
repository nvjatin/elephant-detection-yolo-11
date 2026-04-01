"""
Dataset Download Script for Elephant Detection
Downloads datasets from Kaggle, Roboflow, and Open Images to external SSD.

Usage:
    python scripts/download_dataset.py --source kaggle --dataset your-dataset
    python scripts/download_dataset.py --source roboflow --workspace ws --project proj --version 1
    python scripts/download_dataset.py --source openimages --classes Elephant --limit 1000
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATASET_PATH,
    KAGGLE_DATASET_PATH,
    ROBOFLOW_DATASET_PATH,
    OPENIMAGES_DATASET_PATH,
    ensure_directories,
    verify_ssd_mounted,
    LOG_PATH,
)


class DatasetDownloader:
    """Base class for dataset downloaders."""
    
    def __init__(self, target_path: Path):
        self.target_path = target_path
        self.log_file = LOG_PATH / "download_log.json"
        self.download_log = self._load_log()
    
    def _load_log(self) -> dict:
        """Load download log to track completed downloads."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {"downloads": []}
    
    def _save_log(self):
        """Save download log."""
        LOG_PATH.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.download_log, f, indent=2)
    
    def _is_downloaded(self, dataset_id: str) -> bool:
        """Check if dataset was already downloaded."""
        return any(d["id"] == dataset_id for d in self.download_log["downloads"])
    
    def _log_download(self, dataset_id: str, path: str, source: str):
        """Log successful download."""
        self.download_log["downloads"].append({
            "id": dataset_id,
            "path": path,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        self._save_log()


class KaggleDownloader(DatasetDownloader):
    """Download datasets from Kaggle."""
    
    def __init__(self):
        super().__init__(KAGGLE_DATASET_PATH)
        self._setup_kaggle()
    
    def _setup_kaggle(self):
        """Setup Kaggle API with SSD paths."""
        # Set Kaggle cache to SSD
        os.environ["KAGGLE_CONFIG_DIR"] = str(KAGGLE_DATASET_PATH / ".kaggle")
        
        try:
            import kaggle
            self.api = kaggle.KaggleApi()
            self.api.authenticate()
            print("✓ Kaggle API authenticated")
        except Exception as e:
            print(f"✗ Kaggle setup failed: {e}")
            print("  Run: pip install kaggle")
            print("  Place kaggle.json in: ~/.kaggle/")
            self.api = None
    
    def download(self, dataset: str, unzip: bool = True) -> Path:
        """
        Download a Kaggle dataset.
        
        Args:
            dataset: Kaggle dataset identifier (e.g., "username/dataset-name")
            unzip: Whether to unzip the downloaded files
        
        Returns:
            Path to downloaded dataset
        """
        if self.api is None:
            raise RuntimeError("Kaggle API not configured")
        
        dataset_id = f"kaggle:{dataset}"
        
        # Check if already downloaded
        if self._is_downloaded(dataset_id):
            existing_path = KAGGLE_DATASET_PATH / dataset.replace("/", "_")
            if existing_path.exists():
                print(f"✓ Dataset already exists: {existing_path}")
                return existing_path
        
        # Create target directory
        target_dir = KAGGLE_DATASET_PATH / dataset.replace("/", "_")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading: {dataset}")
        print(f"Target: {target_dir}")
        
        # Download dataset
        self.api.dataset_download_files(
            dataset,
            path=str(target_dir),
            unzip=unzip,
            quiet=False
        )
        
        # Log download
        self._log_download(dataset_id, str(target_dir), "kaggle")
        print(f"✓ Download complete: {target_dir}")
        
        return target_dir
    
    def download_competition(self, competition: str) -> Path:
        """Download competition data."""
        if self.api is None:
            raise RuntimeError("Kaggle API not configured")
        
        target_dir = KAGGLE_DATASET_PATH / f"competition_{competition}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading competition: {competition}")
        self.api.competition_download_files(
            competition,
            path=str(target_dir),
            quiet=False
        )
        
        return target_dir


class RoboflowDownloader(DatasetDownloader):
    """Download datasets from Roboflow."""
    
    def __init__(self, api_key: str = None):
        super().__init__(ROBOFLOW_DATASET_PATH)
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        self._setup_roboflow()
    
    def _setup_roboflow(self):
        """Setup Roboflow API."""
        try:
            from roboflow import Roboflow
            if self.api_key:
                self.rf = Roboflow(api_key=self.api_key)
                print("✓ Roboflow API authenticated")
            else:
                print("✗ Roboflow API key not found")
                print("  Set ROBOFLOW_API_KEY environment variable")
                self.rf = None
        except ImportError:
            print("✗ Roboflow not installed. Run: pip install roboflow")
            self.rf = None
    
    def download(self, workspace: str, project: str, version: int, 
                 format: str = "yolov8") -> Path:
        """
        Download a Roboflow dataset.
        
        Args:
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version number
            format: Export format (yolov8, coco, etc.)
        
        Returns:
            Path to downloaded dataset
        """
        if self.rf is None:
            raise RuntimeError("Roboflow API not configured")
        
        dataset_id = f"roboflow:{workspace}/{project}/v{version}"
        
        # Check if already downloaded
        if self._is_downloaded(dataset_id):
            existing_path = ROBOFLOW_DATASET_PATH / f"{workspace}_{project}_v{version}"
            if existing_path.exists():
                print(f"✓ Dataset already exists: {existing_path}")
                return existing_path
        
        # Create target directory
        target_dir = ROBOFLOW_DATASET_PATH / f"{workspace}_{project}_v{version}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading: {workspace}/{project} v{version}")
        print(f"Target: {target_dir}")
        
        # Download dataset
        project_obj = self.rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(
            format,
            location=str(target_dir)
        )
        
        # Log download
        self._log_download(dataset_id, str(target_dir), "roboflow")
        print(f"✓ Download complete: {target_dir}")
        
        return target_dir


class OpenImagesDownloader(DatasetDownloader):
    """Download datasets from Open Images Dataset."""
    
    def __init__(self):
        super().__init__(OPENIMAGES_DATASET_PATH)
    
    def download(self, classes: list[str], limit: int = 1000, 
                 split: str = "train") -> Path:
        """
        Download Open Images dataset for specific classes.
        
        Args:
            classes: List of class names to download (e.g., ["Elephant"])
            limit: Maximum number of images per class
            split: Dataset split (train, validation, test)
        
        Returns:
            Path to downloaded dataset
        """
        try:
            from openimages.download import download_dataset
        except ImportError:
            print("✗ openimages not installed. Run: pip install openimages")
            raise
        
        classes_str = "_".join(classes)
        dataset_id = f"openimages:{classes_str}:{split}:{limit}"
        
        # Check if already downloaded
        if self._is_downloaded(dataset_id):
            existing_path = OPENIMAGES_DATASET_PATH / f"{classes_str}_{split}"
            if existing_path.exists():
                print(f"✓ Dataset already exists: {existing_path}")
                return existing_path
        
        # Create target directory
        target_dir = OPENIMAGES_DATASET_PATH / f"{classes_str}_{split}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading: {classes} from Open Images")
        print(f"Split: {split}, Limit: {limit}")
        print(f"Target: {target_dir}")
        
        # Download dataset
        download_dataset(
            dest_dir=str(target_dir),
            class_labels=classes,
            annotation_format="darknet",  # YOLO format
            limit=limit,
        )
        
        # Log download
        self._log_download(dataset_id, str(target_dir), "openimages")
        print(f"✓ Download complete: {target_dir}")
        
        return target_dir


class FiftyOneDownloader(DatasetDownloader):
    """Download Open Images using FiftyOne (alternative method)."""
    
    def __init__(self):
        super().__init__(OPENIMAGES_DATASET_PATH)
        # Set FiftyOne dataset directory to SSD
        os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(OPENIMAGES_DATASET_PATH / "zoo")
    
    def download(self, classes: list[str], max_samples: int = 1000,
                 split: str = "train") -> Path:
        """
        Download Open Images using FiftyOne.
        
        Args:
            classes: List of class names
            max_samples: Maximum samples per class
            split: Dataset split
        
        Returns:
            Path to dataset
        """
        try:
            import fiftyone as fo
            import fiftyone.zoo as foz
        except ImportError:
            print("✗ FiftyOne not installed. Run: pip install fiftyone")
            raise
        
        classes_str = "_".join(classes)
        target_dir = OPENIMAGES_DATASET_PATH / f"fiftyone_{classes_str}_{split}"
        
        print(f"Downloading via FiftyOne: {classes}")
        print(f"Target: {target_dir}")
        
        # Download dataset
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            classes=classes,
            max_samples=max_samples,
            dataset_dir=str(target_dir),
            label_types=["detections"],
        )
        
        # Export to YOLO format
        yolo_dir = target_dir / "yolo_format"
        dataset.export(
            export_dir=str(yolo_dir),
            dataset_type=fo.types.YOLOv5Dataset,
        )
        
        print(f"✓ Dataset exported to YOLO format: {yolo_dir}")
        return yolo_dir


def convert_to_yolo_format(source_dir: Path, target_dir: Path):
    """Convert dataset to YOLO format if needed."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create YOLO structure
    for split in ["train", "val", "test"]:
        (target_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (target_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created YOLO format structure at: {target_dir}")
    return target_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets to external SSD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Kaggle dataset
  python download_dataset.py --source kaggle --dataset jayaprakashpondy/elephant-face

  # Roboflow dataset  
  python download_dataset.py --source roboflow --workspace my-workspace --project elephant-detection --version 1

  # Open Images (elephant class)
  python download_dataset.py --source openimages --classes Elephant --limit 1000

  # Open Images via FiftyOne
  python download_dataset.py --source fiftyone --classes Elephant --limit 500
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        choices=["kaggle", "roboflow", "openimages", "fiftyone"],
        required=True,
        help="Dataset source"
    )
    
    # Kaggle arguments
    parser.add_argument("--dataset", help="Kaggle dataset identifier")
    parser.add_argument("--competition", help="Kaggle competition name")
    
    # Roboflow arguments
    parser.add_argument("--workspace", help="Roboflow workspace")
    parser.add_argument("--project", help="Roboflow project")
    parser.add_argument("--version", type=int, help="Roboflow version")
    parser.add_argument("--api-key", help="Roboflow API key")
    
    # Open Images arguments
    parser.add_argument("--classes", nargs="+", default=["Elephant"],
                        help="Classes to download")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max images per class")
    parser.add_argument("--split", default="train",
                        help="Dataset split (train/validation/test)")
    
    args = parser.parse_args()
    
    # Verify SSD is mounted
    verify_ssd_mounted()
    ensure_directories()
    
    print(f"\n{'='*60}")
    print(f"Dataset Download - Target: {DATASET_PATH}")
    print(f"{'='*60}\n")
    
    try:
        if args.source == "kaggle":
            downloader = KaggleDownloader()
            if args.competition:
                path = downloader.download_competition(args.competition)
            elif args.dataset:
                path = downloader.download(args.dataset)
            else:
                parser.error("--dataset or --competition required for Kaggle")
        
        elif args.source == "roboflow":
            if not all([args.workspace, args.project, args.version]):
                parser.error("--workspace, --project, and --version required for Roboflow")
            downloader = RoboflowDownloader(api_key=args.api_key)
            path = downloader.download(args.workspace, args.project, args.version)
        
        elif args.source == "openimages":
            downloader = OpenImagesDownloader()
            path = downloader.download(args.classes, args.limit, args.split)
        
        elif args.source == "fiftyone":
            downloader = FiftyOneDownloader()
            path = downloader.download(args.classes, args.limit, args.split)
        
        print(f"\n✓ Dataset ready at: {path}")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
