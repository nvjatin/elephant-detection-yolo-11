"""
Elephant Dataset Downloader & Merger
Downloads from multiple sources and combines into single YOLO format dataset.

Sources:
- Kaggle elephant datasets
- Roboflow elephant datasets
- Open Images Dataset (elephant class)

Output: Unified YOLO format dataset on external SSD

Usage:
    python scripts/download_and_merge.py --all
    python scripts/download_and_merge.py --kaggle
    python scripts/download_and_merge.py --openimages --limit 500
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime
import json

# Base paths (External SSD)
BASE_PATH = Path("/Volumes/Extended Storage/Elephant-Detection")
DATASET_PATH = BASE_PATH / "datasets"
RAW_PATH = DATASET_PATH / "raw"  # Downloaded datasets before merge
MERGED_PATH = DATASET_PATH  # Final merged dataset

# Create directories
for p in [RAW_PATH, MERGED_PATH / "train" / "images", MERGED_PATH / "train" / "labels",
          MERGED_PATH / "val" / "images", MERGED_PATH / "val" / "labels",
          MERGED_PATH / "test" / "images", MERGED_PATH / "test" / "labels"]:
    p.mkdir(parents=True, exist_ok=True)


class DatasetMerger:
    """Merge multiple datasets into unified YOLO format."""
    
    def __init__(self):
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.stats = {
            "total_images": 0,
            "total_labels": 0,
            "train": 0,
            "val": 0,
            "test": 0,
            "sources": []
        }
    
    def download_kaggle_datasets(self) -> list[Path]:
        """Download elephant datasets from Kaggle."""
        try:
            import kaggle
            kaggle.api.authenticate()
        except Exception as e:
            print(f"⚠ Kaggle not configured: {e}")
            print("  To enable: pip install kaggle && place kaggle.json in ~/.kaggle/")
            return []
        
        datasets = [
            "jayaprakashpondy/elephant-face",
            "vivmankar/african-wildlife",  # Contains elephants
        ]
        
        downloaded = []
        for dataset in datasets:
            try:
                target = RAW_PATH / "kaggle" / dataset.replace("/", "_")
                if target.exists() and any(target.iterdir()):
                    print(f"✓ Already downloaded: {dataset}")
                    downloaded.append(target)
                    continue
                
                target.mkdir(parents=True, exist_ok=True)
                print(f"Downloading: {dataset}")
                kaggle.api.dataset_download_files(dataset, path=str(target), unzip=True)
                downloaded.append(target)
                print(f"✓ Downloaded: {target}")
            except Exception as e:
                print(f"✗ Failed {dataset}: {e}")
        
        return downloaded
    
    def download_openimages(self, limit: int = 500) -> Path:
        """Download elephant images from Open Images Dataset."""
        target = RAW_PATH / "openimages" / "elephant"
        
        if target.exists() and len(list((target / "images").glob("*"))) > 10:
            print(f"✓ Already downloaded: Open Images")
            return target
        
        target.mkdir(parents=True, exist_ok=True)
        
        try:
            # Method 1: Using fiftyone
            import fiftyone as fo
            import fiftyone.zoo as foz
            
            print(f"Downloading Open Images (elephant class, limit={limit})...")
            
            # Set download directory to SSD
            fo.config.dataset_zoo_dir = str(RAW_PATH / "openimages" / "zoo")
            
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                classes=["Elephant"],
                max_samples=limit,
                label_types=["detections"],
            )
            
            # Export to YOLO format
            dataset.export(
                export_dir=str(target),
                dataset_type=fo.types.YOLOv5Dataset,
            )
            
            print(f"✓ Downloaded Open Images: {target}")
            return target
            
        except ImportError:
            print("⚠ FiftyOne not installed. Trying alternative method...")
        
        try:
            # Method 2: Using openimages-downloader
            from openimages.download import download_dataset
            
            print(f"Downloading Open Images via openimages package...")
            download_dataset(
                dest_dir=str(target),
                class_labels=["Elephant"],
                annotation_format="darknet",
                limit=limit,
            )
            return target
            
        except ImportError:
            print("⚠ openimages not installed.")
            print("  Install with: pip install fiftyone  OR  pip install openimages")
            return None
    
    def download_roboflow(self, workspace: str = None, project: str = None, 
                          version: int = 1, api_key: str = None) -> Path:
        """Download from Roboflow."""
        api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        
        if not api_key:
            print("⚠ Roboflow: Set ROBOFLOW_API_KEY environment variable")
            # Try to use public elephant datasets
            workspace = workspace or "animal-detection-kwxs3"
            project = project or "elephant-detection-6ojs7"
        
        try:
            from roboflow import Roboflow
            
            target = RAW_PATH / "roboflow" / f"{project}_v{version}"
            
            if target.exists() and any(target.iterdir()):
                print(f"✓ Already downloaded: Roboflow {project}")
                return target
            
            target.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading from Roboflow: {workspace}/{project}")
            rf = Roboflow(api_key=api_key) if api_key else Roboflow()
            project_obj = rf.workspace(workspace).project(project)
            dataset = project_obj.version(version).download("yolov8", location=str(target))
            
            print(f"✓ Downloaded: {target}")
            return target
            
        except Exception as e:
            print(f"⚠ Roboflow download failed: {e}")
            return None
    
    def find_yolo_structure(self, path: Path) -> dict:
        """Find images and labels in a dataset directory."""
        result = {"images": [], "labels": [], "format": None}
        
        # Check for standard YOLO structure
        for split in ["train", "val", "valid", "test", ""]:
            img_dirs = [
                path / split / "images",
                path / split / "imgs",
                path / "images" / split,
                path / split,
            ]
            label_dirs = [
                path / split / "labels",
                path / split / "annotations",
                path / "labels" / split,
            ]
            
            for img_dir in img_dirs:
                if img_dir.exists():
                    images = [f for f in img_dir.iterdir() 
                              if f.suffix.lower() in self.image_extensions]
                    if images:
                        result["images"].extend(images)
                        result["format"] = "yolo"
                        
                        # Find corresponding labels
                        for label_dir in label_dirs:
                            if label_dir.exists():
                                for img in images:
                                    label_file = label_dir / f"{img.stem}.txt"
                                    if label_file.exists():
                                        result["labels"].append((img, label_file))
        
        # Also search recursively for images
        if not result["images"]:
            for ext in self.image_extensions:
                result["images"].extend(path.rglob(f"*{ext}"))
        
        return result
    
    def convert_to_yolo(self, annotation_file: Path, img_width: int, img_height: int) -> list[str]:
        """Convert various annotation formats to YOLO format."""
        lines = []
        
        suffix = annotation_file.suffix.lower()
        
        if suffix == ".txt":
            # Already YOLO format or similar
            with open(annotation_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Ensure class 0 for elephant
                        lines.append(f"0 {' '.join(parts[1:5])}")
        
        elif suffix == ".xml":
            # Pascal VOC format
            import xml.etree.ElementTree as ET
            tree = ET.parse(annotation_file)
            root = tree.getroot()
            
            for obj in root.findall(".//object"):
                name = obj.find("name").text.lower()
                if "elephant" in name:
                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)
                    
                    # Convert to YOLO format
                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        elif suffix == ".json":
            # COCO format
            with open(annotation_file) as f:
                data = json.load(f)
            
            if "annotations" in data:
                # Full COCO format
                for ann in data["annotations"]:
                    bbox = ann["bbox"]  # [x, y, width, height]
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return lines
    
    def merge_datasets(self, sources: list[Path], train_ratio: float = 0.8, 
                       val_ratio: float = 0.15, test_ratio: float = 0.05):
        """Merge multiple dataset sources into unified YOLO format."""
        
        print("\n" + "=" * 60)
        print("MERGING DATASETS")
        print("=" * 60)
        
        all_samples = []  # List of (image_path, label_path or None)
        
        for source in sources:
            if source is None or not source.exists():
                continue
                
            print(f"\nProcessing: {source.name}")
            data = self.find_yolo_structure(source)
            
            if data["labels"]:
                all_samples.extend(data["labels"])
                print(f"  Found {len(data['labels'])} labeled images")
            elif data["images"]:
                # Images without labels - skip or handle
                print(f"  Found {len(data['images'])} images (no labels)")
            
            self.stats["sources"].append(str(source))
        
        if not all_samples:
            print("\n✗ No labeled images found!")
            print("Please ensure your datasets have YOLO format labels.")
            return False
        
        # Shuffle and split
        random.shuffle(all_samples)
        total = len(all_samples)
        
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        splits = {
            "train": all_samples[:train_end],
            "val": all_samples[train_end:val_end],
            "test": all_samples[val_end:],
        }
        
        # Copy files to merged directory
        for split_name, samples in splits.items():
            img_dir = MERGED_PATH / split_name / "images"
            label_dir = MERGED_PATH / split_name / "labels"
            
            for i, (img_path, label_path) in enumerate(samples):
                # Generate unique filename
                new_name = f"elephant_{split_name}_{i:05d}"
                
                # Copy image
                new_img = img_dir / f"{new_name}{img_path.suffix}"
                shutil.copy2(img_path, new_img)
                
                # Copy label
                new_label = label_dir / f"{new_name}.txt"
                shutil.copy2(label_path, new_label)
                
                self.stats["total_images"] += 1
                self.stats["total_labels"] += 1
            
            self.stats[split_name] = len(samples)
            print(f"  {split_name}: {len(samples)} images")
        
        return True
    
    def create_data_yaml(self):
        """Create data.yaml for merged dataset."""
        yaml_content = f"""# Elephant Detection Dataset (Merged)
# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Sources: {len(self.stats['sources'])} datasets

path: {MERGED_PATH}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: elephant

# Dataset Statistics:
# Total images: {self.stats['total_images']}
# Train: {self.stats['train']}
# Val: {self.stats['val']}
# Test: {self.stats['test']}
"""
        
        yaml_path = MERGED_PATH / "data.yaml"
        yaml_path.write_text(yaml_content)
        print(f"\n✓ Created: {yaml_path}")
    
    def print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("DATASET READY")
        print("=" * 60)
        print(f"Location: {MERGED_PATH}")
        print(f"Total Images: {self.stats['total_images']}")
        print(f"  Train: {self.stats['train']}")
        print(f"  Val: {self.stats['val']}")
        print(f"  Test: {self.stats['test']}")
        print("=" * 60)
        print("\nNext step - Run training:")
        print(f'''
yolo detect train \\
    data="{MERGED_PATH}/data.yaml" \\
    model=yolov8n.pt \\
    epochs=50 \\
    imgsz=640 \\
    batch=8 \\
    device=mps \\
    project="{BASE_PATH}/models" \\
    name=elephant_model
''')


def main():
    parser = argparse.ArgumentParser(
        description="Download and merge elephant datasets into YOLO format"
    )
    
    parser.add_argument("--all", action="store_true",
                        help="Download from all available sources")
    parser.add_argument("--kaggle", action="store_true",
                        help="Download from Kaggle")
    parser.add_argument("--openimages", action="store_true",
                        help="Download from Open Images")
    parser.add_argument("--roboflow", action="store_true",
                        help="Download from Roboflow")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max images per source (default: 500)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing downloads")
    
    args = parser.parse_args()
    
    # Verify SSD
    if not BASE_PATH.exists():
        print(f"✗ External SSD not mounted: {BASE_PATH}")
        sys.exit(1)
    
    print("=" * 60)
    print("ELEPHANT DATASET DOWNLOADER & MERGER")
    print("=" * 60)
    print(f"Target: {MERGED_PATH}")
    print()
    
    merger = DatasetMerger()
    sources = []
    
    if not args.merge_only:
        # Download datasets
        if args.all or args.kaggle:
            sources.extend(merger.download_kaggle_datasets())
        
        if args.all or args.openimages:
            oi_path = merger.download_openimages(limit=args.limit)
            if oi_path:
                sources.append(oi_path)
        
        if args.all or args.roboflow:
            rf_path = merger.download_roboflow()
            if rf_path:
                sources.append(rf_path)
    
    # If no specific source selected, use all from raw folder
    if not sources:
        print("\nSearching for existing datasets in raw folder...")
        if RAW_PATH.exists():
            for source_dir in RAW_PATH.iterdir():
                if source_dir.is_dir():
                    for dataset_dir in source_dir.iterdir():
                        if dataset_dir.is_dir():
                            sources.append(dataset_dir)
                            print(f"  Found: {dataset_dir}")
    
    if not sources:
        print("\n✗ No datasets found!")
        print("\nOptions:")
        print("  1. Run with --kaggle (requires kaggle.json)")
        print("  2. Run with --openimages (requires fiftyone)")
        print("  3. Manually place images in:")
        print(f"     {RAW_PATH}/manual/images/")
        print(f"     {RAW_PATH}/manual/labels/")
        sys.exit(1)
    
    # Merge datasets
    if merger.merge_datasets(sources):
        merger.create_data_yaml()
        merger.print_summary()
    else:
        print("\n✗ Merge failed - check dataset formats")
        sys.exit(1)


if __name__ == "__main__":
    main()
