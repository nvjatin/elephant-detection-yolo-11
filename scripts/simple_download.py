"""
Simple Elephant Dataset Downloader
Downloads a public elephant detection dataset without requiring API keys.

Usage:
    python scripts/simple_download.py
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Paths
BASE_PATH = Path("/Volumes/Extended Storage/Elephant-Detection")
DATASET_PATH = BASE_PATH / "datasets"

# Public elephant detection datasets (direct download links)
DATASETS = [
    {
        "name": "elephant-detection-roboflow",
        "url": "https://universe.roboflow.com/ds/pJ6PxjxKfP?key=0cWE7ep0i7",
        "format": "yolov8"
    }
]


def download_with_progress(url: str, dest: Path):
    """Download file with progress indicator."""
    print(f"Downloading: {url}")
    
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest, progress)
    print()


def setup_yolo_structure():
    """Create YOLO directory structure."""
    for split in ["train", "val", "test"]:
        (DATASET_PATH / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_PATH / split / "labels").mkdir(parents=True, exist_ok=True)
    print("✓ Created YOLO directory structure")


def download_sample_dataset():
    """Download sample elephant images from web for testing."""
    import urllib.request
    
    # Sample elephant images (public domain)
    sample_images = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/1280px-African_Bush_Elephant.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Elephant_%40_kabridge.jpg/1280px-Elephant_%40_kabbridge.jpg",
    ]
    
    img_dir = DATASET_PATH / "train" / "images"
    label_dir = DATASET_PATH / "train" / "labels"
    
    for i, url in enumerate(sample_images):
        try:
            img_path = img_dir / f"elephant_{i:03d}.jpg"
            print(f"Downloading sample image {i+1}...")
            urllib.request.urlretrieve(url, img_path)
            
            # Create dummy label (full image as elephant for demo)
            label_path = label_dir / f"elephant_{i:03d}.txt"
            label_path.write_text("0 0.5 0.5 0.9 0.9\n")
            
        except Exception as e:
            print(f"  Failed: {e}")


def create_data_yaml():
    """Create data.yaml configuration."""
    yaml_content = f"""# Elephant Detection Dataset
path: {DATASET_PATH}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: elephant
"""
    (DATASET_PATH / "data.yaml").write_text(yaml_content)
    print(f"✓ Created data.yaml")


def main():
    print("=" * 60)
    print("ELEPHANT DATASET SETUP")
    print("=" * 60)
    print(f"Target: {DATASET_PATH}")
    print()
    
    # Verify SSD
    if not BASE_PATH.exists():
        print(f"✗ SSD not mounted: {BASE_PATH}")
        sys.exit(1)
    
    setup_yolo_structure()
    
    # Try to download using roboflow CLI if available
    try:
        print("\nAttempting Roboflow download...")
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "roboflow", "-q"
        ], capture_output=True)
        
        from roboflow import Roboflow
        
        # Use a public dataset
        rf = Roboflow()
        project = rf.workspace("animal-detection-kwxs3").project("elephant-detection-6ojs7")
        version = project.version(1)
        dataset = version.download("yolov8", location=str(DATASET_PATH))
        
        print("✓ Downloaded from Roboflow!")
        return
        
    except Exception as e:
        print(f"Roboflow failed: {e}")
    
    # Fallback: provide instructions
    print("\n" + "=" * 60)
    print("MANUAL SETUP REQUIRED")
    print("=" * 60)
    print("""
Since automatic download failed, please manually download a dataset:

Option 1: Roboflow (Recommended)
--------------------------------
1. Go to: https://universe.roboflow.com/search?q=elephant%20detection
2. Find an elephant detection dataset
3. Click "Download" → Select "YOLOv8" format
4. Extract to: /Volumes/Extended Storage/Elephant-Detection/datasets/

Option 2: Open Images via FiftyOne
----------------------------------
pip install fiftyone
python -c "
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    'open-images-v7',
    split='train',
    classes=['Elephant'],
    max_samples=500,
)
dataset.export(
    '/Volumes/Extended Storage/Elephant-Detection/datasets',
    dataset_type=fo.types.YOLOv5Dataset,
)
"

Option 3: Kaggle
----------------
1. Install: pip install kaggle
2. Get API key from https://www.kaggle.com/settings
3. Place kaggle.json in ~/.kaggle/
4. Run: kaggle datasets download -d jayaprakashpondy/elephant-face -p datasets/

After downloading, ensure structure is:
    datasets/
    ├── train/images/  (elephant images)
    ├── train/labels/  (YOLO format .txt files)
    ├── val/images/
    ├── val/labels/
    └── data.yaml
""")
    
    create_data_yaml()


if __name__ == "__main__":
    main()
