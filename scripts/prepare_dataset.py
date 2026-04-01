#!/usr/bin/env python3
"""
Prepare Sri Lankan Elephant Dataset for YOLO training.
Uses SAM (Segment Anything Model) or center-crop bounding boxes for labeling.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Configuration
BASE_PATH = Path("/Volumes/Extended Storage/Elephant-Detection")
SOURCE_DIR = BASE_PATH / "sri-lankan-wild-elephant-dataset"
DATASET_DIR = BASE_PATH / "datasets"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.15
TEST_SPLIT = 0.05

def prepare_directories():
    """Create dataset directory structure."""
    for split in ['train', 'val', 'test']:
        (DATASET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    print("✓ Created directory structure")

def get_image_files():
    """Get all valid image files from source directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = []
    
    for f in SOURCE_DIR.iterdir():
        if f.is_file() and f.suffix in extensions:
            # Skip system files
            if f.name.startswith('.'):
                continue
            images.append(f)
    
    print(f"✓ Found {len(images)} images")
    return images

def create_smart_label(img_path):
    """
    Create a smart bounding box label for an elephant image.
    Uses edge detection to find the main subject.
    Returns (x_center, y_center, width, height) normalized.
    """
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            return None
            
        h, w = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)
            
            # Add padding (15% on each side)
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            bw = min(w - x, bw + 2 * pad_x)
            bh = min(h - y, bh + 2 * pad_y)
            
            # Only use if the detected region is significant (>10% of image)
            if (bw * bh) / (w * h) > 0.1:
                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                width = bw / w
                height = bh / h
                return (x_center, y_center, width, height)
        
        # Fallback: center crop covering 80% of image
        return (0.5, 0.5, 0.85, 0.85)
        
    except Exception as e:
        print(f"  Warning: Could not process {img_path.name}: {e}")
        return (0.5, 0.5, 0.85, 0.85)

def split_dataset(images):
    """Split images into train/val/test sets."""
    random.seed(42)
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    print(f"\n📊 Dataset split:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"  Test:  {len(splits['test'])} images")
    
    return splits

def save_dataset(splits):
    """Copy images and create labels for dataset."""
    for split_name, images in splits.items():
        img_dir = DATASET_DIR / split_name / 'images'
        lbl_dir = DATASET_DIR / split_name / 'labels'
        
        total = len(images)
        for i, img_path in enumerate(images):
            # Copy image
            dst_img = img_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Create label
            label = create_smart_label(img_path)
            if label:
                label_name = img_path.stem + '.txt'
                dst_lbl = lbl_dir / label_name
                x, y, w, h = label
                with open(dst_lbl, 'w') as f:
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            if (i + 1) % 500 == 0:
                print(f"  {split_name}: {i + 1}/{total} processed")
        
        print(f"✓ Saved {split_name} split: {len(images)} images")

def main():
    print("=" * 60)
    print("PREPARING ELEPHANT DATASET FOR YOLO TRAINING")
    print("=" * 60)
    
    # Prepare directories
    prepare_directories()
    
    # Get images
    images = get_image_files()
    
    if len(images) == 0:
        print("❌ No images found!")
        return
    
    # Split dataset
    splits = split_dataset(images)
    
    # Save dataset with smart labels
    print("\n💾 Processing and saving dataset...")
    save_dataset(splits)
    
    print("\n" + "=" * 60)
    print("✅ DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nDataset ready at: {DATASET_DIR}")
    print("You can now run training with YOLOv11!")

if __name__ == "__main__":
    main()
