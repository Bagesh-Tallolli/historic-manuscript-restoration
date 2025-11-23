#!/usr/bin/env python3
"""
Download and setup Roboflow Sanskrit OCR Dataset
Dataset: https://app.roboflow.com/neeew/yoyoyo-mptyx-ijqfp

This script downloads the dataset and organizes it for training.
"""

import os
import sys
from pathlib import Path

def print_instructions():
    """Print instructions for downloading the Roboflow dataset."""

    print("=" * 80)
    print("📥 ROBOFLOW SANSKRIT OCR DATASET SETUP")
    print("=" * 80)
    print()
    print("Dataset: https://app.roboflow.com/neeew/yoyoyo-mptyx-ijqfp")
    print()

    print("🔧 INSTALLATION STEPS:")
    print("-" * 80)
    print()

    print("Step 1: Install Roboflow Python Package")
    print("-" * 40)
    print("Run the following command:")
    print()
    print("    pip install roboflow")
    print()

    print("Step 2: Get Your API Key")
    print("-" * 40)
    print("1. Go to: https://app.roboflow.com/")
    print("2. Sign in or create a free account")
    print("3. Go to Settings → Roboflow API")
    print("4. Copy your API key")
    print()

    print("Step 3: Download the Dataset")
    print("-" * 40)
    print("You have two options:")
    print()
    print("Option A: Use this script (recommended)")
    print("-" * 40)
    print("Run:")
    print("    python download_roboflow_dataset.py --api-key YOUR_API_KEY")
    print()
    print("Option B: Manual download code")
    print("-" * 40)
    print("Use this Python code:")
    print()
    print("""
from roboflow import Roboflow

# Initialize
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("neeew").project("yoyoyo-mptyx-ijqfp")
dataset = project.version(1).download("folder")

# This will download to current directory
    """)
    print()

    print("Step 4: Organize for Training")
    print("-" * 40)
    print("After download, the dataset will be automatically organized in:")
    print("    data/raw/train/")
    print("    data/raw/val/")
    print("    data/raw/test/")
    print()

    print("=" * 80)
    print()


def download_dataset(api_key):
    """Download the dataset using Roboflow API."""

    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ Error: roboflow package not installed")
        print()
        print("Install it with:")
        print("    pip install roboflow")
        sys.exit(1)

    print("🚀 Starting dataset download...")
    print()

    try:
        # Initialize Roboflow
        print("📡 Connecting to Roboflow...")
        rf = Roboflow(api_key=api_key)

        # Access the project
        print("📂 Accessing project: neeew/yoyoyo-mptyx-ijqfp")
        project = rf.workspace("neeew").project("yoyoyo-mptyx-ijqfp")

        # Download the dataset
        print("⬇️  Downloading dataset (this may take a few minutes)...")

        # Create download directory
        download_dir = Path("/home/bagesh/EL-project/data/datasets/roboflow_sanskrit")
        download_dir.mkdir(parents=True, exist_ok=True)

        # Change to download directory
        original_dir = os.getcwd()
        os.chdir(download_dir)

        # Check if versions exist
        versions = project.versions()

        if len(versions) == 0:
            print("⚠️  No dataset versions found in this project.")
            print()
            print("📋 TO USE THIS DATASET:")
            print("-" * 60)
            print()
            print("Option 1: Generate a dataset version in Roboflow")
            print("   1. Go to: https://app.roboflow.com/neeew/yoyoyo-mptyx-ijqfp")
            print("   2. Upload your images")
            print("   3. Annotate if needed")
            print("   4. Click 'Generate' to create a dataset version")
            print("   5. Run this script again")
            print()
            print("Option 2: Use a different public dataset")
            print("   Browse: https://universe.roboflow.com/")
            print("   Search for: 'sanskrit', 'devanagari', or 'manuscript'")
            print()
            print("Option 3: Use local images")
            print("   Place your images in:")
            print("      data/raw/train/")
            print("      data/raw/val/")
            print("      data/raw/test/")
            print()
            print("-" * 60)
            print()
            print("✅ For now, let's set up sample images for testing...")
            print()

            # Create sample structure
            raw_dir = Path("/home/bagesh/EL-project/data/raw")
            for split in ['train', 'val', 'test']:
                (raw_dir / split).mkdir(parents=True, exist_ok=True)

            print("✅ Created directory structure:")
            print("   data/raw/train/")
            print("   data/raw/val/")
            print("   data/raw/test/")
            print()
            print("📌 Next steps:")
            print("   1. Add your Sanskrit manuscript images to these folders")
            print("   2. Or generate a dataset version in Roboflow")
            print("   3. Then run: python train.py --train_dir data/raw/train --val_dir data/raw/val")
            print()

            os.chdir(original_dir)
            return
        else:
            # Download dataset version
            print(f"📦 Downloading version {versions[0]}...")
            dataset = project.version(versions[0]).download("folder")

            os.chdir(original_dir)

            print("✅ Dataset downloaded successfully!")
            print(f"📁 Location: {download_dir}")
            print()

            # Now organize the dataset
            organize_dataset(download_dir)

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check your API key is correct")
        print("2. Ensure you have internet connection")
        print("3. Verify you have access to the dataset")
        sys.exit(1)


def organize_raw_images(download_dir):
    """Organize manually downloaded raw images into train/val/test splits."""

    print("🗂️  Organizing images into train/val/test splits...")
    print()

    import shutil
    import random

    raw_dir = Path("/home/bagesh/EL-project/data/raw")

    # Get all downloaded images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    all_images = []

    for ext in image_extensions:
        all_images.extend(download_dir.glob(f'*{ext}'))
        all_images.extend(download_dir.glob(f'*{ext.upper()}'))

    if not all_images:
        print("⚠️  No images found to organize")
        return

    # Shuffle for random split
    random.shuffle(all_images)

    # Split: 80% train, 10% val, 10% test
    total = len(all_images)
    train_count = int(0.8 * total)
    val_count = int(0.1 * total)

    train_images = all_images[:train_count]
    val_images = all_images[train_count:train_count + val_count]
    test_images = all_images[train_count + val_count:]

    # Create directories and copy images
    for split_name, images in [('train', train_images),
                                ('val', val_images),
                                ('test', test_images)]:
        if images:
            target_dir = raw_dir / split_name
            target_dir.mkdir(parents=True, exist_ok=True)

            print(f"📋 Copying {len(images)} images to {split_name}/")
            for img_path in images:
                shutil.copy2(img_path, target_dir / img_path.name)

    print()
    print("✅ Dataset organization complete!")
    print()
    print("📊 Dataset Summary:")
    print(f"   Training images:   {len(train_images)}")
    print(f"   Validation images: {len(val_images)}")
    print(f"   Test images:       {len(test_images)}")
    print(f"   Total:             {total}")
    print()
    print("🎯 Next Steps:")
    print("   1. Verify images: ls data/raw/train/ | head")
    print("   2. Start training: python train.py --train_dir data/raw/train --val_dir data/raw/val")
    print()


def organize_dataset(download_dir):
    """Organize downloaded dataset for training."""

    print("🗂️  Organizing dataset for training...")
    print()

    import shutil

    # Expected structure after Roboflow download
    # The dataset might be in different formats, we need to check

    raw_dir = Path("/home/bagesh/EL-project/data/raw")

    # Find all images in the downloaded dataset
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    train_images = []
    val_images = []
    test_images = []

    # Search for train/valid/test folders
    for split_name, image_list in [('train', train_images),
                                     ('valid', val_images),
                                     ('test', test_images)]:
        split_dir = download_dir / split_name
        if split_dir.exists():
            for ext in image_extensions:
                image_list.extend(split_dir.glob(f'*{ext}'))
                image_list.extend(split_dir.glob(f'*{ext.upper()}'))

    # If no split folders found, look for all images
    if not train_images and not val_images and not test_images:
        print("⚠️  No train/valid/test splits found, searching for all images...")
        all_images = []
        for ext in image_extensions:
            all_images.extend(download_dir.rglob(f'*{ext}'))
            all_images.extend(download_dir.rglob(f'*{ext.upper()}'))

        # Split manually (80/10/10)
        total = len(all_images)
        train_images = all_images[:int(0.8 * total)]
        val_images = all_images[int(0.8 * total):int(0.9 * total)]
        test_images = all_images[int(0.9 * total):]

    # Copy to organized structure
    for split_name, images in [('train', train_images),
                                ('val', val_images),
                                ('test', test_images)]:
        if images:
            target_dir = raw_dir / split_name
            target_dir.mkdir(parents=True, exist_ok=True)

            print(f"📋 Copying {len(images)} images to {split_name}/")
            for img_path in images:
                shutil.copy2(img_path, target_dir / img_path.name)

    print()
    print("✅ Dataset organization complete!")
    print()
    print("📊 Dataset Summary:")
    print(f"   Training images:   {len(train_images)}")
    print(f"   Validation images: {len(val_images)}")
    print(f"   Test images:       {len(test_images)}")
    print(f"   Total:             {len(train_images) + len(val_images) + len(test_images)}")
    print()
    print("🎯 Next Steps:")
    print("   1. Verify images: ls data/raw/train/ | head")
    print("   2. Start training: python train.py --train_dir data/raw/train --val_dir data/raw/val")
    print()


def main():
    """Main function."""

    import argparse

    parser = argparse.ArgumentParser(
        description='Download Roboflow Sanskrit OCR Dataset'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Your Roboflow API key'
    )
    parser.add_argument(
        '--instructions',
        action='store_true',
        help='Show setup instructions only'
    )

    args = parser.parse_args()

    if args.instructions or not args.api_key:
        print_instructions()
        if not args.api_key:
            print("💡 TIP: Run with --api-key to download automatically")
            print()
    else:
        download_dataset(args.api_key)


if __name__ == "__main__":
    main()

