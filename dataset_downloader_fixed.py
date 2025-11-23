#!/usr/bin/env python3
"""
Dataset downloader and setup for Sanskrit manuscript restoration
Helps users download sample datasets and set up the project structure
"""

import os
from pathlib import Path
import urllib.request
import shutil

class DatasetDownloader:
    def __init__(self):
        self.base_dir = Path("/home/bagesh/EL-project")
        self.data_dir = self.base_dir / "data"

    def setup_directories(self):
        """Create necessary directories"""
        print("📁 Setting up directory structure...")

        directories = [
            self.data_dir / "raw" / "train",
            self.data_dir / "raw" / "val",
            self.data_dir / "raw" / "test",
            self.data_dir / "processed",
            self.data_dir / "datasets" / "samples",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")

        print()

    def show_dataset_sources(self):
        """Display available dataset sources"""
        print("=" * 80)
        print("📚 SANSKRIT MANUSCRIPT DATASET SOURCES")
        print("=" * 80)
        print()

        sources = [
            {
                'name': 'Digital Library of India (DLI)',
                'url': 'https://dli.iiit.ac.in/',
                'description': 'Large collection of digitized Indian books including Sanskrit texts',
                'access': 'Free, requires registration'
            },
            {
                'name': 'Internet Archive - Sanskrit Collection',
                'url': 'https://archive.org/search.php?query=sanskrit+manuscript',
                'description': 'Digitized Sanskrit manuscripts from various institutions',
                'access': 'Free, public domain'
            },
            {
                'name': 'GRETIL (Göttingen Register of Electronic Texts)',
                'url': 'http://gretil.sub.uni-goettingen.de/',
                'description': 'Electronic texts of Indian literature',
                'access': 'Free academic resource'
            },
            {
                'name': 'Roboflow Universe - Devanagari',
                'url': 'https://universe.roboflow.com/search?q=devanagari',
                'description': 'Pre-labeled Devanagari datasets',
                'access': 'Requires Roboflow account'
            },
            {
                'name': 'Kaggle - Sanskrit/Devanagari Datasets',
                'url': 'https://www.kaggle.com/search?q=sanskrit',
                'description': 'Various Sanskrit and Devanagari datasets',
                'access': 'Free with Kaggle account'
            },
            {
                'name': 'GitHub - Sanskrit OCR Datasets',
                'url': 'https://github.com/search?q=sanskrit+ocr+dataset',
                'description': 'Open-source Sanskrit OCR datasets',
                'access': 'Free, open source'
            }
        ]

        for i, source in enumerate(sources, 1):
            print(f"{i}. {source['name']}")
            print(f"   URL: {source['url']}")
            print(f"   Description: {source['description']}")
            print(f"   Access: {source['access']}")
            print()

        print("=" * 80)
        print()

    def download_sample_images(self):
        """Instructions for downloading sample images"""
        print("=" * 80)
        print("📥 DOWNLOADING SAMPLE IMAGES")
        print("=" * 80)
        print()

        print("To get started quickly, you can:")
        print()
        print("Option 1: Use existing test sample")
        print("-" * 40)
        sample_file = self.data_dir / "datasets" / "samples" / "test_sample.png"
        if sample_file.exists():
            print(f"   ✅ Sample exists: {sample_file}")
            print()
            print("   To use for testing:")
            print(f"   cp {sample_file} data/raw/train/sample1.png")
            print(f"   cp {sample_file} data/raw/train/sample2.png")
            print(f"   cp {sample_file} data/raw/val/sample1.png")
        else:
            print("   ⚠️ No sample file found")
        print()

        print("Option 2: Download from Internet Archive")
        print("-" * 40)
        print("   1. Visit: https://archive.org/search.php?query=sanskrit+manuscript")
        print("   2. Find a manuscript you like")
        print("   3. Download images (JPG or PNG)")
        print("   4. Save to: data/raw/train/")
        print()

        print("Option 3: Use your own images")
        print("-" * 40)
        print("   Place your Sanskrit manuscript images in:")
        print(f"   - Training:   {self.data_dir / 'raw' / 'train'}/")
        print(f"   - Validation: {self.data_dir / 'raw' / 'val'}/")
        print(f"   - Testing:    {self.data_dir / 'raw' / 'test'}/")
        print()

        print("=" * 80)
        print()

    def check_status(self):
        """Check current dataset status"""
        print("=" * 80)
        print("📊 DATASET STATUS")
        print("=" * 80)
        print()

        # Check train directory
        train_dir = self.data_dir / "raw" / "train"
        train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
        print(f"Training images:   {len(train_images)}")

        # Check val directory
        val_dir = self.data_dir / "raw" / "val"
        val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
        print(f"Validation images: {len(val_images)}")

        # Check test directory
        test_dir = self.data_dir / "raw" / "test"
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        print(f"Test images:       {len(test_images)}")
        print()

        total = len(train_images) + len(val_images) + len(test_images)

        if total == 0:
            print("⚠️  No images found!")
            print()
            print("Next steps:")
            print("   1. Download or copy images to the directories above")
            print("   2. See options by running this script")
        elif len(train_images) >= 20 and len(val_images) >= 5:
            print("✅ Sufficient images for training!")
            print()
            print("Ready to train:")
            print("   python3 train.py --train_dir data/raw/train --val_dir data/raw/val")
        else:
            print("⚠️  Need more images")
            print()
            print("Recommended:")
            print("   - Training:   20+ images")
            print("   - Validation: 5+ images")

        print()
        print("=" * 80)
        print()

def main():
    """Main function"""
    downloader = DatasetDownloader()

    print()
    print("=" * 80)
    print("🕉️  SANSKRIT MANUSCRIPT DATASET SETUP")
    print("=" * 80)
    print()

    # Setup directories
    downloader.setup_directories()

    # Show dataset sources
    downloader.show_dataset_sources()

    # Show download options
    downloader.download_sample_images()

    # Check current status
    downloader.check_status()

    print("=" * 80)
    print("📖 For more information, see:")
    print("   - DATASET_REQUIREMENTS.md")
    print("   - ROBOFLOW_STATUS.md")
    print("   - SETUP_COMPLETE.md")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()

