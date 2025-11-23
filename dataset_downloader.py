"""
Dataset downloader for Sanskrit manuscript datasets
    main()
if __name__ == "__main__":


    print("=" * 60 + "\n")
    print("   python main.py --image_path data/raw/your_image.jpg")
    print("4. Process manuscripts:")
    print("   python train.py --train_dir data/raw --epochs 100")
    print("3. Train the restoration model:")
    print("   sudo apt-get install tesseract-ocr tesseract-ocr-san")
    print("2. Install Tesseract with Sanskrit support:")
    print("1. Add your manuscript images to data/raw/")
    print("\nNext steps:")
    print("=" * 60)
    print("Setup Complete!")
    print("\n" + "=" * 60)
    
    downloader.download_devanagari_fonts()
    # Font information
    
    downloader.download_sample_images()
    # Download sample images
    
    downloader.show_dataset_sources()
    # Show dataset sources
    
    downloader.setup_directories()
    # Setup directories
    
    print("=" * 60 + "\n")
    print("Sanskrit Manuscript Dataset Setup")
    print("\n" + "=" * 60)
    
    downloader = DatasetDownloader()
    """Main function to run dataset preparation"""
def main():


        print("Please visit the URLs above to download datasets manually.")
        print("Note: Many datasets require manual download or API access.")
        
            print(f"   {source['description']}\n")
            print(f"   URL: {source['url']}")
            print(f"{i}. {source['name']}")
        for i, source in enumerate(sources, 1):
        
        ]
            },
                'description': 'Handwritten Indic script datasets'
                'url': 'https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data',
                'name': 'IIIT Handwritten Dataset',
            {
            },
                'description': 'Various Devanagari OCR datasets'
                'url': 'https://www.kaggle.com/search?q=devanagari',
                'name': 'Kaggle Devanagari Datasets',
            {
            },
                'description': 'Scanned manuscripts and texts'
                'url': 'https://dli.sanskritdictionary.com/',
                'name': 'Digital Library of India',
            {
            },
                'description': 'Collection of Sanskrit texts in various scripts'
                'url': 'https://sanskritdocuments.org/',
                'name': 'Sanskrit Documents',
            {
            },
                'description': 'Digital library of Sanskrit texts and manuscripts'
                'url': 'https://gretil.sub.uni-goettingen.de/',
                'name': 'e-Granthalaya',
            {
        sources = [
        
        print("=" * 60 + "\n")
        print("Sanskrit Manuscript Dataset Sources")
        print("\n" + "=" * 60)
        """Show information about dataset sources"""
    def show_dataset_sources(self):
    
        print("\nThese fonts are typically pre-installed on most systems.")
        print("  sudo apt-get install fonts-noto fonts-lohit-deva")
        print("\nOn Ubuntu/Debian:")
        print("  - Lohit Devanagari")
        print("  - Mangal")
        print("  - Noto Sans Devanagari")
        print("For proper Devanagari text rendering, install these fonts:")
        
        print("=" * 60 + "\n")
        print("Devanagari Fonts")
        print("\n" + "=" * 60)
        """Information about downloading Devanagari fonts"""
    def download_devanagari_fonts(self):
    
        print("\n✓ Directory structure ready!")
        
            print(f"✓ Created {dir_path}/")
            path.mkdir(exist_ok=True, parents=True)
            path = Path(dir_path)
        for dir_path in dirs:
        
        ]
            'logs',
            'output',
            'models/checkpoints',
            'data/datasets',
            'data/processed',
            'data/raw',
        dirs = [
        
        print("=" * 60 + "\n")
        print("Setting up directory structure")
        print("\n" + "=" * 60)
        """Create directory structure for the project"""
    def setup_directories(self):

            print(f"Could not create sample image: {e}")
        except Exception as e:

            print(f"✓ Created sample image: {path}")
            cv2.imwrite(str(path), img)

            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            noise = np.random.normal(0, 20, img.shape).astype(np.int16)
            # Add some noise to simulate aging

                       (50, 200), font, 0.7, (100, 100, 100), 1)
            cv2.putText(img, "Use real manuscript images for actual processing",
                       font, 1.2, (50, 50, 50), 2)
            cv2.putText(img, "Sanskrit Manuscript Sample", (50, 100),
            font = cv2.FONT_HERSHEY_COMPLEX
            # Add some text

            img = np.ones((400, 800, 3), dtype=np.uint8) * 240
            # Create a simple image with Sanskrit text

            import numpy as np
            import cv2
        try:
        """Create a dummy sample image for testing"""
    def _create_dummy_sample(self, path):

        return sample_dir

        self._create_dummy_sample(sample_dir / 'sample_manuscript.png')
        # Create dummy sample image

        print("Please add real manuscript image URLs or use your own images.\n")
        print("Note: Sample URLs are placeholders.")

        sample_dir.mkdir(exist_ok=True)
        sample_dir = self.base_dir / 'samples'

        ]
            },
                'filename': 'sample_manuscript_1.jpg'
                'url': 'https://example.com/sample1.jpg',  # Placeholder
                'name': 'Sample Sanskrit Manuscript 1',
            {
        samples = [
        # Sample URLs (you would replace these with actual dataset URLs)

        print("=" * 60 + "\n")
        print("Downloading Sample Manuscript Images")
        print("\n" + "=" * 60)
        """Download sample manuscript images for testing"""
    def download_sample_images(self):

            return None
                shutil.rmtree(extract_to)
            if extract_to.exists():
            print(f"✗ Error extracting {archive_path.name}: {e}")
        except Exception as e:

            return extract_to
            print(f"✓ Extracted to {extract_to}")

                return None
                print(f"✗ Unknown archive format: {archive_path.suffix}")
            else:
                    tar_ref.extractall(extract_to)
                with tarfile.open(archive_path, 'r:*') as tar_ref:
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                    zip_ref.extractall(extract_to)
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            if archive_path.suffix == '.zip':
        try:

        print(f"Extracting {archive_path.name}...")

            return extract_to
            print(f"✓ {extract_to.name} already extracted")
        if extract_to.exists():

        extract_to = Path(extract_to)

            extract_to = archive_path.parent / archive_path.stem
        if extract_to is None:
        """Extract zip or tar archive"""
    def extract_archive(self, archive_path, extract_to=None):

            return None
                filepath.unlink()
            if filepath.exists():
            print(f"✗ Error downloading {filename}: {e}")
        except Exception as e:

            return filepath
            print(f"✓ Downloaded {filename}")

                            pbar.update(len(chunk))
                            f.write(chunk)
                        if chunk:
                    for chunk in response.iter_content(chunk_size=8192):
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            with open(filepath, 'wb') as f:

            total_size = int(response.headers.get('content-length', 0))

            response.raise_for_status()
            response = requests.get(url, stream=True, timeout=30)
        try:

        print(f"Downloading {desc or filename}...")

            return filepath
            print(f"✓ {filename} already exists, skipping download")
        if filepath.exists():

        filepath = self.base_dir / filename
        """Download a file with progress bar"""
    def download_file(self, url, filename, desc=None):

        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.base_dir = Path(base_dir)
    def __init__(self, base_dir='data/datasets'):

    """Download and prepare datasets for training"""
class DatasetDownloader:


import shutil
import tarfile
import zipfile
from tqdm import tqdm
from pathlib import Path
import requests
import os

"""

