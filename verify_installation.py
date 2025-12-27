#!/usr/bin/env python3
"""
Verify that all required dependencies are installed correctly.
Run this after installing requirements.txt to ensure everything is working.
"""

import sys
import importlib
from typing import List, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and can be imported.
    
    Args:
        package_name: Display name of the package
        import_name: Python import name (if different from package_name)
    
    Returns:
        (success, message)
    """
    if import_name is None:
        import_name = package_name.lower().replace('-', '_')
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"✓ {package_name} ({version})"
    except ImportError as e:
        return False, f"✗ {package_name}: {str(e)}"
    except Exception as e:
        return False, f"✗ {package_name}: {str(e)}"

def main():
    print("=" * 60)
    print("Sanskrit Manuscript Restoration Pipeline")
    print("Dependency Verification")
    print("=" * 60)
    print()
    
    # Core packages to check
    packages = [
        # Deep Learning
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("Transformers", "transformers"),
        
        # Image Processing
        ("NumPy", "numpy"),
        ("OpenCV", "cv2"),
        ("Pillow", "PIL"),
        ("scikit-image", "skimage"),
        
        # Translation (NEW - fixed dependency)
        ("deep-translator", "deep_translator"),
        
        # Dataset Management
        ("Roboflow", "roboflow"),
        ("Kaggle", "kaggle"),
        ("BeautifulSoup4", "bs4"),
        
        # NLP
        ("SentencePiece", "sentencepiece"),
        ("indic-nlp-library", "indicnlp"),
        ("aksharamukha", "aksharamukha"),
        
        # OCR
        ("pytesseract", "pytesseract"),
        
        # Utilities
        ("Pandas", "pandas"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("tqdm", "tqdm"),
        ("YAML", "yaml"),
        
        # ML Tools
        ("scikit-learn", "sklearn"),
        ("LPIPS", "lpips"),
        ("timm", "timm"),
        ("einops", "einops"),
        
        # Web Interface
        ("Streamlit", "streamlit"),
        
        # Experiment Tracking
        ("TensorBoard", "tensorboard"),
        ("Weights & Biases", "wandb"),
        
        # Jupyter
        ("Jupyter", "jupyter"),
        ("IPyWidgets", "ipywidgets"),
    ]
    
    print("Checking Python packages...")
    print()
    
    all_success = True
    failed_packages = []
    
    for package_name, import_name in packages:
        success, message = check_package(package_name, import_name)
        print(message)
        if not success:
            all_success = False
            failed_packages.append(package_name)
    
    print()
    print("=" * 60)
    
    # Check Tesseract
    print()
    print("Checking system dependencies...")
    print()
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract OCR ({version})")
    except Exception as e:
        print(f"⚠ Tesseract OCR: {str(e)}")
        print("  Install with:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-san")
        print("  macOS: brew install tesseract tesseract-lang")
    
    # Check GPU availability
    print()
    print("Checking GPU availability...")
    print()
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("ℹ CUDA not available (CPU only)")
            print("  For GPU support, install CUDA-enabled PyTorch")
    except:
        print("⚠ Could not check CUDA availability")
    
    # Test key functionality
    print()
    print("=" * 60)
    print()
    print("Testing key functionality...")
    print()
    
    # Test deep-translator functionality
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='en')
        print("✓ Translation module (deep-translator) working")
    except Exception as e:
        print(f"✗ Translation module: {e}")
        all_success = False
    
    # Test roboflow compatibility
    try:
        from roboflow import Roboflow
        print("✓ Roboflow import successful")
    except Exception as e:
        print(f"✗ Roboflow: {e}")
        all_success = False
    
    # Test project modules
    print()
    print("Checking project modules...")
    print()
    
    project_modules = [
        ("Models", "models.vit_restorer"),
        ("OCR", "ocr.run_ocr"),
        ("NLP", "nlp.unicode_normalizer"),
        ("Translation", "nlp.translation"),
        ("Utils", "utils.metrics"),
    ]
    
    for name, module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {name} module")
        except Exception as e:
            print(f"✗ {name} module: {e}")
            all_success = False
    
    # Final summary
    print()
    print("=" * 60)
    print()
    
    if all_success:
        print("✅ All required dependencies are installed correctly!")
        print()
        print("You can now:")
        print("  • Run the pipeline: python main.py --image_path <path>")
        print("  • Train a model: python train.py")
        print("  • Start the web app: streamlit run app.py")
        print("  • Open the demo: jupyter notebook demo.ipynb")
        print()
        return 0
    else:
        print("⚠️  Some dependencies are missing or have issues.")
        print()
        if failed_packages:
            print("Failed packages:")
            for pkg in failed_packages:
                print(f"  • {pkg}")
            print()
        print("Try reinstalling:")
        print("  pip install -r requirements.txt")
        print()
        print("For help, see INSTALLATION_GUIDE.md")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
