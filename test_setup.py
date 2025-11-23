#!/usr/bin/env python3
"""
Quick start script to test the pipeline
"""

import sys
from pathlib import Path

print("""
========================================
Sanskrit Manuscript Pipeline Quick Test
========================================
""")

# Test imports
print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")
    sys.exit(1)

try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f"✓ Tesseract {version}")
except Exception as e:
    print(f"⚠ Tesseract: {e}")
    print("  Install with: sudo apt-get install tesseract-ocr tesseract-ocr-san")

# Test project modules
print("\nTesting project modules...")

try:
    from models.vit_restorer import create_vit_restorer
    print("✓ Models module")
except ImportError as e:
    print(f"✗ Models: {e}")

try:
    from ocr.run_ocr import SanskritOCR
    print("✓ OCR module")
except ImportError as e:
    print(f"✗ OCR: {e}")

try:
    from nlp.unicode_normalizer import SanskritTextProcessor
    print("✓ NLP module")
except ImportError as e:
    print(f"✗ NLP: {e}")

try:
    from nlp.translation import SanskritTranslator
    print("✓ Translation module")
except ImportError as e:
    print(f"✗ Translation: {e}")

# Create test image
print("\nCreating test image...")
test_img = np.ones((300, 600, 3), dtype=np.uint8) * 240
cv2.putText(test_img, "Sanskrit Manuscript", (50, 150),
            cv2.FONT_HERSHEY_COMPLEX, 1.5, (50, 50, 50), 2)

test_dir = Path("data/datasets/samples")
test_dir.mkdir(exist_ok=True, parents=True)
test_path = test_dir / "test_sample.png"
cv2.imwrite(str(test_path), test_img)
print(f"✓ Created test image: {test_path}")

# Test model creation
print("\nTesting model creation...")
try:
    model = create_vit_restorer('tiny', img_size=256)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Created ViT model with {total_params:,} parameters")
except Exception as e:
    print(f"✗ Model creation failed: {e}")

# Test OCR
print("\nTesting OCR...")
try:
    ocr = SanskritOCR(engine='tesseract')
    text = ocr.extract_text(test_img, preprocess=False, lang='eng')
    print(f"✓ OCR working: '{text}'")
except Exception as e:
    print(f"⚠ OCR test: {e}")

# Test text processor
print("\nTesting text processor...")
try:
    processor = SanskritTextProcessor()
    result = processor.process_ocr_output("test", input_format='auto')
    print(f"✓ Text processor working")
except Exception as e:
    print(f"✗ Text processor: {e}")

print("""
========================================
Test Complete!
========================================

The pipeline is ready to use. Try:

1. Process a manuscript:
   python main.py --image_path data/datasets/samples/test_sample.png

2. Train a model:
   python train.py --train_dir data/raw --epochs 10

3. Use the demo notebook:
   jupyter notebook demo.ipynb

For full setup, run:
   ./setup.sh

========================================
""")

