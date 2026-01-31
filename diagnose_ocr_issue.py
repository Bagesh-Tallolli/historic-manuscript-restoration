"""
Diagnostic script to identify OCR extraction issues
"""

import sys
import os
from pathlib import Path

print("=" * 70)
print("OCR DIAGNOSTIC TOOL")
print("=" * 70)

# Test 1: Check Tesseract
print("\n1. TESSERACT CHECK")
print("-" * 70)
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    langs = pytesseract.get_languages()
    print(f"✅ Tesseract version: {version}")
    print(f"✅ Languages available: {langs}")
    has_san = 'san' in langs
    has_hin = 'hin' in langs
    print(f"✅ Sanskrit support: {'YES' if has_san else 'NO'}")
    print(f"✅ Hindi support: {'YES' if has_hin else 'NO'}")
except Exception as e:
    print(f"❌ Tesseract error: {e}")
    sys.exit(1)

# Test 2: Check OpenCV and PIL
print("\n2. IMAGE LIBRARY CHECK")
print("-" * 70)
try:
    import cv2
    import numpy as np
    from PIL import Image
    print(f"✅ OpenCV version: {cv2.__version__}")
    print(f"✅ PIL/Pillow available")
except Exception as e:
    print(f"❌ Image library error: {e}")
    sys.exit(1)

# Test 3: Load a test image
print("\n3. IMAGE LOADING TEST")
print("-" * 70)
test_image_path = 'data/raw/test/test_0009.jpg'
if not Path(test_image_path).exists():
    # Try to find any image
    data_dir = Path('data/raw')
    images = list(data_dir.rglob('*.jpg')) + list(data_dir.rglob('*.png'))
    if images:
        test_image_path = str(images[0])
    else:
        print("❌ No test images found")
        sys.exit(1)

print(f"Using: {test_image_path}")
img = cv2.imread(test_image_path)
if img is None:
    print(f"❌ Failed to load image: {test_image_path}")
    sys.exit(1)
print(f"✅ Image loaded: {img.shape}")

# Test 4: Basic Tesseract OCR
print("\n4. BASIC TESSERACT OCR TEST")
print("-" * 70)
try:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Test with different configs
    configs = {
        'san (PSM 6)': ('san', '--psm 6'),
        'hin (PSM 6)': ('hin', '--psm 6'),
        'san (PSM 3)': ('san', '--psm 3'),
        'san (PSM 11)': ('san', '--psm 11'),
    }

    for name, (lang, config) in configs.items():
        text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
        char_count = len(text.strip())
        status = "✅" if char_count > 50 else "⚠️"
        print(f"{status} {name}: {char_count} chars")
        if char_count > 0:
            preview = text.strip()[:80].replace('\n', ' ')
            print(f"   Preview: {preview}...")

except Exception as e:
    print(f"❌ OCR error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check OCR classes
print("\n5. OCR CLASS CHECK")
print("-" * 70)
try:
    from ocr.run_ocr import SanskritOCR
    ocr = SanskritOCR(engine='tesseract')
    print("✅ SanskritOCR initialized")

    text = ocr.extract_text(test_image_path, preprocess=True, lang='san')
    print(f"✅ SanskritOCR extraction: {len(text)} chars")
    if len(text) > 0:
        print(f"   Preview: {text[:100].replace(chr(10), ' ')}...")
    else:
        print("   ⚠️ No text extracted - this is the issue!")

except Exception as e:
    print(f"❌ SanskritOCR error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check EnhancedSanskritOCR
print("\n6. ENHANCED OCR CLASS CHECK")
print("-" * 70)
try:
    from ocr.enhanced_ocr import EnhancedSanskritOCR
    enhanced_ocr = EnhancedSanskritOCR(engine='tesseract')
    print("✅ EnhancedSanskritOCR initialized")

    result = enhanced_ocr.extract_complete_paragraph(test_image_path, preprocess=True)
    print(f"✅ EnhancedSanskritOCR extraction: {len(result.get('text', ''))} chars")
    print(f"   Confidence: {result.get('confidence', 0):.2f}%")
    print(f"   Method: {result.get('method', 'unknown')}")
    if result.get('text'):
        print(f"   Preview: {result['text'][:100].replace(chr(10), ' ')}...")
    else:
        print("   ⚠️ No text extracted - this is the issue!")

except Exception as e:
    print(f"❌ EnhancedSanskritOCR error: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Check preprocessing
print("\n7. PREPROCESSING CHECK")
print("-" * 70)
try:
    from ocr.preprocess import OCRPreprocessor
    preprocessor = OCRPreprocessor()
    preprocessed = preprocessor.preprocess(img)
    print(f"✅ Preprocessing successful: {preprocessed.shape}")

    # Try OCR on preprocessed image
    pil_preprocessed = Image.fromarray(preprocessed)
    text = pytesseract.image_to_string(pil_preprocessed, lang='san', config='--psm 6')
    print(f"✅ OCR on preprocessed: {len(text)} chars")

except Exception as e:
    print(f"❌ Preprocessing error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print("""
If Tesseract works directly but OCR classes fail, check:

1. Preprocessing settings might be too aggressive
2. Language parameter not being passed correctly
3. Image format conversion issues
4. PSM (Page Segmentation Mode) settings

RECOMMENDED FIXES:
- Use lang='san' for Sanskrit
- Use --psm 6 for uniform text blocks
- Try preprocess=False if preprocessing causes issues
- Check that images are properly converted to RGB/PIL format
""")

