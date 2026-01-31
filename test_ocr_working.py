"""
Simple OCR test to demonstrate text extraction from Sanskrit manuscripts
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import pytesseract

# Method 1: Direct Tesseract (Simplest)
def test_direct_tesseract(image_path):
    print("\n" + "=" * 70)
    print("METHOD 1: DIRECT TESSERACT")
    print("=" * 70)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to load: {image_path}")
        return None

    print(f"✅ Image loaded: {img.shape}")

    # Convert BGR to RGB (OpenCV uses BGR, PIL/Tesseract expect RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_img = Image.fromarray(img_rgb)

    # Extract text with Tesseract
    text = pytesseract.image_to_string(
        pil_img,
        lang='san',           # Sanskrit language
        config='--psm 6'      # Page segmentation mode: uniform text block
    )

    print(f"✅ Extracted {len(text)} characters")
    print(f"\nText preview (first 300 chars):")
    print("-" * 70)
    print(text[:300])
    print("-" * 70)

    return text


# Method 2: Using SanskritOCR class
def test_sanskrit_ocr_class(image_path):
    print("\n" + "=" * 70)
    print("METHOD 2: USING SanskritOCR CLASS")
    print("=" * 70)

    from ocr.run_ocr import SanskritOCR

    # Initialize OCR engine
    ocr = SanskritOCR(engine='tesseract')
    print("✅ SanskritOCR initialized")

    # Extract text
    text = ocr.extract_text(
        image_path,
        preprocess=True,  # Apply preprocessing
        lang='san'        # Sanskrit
    )

    print(f"✅ Extracted {len(text)} characters")
    print(f"\nText preview (first 300 chars):")
    print("-" * 70)
    print(text[:300])
    print("-" * 70)

    return text


# Method 3: Using EnhancedSanskritOCR class (Best quality)
def test_enhanced_ocr_class(image_path):
    print("\n" + "=" * 70)
    print("METHOD 3: USING EnhancedSanskritOCR CLASS (RECOMMENDED)")
    print("=" * 70)

    from ocr.enhanced_ocr import EnhancedSanskritOCR

    # Initialize with best settings
    ocr = EnhancedSanskritOCR(engine='tesseract', device='cpu')
    print("✅ EnhancedSanskritOCR initialized")

    # Extract complete paragraph
    result = ocr.extract_complete_paragraph(
        image_path,
        preprocess=True,
        multi_pass=True
    )

    text = result['text']
    confidence = result.get('confidence', 0)
    method = result.get('method', 'unknown')
    word_count = result.get('word_count', 0)

    print(f"✅ Extracted {len(text)} characters")
    print(f"✅ Confidence: {confidence:.2f}%")
    print(f"✅ Method: {method}")
    print(f"✅ Word count: {word_count}")
    print(f"\nText preview (first 300 chars):")
    print("-" * 70)
    print(text[:300])
    print("-" * 70)

    return text


# Method 4: Using the complete ManuscriptPipeline
def test_full_pipeline(image_path):
    print("\n" + "=" * 70)
    print("METHOD 4: COMPLETE MANUSCRIPT PIPELINE")
    print("=" * 70)

    from main import ManuscriptPipeline

    # Initialize pipeline (without restoration model for speed)
    pipeline = ManuscriptPipeline(
        restoration_model_path=None,  # Skip restoration for this test
        ocr_engine='tesseract',
        translation_method='google',
        device='cpu'
    )

    print("✅ Pipeline initialized")

    # Process manuscript
    results = pipeline.process_manuscript(
        image_path,
        save_output=False
    )

    print(f"\n✅ OCR Text ({len(results['ocr_text_raw'])} chars):")
    print("-" * 70)
    print(results['ocr_text_raw'][:300])
    print("-" * 70)

    print(f"\n✅ Cleaned Text:")
    print("-" * 70)
    print(results['ocr_text_cleaned'][:300])
    print("-" * 70)

    if results['translation']:
        print(f"\n✅ Translation:")
        print("-" * 70)
        print(results['translation'][:300])
        print("-" * 70)

    return results


if __name__ == "__main__":
    # Find a test image
    test_image = Path('data/raw/test/test_0009.jpg')

    if not test_image.exists():
        # Find any image
        data_dir = Path('data/raw')
        images = list(data_dir.rglob('*.jpg')) + list(data_dir.rglob('*.png'))
        if images:
            test_image = images[0]
        else:
            print("❌ No test images found!")
            exit(1)

    print("\n" + "🔬" * 35)
    print("TESSERACT SANSKRIT OCR - WORKING EXAMPLES")
    print("🔬" * 35)
    print(f"\nTest image: {test_image}")

    # Test all methods
    try:
        text1 = test_direct_tesseract(test_image)
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        text2 = test_sanskrit_ocr_class(test_image)
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        text3 = test_enhanced_ocr_class(test_image)
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        results = test_full_pipeline(test_image)
    except Exception as e:
        print(f"❌ Method 4 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
✅ All methods should work and extract Sanskrit text from manuscripts.

If you see text extracted above, Tesseract is working correctly!

COMMON ISSUES AND SOLUTIONS:

1. "No text extracted" or empty string:
   - Check if the image path is correct
   - Ensure image is loaded properly (not None)
   - Try different PSM modes: --psm 3, --psm 6, --psm 11
   - Try preprocess=False if preprocessing is too aggressive

2. "Language not found" error:
   - Install Sanskrit: sudo apt-get install tesseract-ocr-san
   - Check with: tesseract --list-langs

3. Low quality text or garbled output:
   - Use image restoration first
   - Adjust preprocessing parameters
   - Try different language combinations: 'san+hin', 'san+eng'
   - Use EnhancedSanskritOCR with multi_pass=True

4. "Module not found" errors:
   - Install dependencies: pip install pytesseract opencv-python pillow
   - Activate virtual environment: source venv/bin/activate

BEST PRACTICE:
Use Method 3 (EnhancedSanskritOCR) for production use as it:
- Handles multiple image formats
- Applies optimal preprocessing
- Returns confidence scores
- Supports multi-pass extraction
""")

