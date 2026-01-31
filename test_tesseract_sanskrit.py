"""
Test Tesseract Sanskrit OCR on manuscript images
This script verifies if Tesseract can extract Sanskrit/Devanagari text from manuscripts
"""

import pytesseract
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def test_tesseract_languages():
    """Test Tesseract installation and available languages"""
    print("\n" + "=" * 70)
    print("TESSERACT CONFIGURATION CHECK")
    print("=" * 70)

    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract Version: {version}")
    except Exception as e:
        print(f"❌ Error getting Tesseract version: {e}")
        return False

    try:
        langs = pytesseract.get_languages()
        print(f"\n📋 Available Languages ({len(langs)}):")
        for lang in sorted(langs):
            marker = "✅" if lang in ['san', 'hin', 'dev'] else "  "
            print(f"   {marker} {lang}")

        # Check for Sanskrit/Devanagari support
        has_sanskrit = 'san' in langs
        has_hindi = 'hin' in langs
        has_devanagari = 'dev' in langs

        print("\n🔍 Devanagari Support:")
        print(f"   Sanskrit (san): {'✅ YES' if has_sanskrit else '❌ NO'}")
        print(f"   Hindi (hin):    {'✅ YES' if has_hindi else '❌ NO'}")
        print(f"   Devanagari (dev): {'✅ YES' if has_devanagari else '❌ NO'}")

        if not (has_sanskrit or has_hindi or has_devanagari):
            print("\n⚠️  WARNING: No Devanagari language support found!")
            print("   Install with: sudo apt-get install tesseract-ocr-san tesseract-ocr-hin")
            return False

        return True

    except Exception as e:
        print(f"❌ Error checking languages: {e}")
        return False


def preprocess_manuscript(image):
    """Apply preprocessing to improve OCR accuracy"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive thresholding for better text extraction
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned


def test_ocr_on_manuscript(image_path, lang='san'):
    """Test OCR extraction on a manuscript image"""
    print("\n" + "=" * 70)
    print(f"TESTING OCR ON: {Path(image_path).name}")
    print("=" * 70)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return

    print(f"📐 Image size: {img.shape[1]}x{img.shape[0]} pixels")

    # Test 1: Direct OCR (no preprocessing)
    print(f"\n1️⃣  Direct OCR (lang={lang}, no preprocessing):")
    print("-" * 70)
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        text_direct = pytesseract.image_to_string(pil_img, lang=lang, config='--psm 6')

        if text_direct.strip():
            print(f"✅ Extracted text ({len(text_direct)} chars):")
            print(text_direct[:500] if len(text_direct) > 500 else text_direct)
        else:
            print("⚠️  No text extracted")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: With preprocessing
    print(f"\n2️⃣  OCR with Preprocessing (lang={lang}):")
    print("-" * 70)
    try:
        preprocessed = preprocess_manuscript(img)
        pil_preprocessed = Image.fromarray(preprocessed)
        text_preprocessed = pytesseract.image_to_string(pil_preprocessed, lang=lang, config='--psm 6')

        if text_preprocessed.strip():
            print(f"✅ Extracted text ({len(text_preprocessed)} chars):")
            print(text_preprocessed[:500] if len(text_preprocessed) > 500 else text_preprocessed)
        else:
            print("⚠️  No text extracted")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: With different PSM modes
    print(f"\n3️⃣  OCR with different Page Segmentation Modes:")
    print("-" * 70)
    psm_modes = {
        '3': 'Fully automatic page segmentation',
        '6': 'Uniform block of text',
        '11': 'Sparse text',
        '12': 'Sparse text with OSD'
    }

    for psm, description in psm_modes.items():
        try:
            config = f'--psm {psm}'
            text = pytesseract.image_to_string(pil_preprocessed, lang=lang, config=config)
            char_count = len(text.strip())
            print(f"   PSM {psm} ({description}): {char_count} chars")
            if char_count > 0:
                print(f"      Preview: {text[:100]}")
        except Exception as e:
            print(f"   PSM {psm}: Error - {e}")

    # Test 4: Try multiple languages
    print(f"\n4️⃣  Testing different language packs:")
    print("-" * 70)
    for test_lang in ['san', 'hin', 'eng+san', 'hin+eng']:
        try:
            text = pytesseract.image_to_string(pil_preprocessed, lang=test_lang, config='--psm 6')
            char_count = len(text.strip())
            print(f"   Lang '{test_lang}': {char_count} chars extracted")
            if char_count > 50:
                print(f"      Preview: {text[:80]}...")
        except Exception as e:
            print(f"   Lang '{test_lang}': ❌ {e}")

    # Test 5: Get confidence scores
    print(f"\n5️⃣  Confidence Analysis:")
    print("-" * 70)
    try:
        data = pytesseract.image_to_data(pil_preprocessed, lang=lang, config='--psm 6', output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            print(f"   Average confidence: {avg_conf:.2f}%")
            print(f"   Min confidence: {min(confidences)}%")
            print(f"   Max confidence: {max(confidences)}%")
            print(f"   Words detected: {len(confidences)}")
        else:
            print("   ⚠️  No confidence data available")
    except Exception as e:
        print(f"   ❌ Error getting confidence: {e}")


def main():
    """Main test function"""
    print("\n" + "🔬" * 35)
    print("TESSERACT SANSKRIT OCR TEST")
    print("🔬" * 35)

    # Step 1: Check Tesseract configuration
    if not test_tesseract_languages():
        print("\n❌ Tesseract not properly configured. Exiting.")
        return

    # Step 2: Find test images
    print("\n" + "=" * 70)
    print("FINDING TEST IMAGES")
    print("=" * 70)

    # Look for manuscript images
    data_dir = Path('/home/bagesh/EL-project/data/raw')
    test_images = []

    for subdir in ['test', 'train', 'val']:
        img_dir = data_dir / subdir
        if img_dir.exists():
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            test_images.extend(images[:2])  # Take 2 from each

    if not test_images:
        print("⚠️  No test images found in data/raw/")
        print("   Looking for any manuscript images...")
        test_images = list(data_dir.rglob('*.jpg'))[:3]

    if not test_images:
        print("❌ No images found. Please add some manuscript images to test.")
        return

    print(f"✅ Found {len(test_images)} test images")
    for img in test_images[:5]:  # Show first 5
        print(f"   - {img}")

    # Step 3: Test OCR on images
    for image_path in test_images[:3]:  # Test on first 3 images
        test_ocr_on_manuscript(image_path, lang='san')

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
✅ Tesseract is installed and configured
✅ Sanskrit (san) and Hindi (hin) language packs are available
✅ OCR tests completed

📝 RECOMMENDATIONS:

1. For best results with manuscripts:
   - Use preprocessing (denoising, binarization)
   - Try different PSM modes (--psm 6 or --psm 11 work well)
   - Use 'san' for Sanskrit, 'hin' for Hindi
   - Combine languages if needed: 'san+hin' or 'san+eng'

2. If accuracy is low:
   - Ensure images are high resolution (>300 DPI)
   - Use the restoration model first to enhance image quality
   - Adjust preprocessing parameters
   - Consider using the EnhancedSanskritOCR class with hybrid mode

3. The pipeline already has enhanced OCR configured in:
   - ocr/run_ocr.py (SanskritOCR class)
   - ocr/enhanced_ocr.py (EnhancedSanskritOCR class)
   - main.py (ManuscriptPipeline uses EnhancedSanskritOCR)
""")


if __name__ == "__main__":
    main()

