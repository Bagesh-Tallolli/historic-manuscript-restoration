#!/usr/bin/env python3
"""
Direct OCR Test Script - Test Tesseract without Streamlit
This helps diagnose OCR issues by testing directly
"""

import pytesseract
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import sys

def preprocess_image(image, contrast=1.5, sharpness=2.0, denoise=True, threshold=True):
    """Preprocess image for better OCR"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)

    # Convert to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Apply denoising
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Apply adaptive thresholding
    if threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    # Convert back to PIL
    return Image.fromarray(gray)

def test_ocr(image_path):
    """Test OCR with different configurations"""
    print("="*60)
    print("OCR Direct Test")
    print("="*60)
    print(f"\n📁 Image: {image_path}\n")

    try:
        # Load image
        print("Loading image...")
        image = Image.open(image_path)
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]} pixels, Mode: {image.mode}")

        # Test configurations
        configs = [
            ("Original Image, PSM 3, OEM 1", image, "san", "--oem 1 --psm 3"),
            ("Original Image, PSM 6, OEM 1", image, "san", "--oem 1 --psm 6"),
            ("Preprocessed, PSM 3, OEM 1", preprocess_image(image), "san", "--oem 1 --psm 3"),
            ("Preprocessed, PSM 6, OEM 1", preprocess_image(image), "san", "--oem 1 --psm 6"),
            ("Preprocessed High Contrast, PSM 6", preprocess_image(image, contrast=2.5, sharpness=3.0), "san", "--oem 1 --psm 6"),
        ]

        for i, (desc, img, lang, config) in enumerate(configs, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}: {desc}")
            print(f"Config: lang={lang}, {config}")
            print("-"*60)

            try:
                text = pytesseract.image_to_string(img, lang=lang, config=config)

                if text.strip():
                    print(f"✅ SUCCESS - Extracted {len(text)} chars, {len(text.split())} words")
                    print("\nExtracted Text:")
                    print("-"*60)
                    print(text[:500])  # First 500 chars
                    if len(text) > 500:
                        print(f"\n... ({len(text) - 500} more characters)")
                else:
                    print("⚠️  NO TEXT DETECTED")

            except Exception as e:
                print(f"❌ ERROR: {e}")

        print("\n" + "="*60)
        print("Test Complete")
        print("="*60)

    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_ocr(sys.argv[1])
    else:
        print("Usage: python test_ocr_direct.py <image_path>")
        print("\nExample:")
        print("  python test_ocr_direct.py /path/to/sanskrit_image.jpg")

