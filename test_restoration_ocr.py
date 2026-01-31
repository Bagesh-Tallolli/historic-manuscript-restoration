"""
Test OCR on restored images to diagnose the issue
"""

import cv2
import numpy as np
from pathlib import Path
import pytesseract
from PIL import Image

print("=" * 70)
print("TESTING OCR AFTER IMAGE RESTORATION")
print("=" * 70)

# Find a test image
test_image_path = Path('data/raw/test/test_0009.jpg')
if not test_image_path.exists():
    images = list(Path('data/raw').rglob('*.jpg'))
    if images:
        test_image_path = images[0]
    else:
        print("❌ No test images found")
        exit(1)

print(f"\nTest image: {test_image_path}")

# Load original image
original_img = cv2.imread(str(test_image_path))
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
print(f"✅ Original image loaded: {original_img_rgb.shape}, dtype={original_img_rgb.dtype}")

# Test 1: OCR on original image
print("\n" + "=" * 70)
print("TEST 1: OCR ON ORIGINAL IMAGE (BASELINE)")
print("=" * 70)
pil_original = Image.fromarray(original_img_rgb)
text_original = pytesseract.image_to_string(pil_original, lang='san', config='--psm 6')
print(f"✅ Extracted {len(text_original)} characters")
print(f"Preview: {text_original[:150]}")

# Test 2: Check if restoration model exists
print("\n" + "=" * 70)
print("TEST 2: LOADING RESTORATION MODEL")
print("=" * 70)

model_paths = [
    'outputs/models/manuscript_restorer_best.pth',
    'checkpoints/best_model.pth',
    'outputs/checkpoints/best_model.pth',
]

restoration_model_path = None
for path in model_paths:
    if Path(path).exists():
        restoration_model_path = path
        print(f"✅ Found model: {path}")
        break

if restoration_model_path is None:
    print("⚠️  No restoration model found. Testing with simulated restoration...")
    # Simulate restoration (just copy the image)
    restored_img = original_img_rgb.copy()
    print("   Using original image as 'restored'")
else:
    # Load actual restoration model
    print(f"   Loading model from: {restoration_model_path}")
    try:
        from main import ManuscriptPipeline
        import torch

        # Create pipeline with restoration
        pipeline = ManuscriptPipeline(
            restoration_model_path=restoration_model_path,
            ocr_engine='tesseract',
            device='cpu'
        )
        print("✅ Pipeline initialized with restoration model")

        # Restore the image
        restored_img = pipeline._restore_image(original_img_rgb, use_enhanced=True)
        print(f"✅ Image restored: {restored_img.shape}, dtype={restored_img.dtype}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        restored_img = original_img_rgb.copy()

# Test 3: Check restored image properties
print("\n" + "=" * 70)
print("TEST 3: RESTORED IMAGE DIAGNOSTICS")
print("=" * 70)
print(f"Shape: {restored_img.shape}")
print(f"Dtype: {restored_img.dtype}")
print(f"Value range: [{restored_img.min()}, {restored_img.max()}]")
print(f"Mean: {restored_img.mean():.2f}")

# Check if it's in valid range
if restored_img.max() <= 1.0:
    print("⚠️  WARNING: Image values are in [0, 1] range (should be [0, 255])")
    print("   Converting to [0, 255]...")
    restored_img = (restored_img * 255).astype(np.uint8)
    print(f"   After conversion: {restored_img.shape}, dtype={restored_img.dtype}, range=[{restored_img.min()}, {restored_img.max()}]")

# Check if it's grayscale
if len(restored_img.shape) == 2:
    print("⚠️  WARNING: Image is grayscale (should be RGB)")
    print("   Converting to RGB...")
    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_GRAY2RGB)
elif restored_img.shape[2] == 1:
    print("⚠️  WARNING: Image has 1 channel (should be 3)")
    print("   Converting to RGB...")
    restored_img = cv2.cvtColor(restored_img.squeeze(), cv2.COLOR_GRAY2RGB)

print(f"✅ Final restored image: {restored_img.shape}, dtype={restored_img.dtype}, range=[{restored_img.min()}, {restored_img.max()}]")

# Test 4: OCR on restored image (direct)
print("\n" + "=" * 70)
print("TEST 4: OCR ON RESTORED IMAGE (DIRECT)")
print("=" * 70)
try:
    pil_restored = Image.fromarray(restored_img)
    print(f"   PIL Image mode: {pil_restored.mode}")
    print(f"   PIL Image size: {pil_restored.size}")

    text_restored = pytesseract.image_to_string(pil_restored, lang='san', config='--psm 6')
    print(f"✅ Extracted {len(text_restored)} characters")
    print(f"Preview: {text_restored[:150]}")

    if len(text_restored) < 50:
        print("⚠️  WARNING: Very little text extracted from restored image!")
        print("   This indicates the restoration might be making text unreadable")
except Exception as e:
    print(f"❌ OCR failed on restored image: {e}")
    import traceback
    traceback.print_exc()

# Test 5: OCR with preprocessing
print("\n" + "=" * 70)
print("TEST 5: OCR WITH PREPROCESSING")
print("=" * 70)
try:
    from ocr.preprocess import OCRPreprocessor
    preprocessor = OCRPreprocessor()

    preprocessed = preprocessor.preprocess(restored_img, apply_all=True)
    print(f"✅ Preprocessed: {preprocessed.shape}, dtype={preprocessed.dtype}")

    pil_preprocessed = Image.fromarray(preprocessed)
    text_preprocessed = pytesseract.image_to_string(pil_preprocessed, lang='san', config='--psm 6')
    print(f"✅ Extracted {len(text_preprocessed)} characters")
    print(f"Preview: {text_preprocessed[:150]}")
except Exception as e:
    print(f"❌ Preprocessing/OCR failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: OCR using EnhancedSanskritOCR
print("\n" + "=" * 70)
print("TEST 6: OCR USING EnhancedSanskritOCR CLASS")
print("=" * 70)
try:
    from ocr.enhanced_ocr import EnhancedSanskritOCR
    ocr = EnhancedSanskritOCR(engine='tesseract', device='cpu')

    result = ocr.extract_complete_paragraph(restored_img, preprocess=True, multi_pass=True)
    print(f"✅ Extracted {len(result['text'])} characters")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Method: {result['method']}")
    print(f"Preview: {result['text'][:150]}")
except Exception as e:
    print(f"❌ EnhancedSanskritOCR failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Save images for visual inspection
print("\n" + "=" * 70)
print("TEST 7: SAVING IMAGES FOR VISUAL INSPECTION")
print("=" * 70)
output_dir = Path('output/restoration_test')
output_dir.mkdir(exist_ok=True, parents=True)

cv2.imwrite(str(output_dir / 'original.jpg'), cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(output_dir / 'restored.jpg'), cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))

if 'preprocessed' in locals():
    if len(preprocessed.shape) == 2:
        cv2.imwrite(str(output_dir / 'preprocessed.jpg'), preprocessed)
    else:
        cv2.imwrite(str(output_dir / 'preprocessed.jpg'), cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))

print(f"✅ Images saved to: {output_dir}/")
print(f"   - original.jpg")
print(f"   - restored.jpg")
print(f"   - preprocessed.jpg")

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(f"""
Original image OCR: {len(text_original)} chars
Restored image OCR: {len(text_restored) if 'text_restored' in locals() else 'FAILED'} chars

ISSUE DIAGNOSIS:
""")

if 'text_restored' in locals():
    original_len = len(text_original.strip())
    restored_len = len(text_restored.strip())

    if restored_len < original_len * 0.3:
        print("""
❌ PROBLEM FOUND: Restored image has significantly less text than original!

POSSIBLE CAUSES:
1. Restoration model is blurring the text too much
2. Restoration output format is incorrect (wrong dtype or value range)
3. Restoration is removing text instead of enhancing it

SOLUTIONS:
1. Check restoration model output:
   - Should be uint8 with values [0, 255]
   - Should be RGB (3 channels)
   - Should preserve or enhance text, not blur it

2. Try these fixes in main.py:
   a) Add dtype/range conversion after restoration
   b) Reduce restoration strength/blur
   c) Skip restoration for high-quality images
   d) Use preprocess=False if preprocessing is too aggressive

3. Test without restoration:
   pipeline = ManuscriptPipeline(restoration_model_path=None, ...)
""")
    elif restored_len < original_len * 0.7:
        print("""
⚠️  MILD ISSUE: Restored image has somewhat less text than original

This might be acceptable depending on image quality. Check:
- Restoration model hyperparameters
- Preprocessing settings (might be too aggressive)
- Try multi_pass=False or different PSM modes
""")
    else:
        print("""
✅ RESTORATION IS WORKING: Text extraction from restored image is good!

If you're still having issues in Streamlit:
1. Check image format in Streamlit app
2. Ensure proper RGB conversion
3. Verify temp file is saved/loaded correctly
4. Check Streamlit image upload format
""")
else:
    print("""
❌ OCR FAILED ON RESTORED IMAGE

Check error messages above for specific issues.
""")

print("\nVisual inspection: Check the saved images in output/restoration_test/")

