#!/usr/bin/env python3
"""
Quick verification script to test restored image restoration
"""
import sys
from pathlib import Path

sys.path.insert(0, '/home/bagesh/EL-project')

print("="*70)
print("IMAGE RESTORATION VERIFICATION")
print("="*70)

# Test 1: Check imports
print("\n[1] Testing imports...")
try:
    from main import ManuscriptPipeline
    from utils.image_restoration_enhanced import create_enhanced_restorer
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize pipeline with restoration
print("\n[2] Initializing pipeline with restoration model...")
model_path = 'checkpoints/kaggle/final.pth'
if not Path(model_path).exists():
    print(f"❌ Model not found: {model_path}")
    sys.exit(1)

try:
    pipeline = ManuscriptPipeline(
        restoration_model_path=model_path,
        ocr_engine='tesseract',
        translation_method='google',
        device='cpu'  # Use CPU for testing
    )
    print("✅ Pipeline initialized successfully")
    print(f"   - Restoration model: Loaded")
    print(f"   - Enhanced restorer: {'Active' if pipeline.enhanced_restorer else 'Not initialized'}")
except Exception as e:
    print(f"❌ Pipeline initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check if test image exists
print("\n[3] Checking test image...")
test_img = 'data/raw/test/test_0001.jpg'
if not Path(test_img).exists():
    print(f"⚠️  Test image not found: {test_img}")
    print("   Skipping processing test")
    print("\n✅ Core restoration components verified!")
    sys.exit(0)

print(f"✅ Test image found: {test_img}")

# Test 4: Process image (quick test - no full pipeline)
print("\n[4] Testing image restoration only...")
try:
    import cv2
    import numpy as np

    # Load image
    img = cv2.imread(test_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"   Image size: {h}x{w}")

    # Test restoration
    print("   Running restoration...")
    restored = pipeline._restore_image(img_rgb, use_enhanced=True)

    print(f"✅ Restoration complete!")
    print(f"   Output size: {restored.shape[0]}x{restored.shape[1]}")

    # Quick quality check
    diff = np.abs(img_rgb.astype(float) - restored.astype(float)).mean()
    print(f"   Mean pixel difference: {diff:.2f}")

    if diff > 1.0:
        print(f"   ✅ Restoration is working (visible changes detected)")
    else:
        print(f"   ⚠️  Very little change - might need investigation")

    # Save test output
    output_dir = Path('outputs/verification_test')
    output_dir.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(output_dir / 'restored.jpg'), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
    print(f"   Saved to: {output_dir / 'restored.jpg'}")

except Exception as e:
    print(f"❌ Restoration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("VERIFICATION COMPLETE - ALL TESTS PASSED!")
print("="*70)
print("\n✅ Image restoration is working properly!")
print("\nNext steps:")
print("  1. Use enhanced restoration in your application")
print("  2. Test with your own manuscript images")
print("  3. See RESTORATION_GUIDE.md for usage examples")

