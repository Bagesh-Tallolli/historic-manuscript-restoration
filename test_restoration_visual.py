#!/usr/bin/env python3
"""
Visual comparison test - Check if restoration is visually improving images
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '/home/bagesh/EL-project')

from pipeline_clean import SanskritManuscriptPipeline

print("="*70)
print("VISUAL RESTORATION TEST")
print("="*70)

# Initialize pipeline
print("\nInitializing pipeline...")
pipeline = SanskritManuscriptPipeline(
    restoration_model_path='checkpoints/kaggle/final.pth',
    google_api_key=os.getenv('GOOGLE_API_KEY', ''),
    gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
    device='cpu'
)

# Test restoration on a sample image
test_img = 'data/raw/test/test_0001.jpg'
output_dir = Path('outputs/restoration_test')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nRestoring test image: {test_img}")
restored_path = str(output_dir / "restored.png")

# Restore
try:
    restored_img = pipeline.restore_image(test_img, restored_path)
    print(f"✅ Restoration complete: {restored_path}")

    # Load both images for comparison
    original = cv2.imread(test_img)
    restored = cv2.imread(restored_path)

    if original is None or restored is None:
        print("❌ Failed to load images for comparison")
        sys.exit(1)

    # Calculate metrics
    print("\nComparison Metrics:")
    print(f"  Original shape: {original.shape}")
    print(f"  Restored shape: {restored.shape}")

    # Mean pixel difference
    if original.shape == restored.shape:
        diff = np.abs(original.astype(float) - restored.astype(float))
        mean_diff = diff.mean()
        max_diff = diff.max()

        print(f"  Mean pixel difference: {mean_diff:.2f}")
        print(f"  Max pixel difference: {max_diff:.2f}")

        # Check if images are too similar
        if mean_diff < 5.0:
            print("\n⚠️  WARNING: Images are very similar!")
            print("     Restoration may not be working effectively")
        else:
            print("\n✅ Good: Noticeable differences detected")

        # Calculate PSNR
        mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
            print(f"  PSNR: {psnr:.2f} dB")

        # Calculate percentage of changed pixels
        changed_pixels = np.sum(diff > 5) / diff.size * 100
        print(f"  Changed pixels (>5 diff): {changed_pixels:.2f}%")

    else:
        print("  ⚠️ Shapes don't match, can't compare directly")

    # Create side-by-side comparison
    print("\nCreating side-by-side comparison...")
    h, w = min(original.shape[0], 800), min(original.shape[1], 600)
    orig_resized = cv2.resize(original, (w, h))
    rest_resized = cv2.resize(restored, (w, h))

    comparison = np.hstack([orig_resized, rest_resized])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'ORIGINAL', (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, 'RESTORED', (w + 10, 30), font, 1, (0, 255, 0), 2)

    comparison_path = str(output_dir / "comparison.png")
    cv2.imwrite(comparison_path, comparison)
    print(f"✅ Comparison saved: {comparison_path}")

    # Create difference heatmap
    if original.shape == restored.shape:
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        diff_heatmap = cv2.applyColorMap(diff_gray * 5, cv2.COLORMAP_JET)
        heatmap_path = str(output_dir / "difference_heatmap.png")
        cv2.imwrite(heatmap_path, diff_heatmap)
        print(f"✅ Difference heatmap: {heatmap_path}")

    print("\n" + "="*70)
    print("TEST COMPLETE - Check outputs in:", output_dir)
    print("="*70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

