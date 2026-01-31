#!/usr/bin/env python3
"""
Test enhanced restoration implementation
"""
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, '/home/bagesh/EL-project')

print("="*70)
print("ENHANCED RESTORATION TEST")
print("="*70)

# Check if model exists
model_path = 'checkpoints/kaggle/final.pth'
if not Path(model_path).exists():
    print(f"❌ Model not found: {model_path}")
    sys.exit(1)

# Load model
print("\n[1] Loading model...")
from models.vit_restorer import create_vit_restorer
from utils.image_restoration_enhanced import create_enhanced_restorer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Detect architecture
has_head = any('head.' in k for k in state_dict.keys())
has_patch_recon = any('patch_recon' in k for k in state_dict.keys())
has_skip = any('skip' in k for k in state_dict.keys())

if has_head and not has_patch_recon:
    model = create_vit_restorer('base', img_size=256,
                                use_skip_connections=False,
                                use_simple_head=True)
elif has_patch_recon and has_skip:
    model = create_vit_restorer('base', img_size=256,
                                use_skip_connections=True,
                                use_simple_head=False)
else:
    model = create_vit_restorer('base', img_size=256,
                                use_skip_connections=False,
                                use_simple_head=False)

model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("✅ Model loaded successfully")

# Create enhanced restorer
print("\n[2] Creating enhanced restorer...")
enhanced_restorer = create_enhanced_restorer(model, device, patch_size=256, overlap=32)
print("✅ Enhanced restorer created")

# Load test image
test_img_path = 'data/raw/test/test_0001.jpg'
if not Path(test_img_path).exists():
    print(f"\n⚠️  Test image not found: {test_img_path}")
    print("   Please provide a test image")
    sys.exit(1)

print(f"\n[3] Loading test image: {test_img_path}")
img = cv2.imread(test_img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]
print(f"   Image size: {h}x{w}")

# Create output directory
output_dir = Path('outputs/enhanced_restoration_test')
output_dir.mkdir(exist_ok=True, parents=True)

# Test 1: Simple restoration (original method)
print("\n[4] Testing simple restoration (resize method)...")
start_time = time.time()
restored_simple = enhanced_restorer.restore_image(img_rgb, use_patches=False, apply_postprocess=False)
simple_time = time.time() - start_time
print(f"✅ Simple restoration complete in {simple_time:.2f}s")

# Test 2: Enhanced restoration with patches
print("\n[5] Testing enhanced restoration (patch-based)...")
start_time = time.time()
restored_enhanced = enhanced_restorer.restore_image(img_rgb, use_patches=True, apply_postprocess=True)
enhanced_time = time.time() - start_time
print(f"✅ Enhanced restoration complete in {enhanced_time:.2f}s")

# Save results
print("\n[6] Saving results...")

# Original
cv2.imwrite(str(output_dir / 'original.png'), img)
print(f"   ✓ Saved: {output_dir / 'original.png'}")

# Simple restoration
cv2.imwrite(str(output_dir / 'restored_simple.png'), cv2.cvtColor(restored_simple, cv2.COLOR_RGB2BGR))
print(f"   ✓ Saved: {output_dir / 'restored_simple.png'}")

# Enhanced restoration
cv2.imwrite(str(output_dir / 'restored_enhanced.png'), cv2.cvtColor(restored_enhanced, cv2.COLOR_RGB2BGR))
print(f"   ✓ Saved: {output_dir / 'restored_enhanced.png'}")

# Create comparison image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_rgb)
axes[0].set_title('Original', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(restored_simple)
axes[1].set_title(f'Simple Restoration\n({simple_time:.2f}s)', fontsize=14, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(restored_enhanced)
axes[2].set_title(f'Enhanced Restoration\n({enhanced_time:.2f}s)', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
comparison_path = output_dir / 'comparison.png'
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {comparison_path}")

# Calculate statistics
print("\n[7] Statistics:")
print(f"   Simple restoration time: {simple_time:.2f}s")
print(f"   Enhanced restoration time: {enhanced_time:.2f}s")
print(f"   Time difference: {enhanced_time - simple_time:.2f}s")

diff_simple = np.abs(img_rgb.astype(float) - restored_simple.astype(float)).mean()
diff_enhanced = np.abs(img_rgb.astype(float) - restored_enhanced.astype(float)).mean()

print(f"\n   Mean difference (simple): {diff_simple:.2f}")
print(f"   Mean difference (enhanced): {diff_enhanced:.2f}")

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)
print(f"\nResults saved to: {output_dir}/")
print("\nComparison:")
print(f"  - Simple method: Faster ({simple_time:.2f}s) but may lose details")
print(f"  - Enhanced method: Slower ({enhanced_time:.2f}s) but better quality")
print("\nRecommendation:")
if h > 512 or w > 512:
    print(f"  ✅ Use ENHANCED method for large images ({h}x{w})")
else:
    print(f"  ✅ Use SIMPLE method for small images ({h}x{w})")

