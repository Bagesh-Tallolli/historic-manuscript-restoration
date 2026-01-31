
#!/usr/bin/env python3
"""
Debug script to test image restoration in isolation
"""
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.vit_restorer import create_vit_restorer
from utils.image_restoration_enhanced import create_enhanced_restorer

def test_restoration(image_path):
    """Test restoration on a single image"""

    print("=" * 70)
    print("IMAGE RESTORATION DEBUG TEST")
    print("=" * 70)

    # Configuration
    CHECKPOINT_PATH = "checkpoints/kaggle/final_converted.pth"
    MODEL_SIZE = "base"
    IMG_SIZE = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n1. CONFIGURATION")
    print(f"   Device: {device}")
    print(f"   Model size: {MODEL_SIZE}")
    print(f"   Patch size: {IMG_SIZE}")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")

    # Load image
    print(f"\n2. LOADING IMAGE")
    print(f"   Path: {image_path}")

    if not os.path.exists(image_path):
        print(f"   ❌ Image not found!")
        return

    # Load with PIL
    image_pil = Image.open(image_path)
    print(f"   ✓ PIL Image loaded")
    print(f"   Format: {image_pil.format}")
    print(f"   Mode: {image_pil.mode}")
    print(f"   Size: {image_pil.size}")

    # Convert to numpy
    img_array = np.array(image_pil)
    print(f"   ✓ Converted to numpy")
    print(f"   Shape: {img_array.shape}")
    print(f"   Dtype: {img_array.dtype}")
    print(f"   Min/Max: {img_array.min()}/{img_array.max()}")

    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        print(f"   → Converting grayscale to RGB")
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        print(f"   → Converting RGBA to RGB")
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    print(f"   Final shape: {img_array.shape}")

    # Load model
    print(f"\n3. LOADING MODEL")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"   ✓ Wrapped checkpoint")
    else:
        state_dict = checkpoint
        print(f"   ✓ Direct state_dict")

    # Check format
    has_patch_recon = any(k.startswith('patch_recon.') for k in state_dict.keys())
    has_head = any(k.startswith('head.') for k in state_dict.keys())

    print(f"   Has patch_recon: {has_patch_recon}")
    print(f"   Has head: {has_head}")

    # Create model
    model = create_vit_restorer(
        model_size=MODEL_SIZE,
        img_size=IMG_SIZE,
        use_simple_head=not has_patch_recon
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"   ✓ Model loaded and ready")

    # Create enhanced restorer
    print(f"\n4. CREATING ENHANCED RESTORER")
    enhanced_restorer = create_enhanced_restorer(
        model, device=device, patch_size=IMG_SIZE, overlap=32
    )
    print(f"   ✓ Enhanced restorer created")

    # Test restoration
    print(f"\n5. TESTING RESTORATION")
    h, w = img_array.shape[:2]
    use_patches = (h > 512 or w > 512)

    print(f"   Image size: {w}x{h}")
    print(f"   Use patches: {use_patches}")

    print(f"\n   → Running restoration...")
    try:
        restored_array = enhanced_restorer.restore_image(
            img_array,
            use_patches=use_patches,
            apply_postprocess=True
        )

        print(f"   ✓ Restoration complete!")
        print(f"   Input shape: {img_array.shape}")
        print(f"   Output shape: {restored_array.shape}")
        print(f"   Output dtype: {restored_array.dtype}")
        print(f"   Output min/max: {restored_array.min()}/{restored_array.max()}")

        # Check if image is blank
        if restored_array.max() == 0:
            print(f"   ❌ WARNING: Output image is completely black!")
        elif restored_array.min() == restored_array.max():
            print(f"   ❌ WARNING: Output image is uniform (all same value)!")
        else:
            print(f"   ✓ Output image has variation")

        # Save results
        print(f"\n6. SAVING RESULTS")

        # Save original
        cv2.imwrite("debug_original.png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        print(f"   ✓ Saved: debug_original.png")

        # Save restored
        cv2.imwrite("debug_restored.png", cv2.cvtColor(restored_array, cv2.COLOR_RGB2BGR))
        print(f"   ✓ Saved: debug_restored.png")

        # Save as PIL
        restored_pil = Image.fromarray(restored_array)
        restored_pil.save("debug_restored_pil.png")
        print(f"   ✓ Saved: debug_restored_pil.png")

        print(f"\n" + "=" * 70)
        print(f"✅ TEST COMPLETE - Check debug_*.png files")
        print(f"=" * 70)

    except Exception as e:
        print(f"   ❌ Error during restoration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python debug_restoration.py <image_path>")
        print("\nExample:")
        print("  python debug_restoration.py data/test/degraded/sample.jpg")
        sys.exit(1)

    test_restoration(sys.argv[1])

