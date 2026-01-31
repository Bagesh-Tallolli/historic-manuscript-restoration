#!/usr/bin/env python3
"""
Test restoration functionality and diagnose issues
"""
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/bagesh/EL-project')

print("="*60)
print("RESTORATION DIAGNOSTIC TEST")
print("="*60)

# Test 1: Check model file
print("\n[1] Checking model file...")
model_path = 'checkpoints/kaggle/final.pth'
if not Path(model_path).exists():
    print(f"❌ Model not found: {model_path}")
    sys.exit(1)
print(f"✅ Model file exists: {Path(model_path).stat().st_size / (1024**2):.1f} MB")

# Test 2: Load checkpoint
print("\n[2] Loading checkpoint...")
try:
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Found model_state_dict in checkpoint")
    else:
        state_dict = checkpoint
        print("✅ Using checkpoint directly as state_dict")

    keys = list(state_dict.keys())
    print(f"   Total parameters: {len(keys)}")
    print(f"   Sample keys:")
    for k in keys[:5]:
        print(f"     - {k}")

    # Check architecture indicators
    has_head = any('head.' in k for k in keys)
    has_patch_recon = any('patch_recon' in k for k in keys)
    has_skip = any('skip' in k for k in keys)

    print(f"\n   Architecture detection:")
    print(f"     has_head: {has_head}")
    print(f"     has_patch_recon: {has_patch_recon}")
    print(f"     has_skip: {has_skip}")

except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create model
print("\n[3] Creating model architecture...")
try:
    from models.vit_restorer import create_vit_restorer

    if has_head and not has_patch_recon:
        print("   Using: simple head, no skip connections")
        model = create_vit_restorer('base', img_size=256,
                                    use_skip_connections=False,
                                    use_simple_head=True)
    elif has_patch_recon and has_skip:
        print("   Using: patch_recon with skip connections")
        model = create_vit_restorer('base', img_size=256,
                                    use_skip_connections=True,
                                    use_simple_head=False)
    else:
        print("   Using: patch_recon without skip")
        model = create_vit_restorer('base', img_size=256,
                                    use_skip_connections=False,
                                    use_simple_head=False)

    print("✅ Model architecture created")

except Exception as e:
    print(f"❌ Error creating model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load weights
print("\n[4] Loading model weights...")
try:
    model.load_state_dict(state_dict)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Weights loaded successfully")
    print(f"   Parameters: {param_count:.1f}M")

except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("\n   Trying to match keys...")
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        print(f"   Missing keys ({len(missing)}):")
        for k in list(missing)[:5]:
            print(f"     - {k}")
    if unexpected:
        print(f"   Unexpected keys ({len(unexpected)}):")
        for k in list(unexpected)[:5]:
            print(f"     - {k}")

    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test image restoration
print("\n[5] Testing image restoration...")
test_img_path = 'data/raw/test/test_0001.jpg'
if not Path(test_img_path).exists():
    print(f"⚠️  Test image not found: {test_img_path}")
    print("   Skipping restoration test")
else:
    try:
        # Load image
        img = cv2.imread(test_img_path)
        if img is None:
            print(f"❌ Failed to load image: {test_img_path}")
        else:
            print(f"✅ Loaded test image: {img.shape}")

            # Convert and resize
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_shape = img_rgb.shape[:2]
            img_resized = cv2.resize(img_rgb, (256, 256))

            # Convert to tensor
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            print(f"   Input tensor shape: {img_tensor.shape}")

            # Restore
            with torch.no_grad():
                restored_tensor = model(img_tensor)

            print(f"   Output tensor shape: {restored_tensor.shape}")

            # Convert back
            restored = restored_tensor.squeeze(0).permute(1, 2, 0).numpy()
            restored = np.clip(restored * 255, 0, 255).astype(np.uint8)

            # Resize to original
            restored = cv2.resize(restored, (original_shape[1], original_shape[0]))

            # Save test output
            output_path = 'outputs/test_restoration.png'
            Path('outputs').mkdir(exist_ok=True)
            restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, restored_bgr)

            print(f"✅ Restoration successful!")
            print(f"   Output saved: {output_path}")
            print(f"   Output shape: {restored.shape}")

            # Check if restoration actually did something
            diff = np.abs(img_rgb.astype(float) - restored.astype(float)).mean()
            print(f"   Mean difference from input: {diff:.2f}")

            if diff < 1.0:
                print("   ⚠️  Warning: Very little change detected (output ≈ input)")
                print("      This might indicate restoration is not working properly")
            else:
                print("   ✅ Restoration produced visible changes")

    except Exception as e:
        print(f"❌ Error during restoration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)

