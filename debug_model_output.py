#!/usr/bin/env python3
"""
Deep debug - check model raw output
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.vit_restorer import create_vit_restorer

# Load model
device = 'cpu'
checkpoint = torch.load('checkpoints/kaggle/final_converted.pth', map_location=device, weights_only=False)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

has_patch_recon = any(k.startswith('patch_recon.') for k in state_dict.keys())

model = create_vit_restorer(
    model_size='base',
    img_size=256,
    use_simple_head=not has_patch_recon
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Model loaded successfully")
print(f"Uses patch_recon: {has_patch_recon}")

# Load test image
img = Image.open('data/datasets/samples/test_sample.png').convert('RGB')
img_array = np.array(img)

print(f"\nOriginal image:")
print(f"  Shape: {img_array.shape}")
print(f"  Min/Max: {img_array.min()}/{img_array.max()}")
print(f"  Mean: {img_array.mean():.2f}")

# Take a 256x256 patch
patch = img_array[:256, :256, :]
print(f"\nPatch:")
print(f"  Shape: {patch.shape}")
print(f"  Min/Max: {patch.min()}/{patch.max()}")

# Convert to tensor (normalize to [0, 1])
patch_tensor = torch.from_numpy(patch).float() / 255.0
patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0)

print(f"\nInput tensor:")
print(f"  Shape: {patch_tensor.shape}")
print(f"  Min/Max: {patch_tensor.min():.4f}/{patch_tensor.max():.4f}")
print(f"  Mean: {patch_tensor.mean():.4f}")

# Run model
with torch.no_grad():
    output = model(patch_tensor)

print(f"\nOutput tensor:")
print(f"  Shape: {output.shape}")
print(f"  Min/Max: {output.min():.4f}/{output.max():.4f}")
print(f"  Mean: {output.mean():.4f}")
print(f"  Std: {output.std():.4f}")

# Check if output is in [0, 1] range
if output.min() >= 0 and output.max() <= 1:
    print("  ✓ Output in [0, 1] range")
elif output.min() >= -1 and output.max() <= 1:
    print("  ⚠️  Output in [-1, 1] range - needs adjustment!")
else:
    print(f"  ⚠️  Output in unexpected range!")

# Convert back
output_np = output.squeeze(0).permute(1, 2, 0).numpy()
print(f"\nOutput numpy (before scaling):")
print(f"  Shape: {output_np.shape}")
print(f"  Min/Max: {output_np.min():.4f}/{output_np.max():.4f}")

# Scale to [0, 255]
output_scaled = (output_np * 255).clip(0, 255).astype(np.uint8)
print(f"\nOutput scaled to uint8:")
print(f"  Min/Max: {output_scaled.min()}/{output_scaled.max()}")
print(f"  Mean: {output_scaled.mean():.2f}")

# Check if it's too dark
if output_scaled.max() < 50:
    print("\n❌ OUTPUT IS TOO DARK!")
    print("   Possible issues:")
    print("   1. Model outputs in [-1, 1] range instead of [0, 1]")
    print("   2. Model was trained with different normalization")
    print("   3. Model checkpoint is corrupted")

    # Try alternative scaling
    print("\n   Trying alternative scaling methods:")

    # Method 1: Assume [-1, 1] range
    if output_np.min() < 0:
        output_alt1 = ((output_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        print(f"   [-1,1]→[0,255]: Min/Max = {output_alt1.min()}/{output_alt1.max()}")

    # Method 2: Normalize to full range
    if output_np.max() != output_np.min():
        output_alt2 = ((output_np - output_np.min()) / (output_np.max() - output_np.min()) * 255).astype(np.uint8)
        print(f"   Normalize: Min/Max = {output_alt2.min()}/{output_alt2.max()}")
else:
    print("\n✓ Output looks OK")

# Save comparison
Image.fromarray(patch).save('debug_patch_input.png')
Image.fromarray(output_scaled).save('debug_patch_output.png')
print(f"\nSaved debug images:")
print(f"  - debug_patch_input.png")
print(f"  - debug_patch_output.png")

