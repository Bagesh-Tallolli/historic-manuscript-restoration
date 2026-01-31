#!/usr/bin/env python3
"""
Test script to verify the restoration model loads correctly
"""
import os
import sys
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.vit_restorer import create_vit_restorer

def test_model_loading():
    print("=" * 60)
    print("Testing Restoration Model Loading")
    print("=" * 60)

    # Configuration
    CHECKPOINT_PATHS = [
        "checkpoints/kaggle/final_converted.pth",
        "checkpoints/kaggle/final.pth",
        "models/trained_models/final.pth",
    ]

    MODEL_SIZE = "base"
    IMG_SIZE = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n✓ Device: {device}")

    # Find checkpoint
    checkpoint_path = None
    for path in CHECKPOINT_PATHS:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"✓ Found checkpoint: {checkpoint_path}")
            break

    if checkpoint_path is None:
        print("✗ No checkpoint found!")
        return False

    # Load checkpoint
    print(f"\n→ Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✓ Checkpoint format: Wrapped (has 'model_state_dict')")
    else:
        state_dict = checkpoint
        print("✓ Checkpoint format: Direct state_dict")

    # Determine format
    has_patch_recon = any(k.startswith('patch_recon.') for k in state_dict.keys())
    has_head = any(k.startswith('head.') for k in state_dict.keys())

    print(f"\n→ Analyzing checkpoint structure:")
    print(f"   - Total keys: {len(state_dict)}")
    print(f"   - Has 'patch_recon.*': {has_patch_recon}")
    print(f"   - Has 'head.*': {has_head}")

    # Create model with appropriate format
    print(f"\n→ Creating model...")
    if has_patch_recon:
        print("   Using patch_recon decoder (newer format)")
        model = create_vit_restorer(
            model_size=MODEL_SIZE,
            img_size=IMG_SIZE,
            use_simple_head=False
        )
    else:
        print("   Using simple head decoder (older format)")
        model = create_vit_restorer(
            model_size=MODEL_SIZE,
            img_size=IMG_SIZE,
            use_simple_head=True
        )

    # Load state dict
    print(f"\n→ Loading weights...")
    try:
        model.load_state_dict(state_dict)
        print("✓ Weights loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        return False

    # Move to device and set eval mode
    model.to(device)
    model.eval()
    print(f"✓ Model moved to {device} and set to eval mode")

    # Test forward pass
    print(f"\n→ Testing forward pass...")
    test_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    try:
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Forward pass successful!")
        print(f"   Input shape:  {test_input.shape}")
        print(f"   Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return False

    # Success
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! Model is ready to use.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)

