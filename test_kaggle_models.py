#!/usr/bin/env python3
"""
Test script to validate Kaggle-trained model files
Usage: python test_kaggle_models.py
"""

import torch
import sys
from pathlib import Path
from models.vit_restorer import create_vit_restorer

def test_model_checkpoint(checkpoint_path, model_size='base', img_size=256):
    """
    Test if a model checkpoint can be loaded successfully

    Args:
        checkpoint_path: Path to .pth checkpoint file
        model_size: Model architecture size
        img_size: Image input size

    Returns:
        bool: True if successful, False otherwise
    """
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'='*70}")
    print(f"Testing: {checkpoint_path.name}")
    print(f"{'='*70}")

    # Check if file exists
    if not checkpoint_path.exists():
        print(f"❌ File not found: {checkpoint_path}")
        return False

    # Check file size
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"✓ File exists")
    print(f"✓ File size: {file_size_mb:.2f} MB")

    # Load checkpoint
    try:
        print(f"\nLoading checkpoint...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

    # Check checkpoint contents
    print(f"\nCheckpoint contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  • {key}: <dict with {len(checkpoint[key])} items>")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  • {key}: <tensor {checkpoint[key].shape}>")
        else:
            print(f"  • {key}: {checkpoint[key]}")

    # Check checkpoint format
    if 'model_state_dict' in checkpoint:
        print(f"\n✓ Checkpoint format: Wrapped (contains metadata)")
    else:
        print(f"\n✓ Checkpoint format: Direct state dict (weights only)")

    # Try to load into model
    try:
        print(f"\nCreating model architecture...")
        model = create_vit_restorer(model_size, img_size=img_size)
        print(f"✓ Model created")

        print(f"Loading weights into model...")
        # Handle both wrapped and unwrapped checkpoints
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Checkpoint is the state dict itself
            model.load_state_dict(checkpoint)
        print(f"✓ Weights loaded successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel info:")
        print(f"  • Total parameters: {total_params:,}")
        print(f"  • Trainable parameters: {trainable_params:,}")

        # Test forward pass
        print(f"\nTesting forward pass...")
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
            model = model.to(device)
            output = model(dummy_input)
            print(f"✓ Forward pass successful")
            print(f"  • Input shape: {dummy_input.shape}")
            print(f"  • Output shape: {output.shape}")

        print(f"\n{'='*70}")
        print(f"✅ SUCCESS: Model is ready to use!")
        print(f"{'='*70}")
        return True

    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print(f"{'='*70}")
        return False


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("KAGGLE MODEL VALIDATION TEST")
    print("="*70)

    # Define paths to test
    test_models = [
        ('checkpoints/kaggle/final.pth', 'base', 256),
        ('checkpoints/kaggle/desti.pth', 'base', 256),
        ('checkpoints/latest_model.pth', 'base', 256),
    ]

    results = {}

    for model_path, model_size, img_size in test_models:
        full_path = Path(model_path)
        if full_path.exists():
            success = test_model_checkpoint(full_path, model_size, img_size)
            results[model_path] = success
        else:
            print(f"\n{'='*70}")
            print(f"Skipping: {model_path} (not found)")
            print(f"{'='*70}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if not results:
        print("\n❌ No model files found to test!")
        print("\nPlease place your Kaggle-trained models in:")
        print("  • checkpoints/kaggle/final.pth")
        print("  • checkpoints/kaggle/desti.pth")
        print("\nRun this command to set up:")
        print("  bash setup_kaggle_models.sh ~/Downloads")
    else:
        for model_path, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {model_path}")

        all_passed = all(results.values())
        if all_passed:
            print("\n🎉 All models validated successfully!")
            print("\nYou can now use them for inference:")
            print("  python inference.py --input data/datasets/samples/ \\")
            print("      --output output/test --checkpoint checkpoints/kaggle/final.pth")
        else:
            print("\n⚠️  Some models failed validation")
            print("Please check the error messages above")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

