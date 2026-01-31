#!/usr/bin/env python3
"""
Test your trained model after downloading from Kaggle
"""

import torch
import sys
from pathlib import Path

def test_model_checkpoint(checkpoint_path):
    """Test if the model checkpoint is valid"""

    print("=" * 70)
    print("🔍 TESTING TRAINED MODEL CHECKPOINT")
    print("=" * 70)
    print()

    # Check if file exists
    if not Path(checkpoint_path).exists():
        print(f"❌ ERROR: Model file not found at: {checkpoint_path}")
        print()
        print("📥 Please download from Kaggle:")
        print("   1. Go to your Kaggle notebook")
        print("   2. Right sidebar → Output")
        print("   3. Download: /kaggle/working/checkpoints/best_psnr.pth")
        print(f"   4. Copy to: {checkpoint_path}")
        return False

    # Check file size
    file_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)  # MB
    print(f"✓ Model file found")
    print(f"✓ File size: {file_size:.1f} MB")

    if file_size < 100:
        print(f"⚠️  Warning: File seems too small (expected ~330 MB)")
        print(f"   Download may have been incomplete")
        return False

    print()

    # Load checkpoint
    try:
        print("📦 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading checkpoint: {e}")
        return False

    print()

    # Check checkpoint contents
    print("📊 Checkpoint Information:")
    print("-" * 70)

    if 'epoch' in checkpoint:
        print(f"  Epochs trained: {checkpoint['epoch'] + 1}")

    if 'best_val_psnr' in checkpoint:
        print(f"  Best validation PSNR: {checkpoint['best_val_psnr']:.2f} dB")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        num_params = sum(p.numel() for p in state_dict.values())
        print(f"  Model parameters: {num_params:,}")
        print(f"  Model size: ~{num_params * 4 / (1024**2):.1f} MB (FP32)")

    print()

    # Load into model
    try:
        print("🤖 Creating model architecture...")
        from models.vit_restorer import ViTRestorer

        model = ViTRestorer(
            img_size=256,
            patch_size=16,
            in_channels=3,
            out_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            use_skip_connections=True
        )

        print("✓ Model architecture created")
        print()

        print("📥 Loading weights into model...")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print("✓ Weights loaded successfully")
        print()

        # Test forward pass
        print("🧪 Testing forward pass...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            output = model(dummy_input)

        print(f"✓ Forward pass successful")
        print(f"  Input shape: {tuple(dummy_input.shape)}")
        print(f"  Output shape: {tuple(output.shape)}")
        print()

    except Exception as e:
        print(f"❌ ERROR testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Success!
    print("=" * 70)
    print("✅ MODEL IS READY TO USE!")
    print("=" * 70)
    print()
    print("🚀 Next steps:")
    print("   1. Test on a real image:")
    print("      python inference.py --model checkpoints/best_psnr.pth \\")
    print("                          --input test_image.jpg \\")
    print("                          --output restored.jpg")
    print()
    print("   2. Start the Streamlit app:")
    print("      streamlit run app.py")
    print()

    return True


def show_usage():
    """Show usage instructions"""
    print("Usage:")
    print("  python test_trained_model.py [checkpoint_path]")
    print()
    print("Example:")
    print("  python test_trained_model.py checkpoints/best_psnr.pth")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "checkpoints/best_psnr.pth"

    success = test_model_checkpoint(checkpoint_path)

    if not success:
        print()
        print("=" * 70)
        print("❌ MODEL TEST FAILED")
        print("=" * 70)
        print()
        print("Please check the error messages above and:")
        print("  1. Verify the model file is downloaded from Kaggle")
        print("  2. Ensure it's placed in the correct location")
        print("  3. Check that the download completed successfully")
        print()
        sys.exit(1)

    sys.exit(0)

