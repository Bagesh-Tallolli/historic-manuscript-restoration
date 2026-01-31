#!/usr/bin/env python3
"""
Convert Kaggle checkpoint to match current model architecture
This script updates old checkpoint formats to work with the current codebase
"""

import torch
import sys
from pathlib import Path

def convert_checkpoint(input_path, output_path=None):
    """
    Convert checkpoint from old format (with head) to new format (with patch_recon)

    Args:
        input_path: Path to input checkpoint
        output_path: Path to save converted checkpoint (None = overwrite with _converted suffix)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return False

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_converted.pth"
    else:
        output_path = Path(output_path)

    print(f"Converting: {input_path.name}")
    print(f"Output: {output_path.name}")

    # Load checkpoint
    try:
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        print(f"✓ Loaded checkpoint")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False

    # Check if it's wrapped or unwrapped
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        is_wrapped = True
    else:
        state_dict = checkpoint
        is_wrapped = False

    # Check if conversion is needed
    needs_conversion = 'head.weight' in state_dict and 'patch_recon.proj.weight' not in state_dict

    if not needs_conversion:
        print("✓ Checkpoint already in correct format")
        return True

    print("Converting checkpoint format...")

    # Create new state dict
    new_state_dict = {}

    for key, value in state_dict.items():
        if key == 'head.weight':
            # Rename head to patch_recon.proj
            new_state_dict['patch_recon.proj.weight'] = value
            print(f"  Converted: head.weight -> patch_recon.proj.weight")
        elif key == 'head.bias':
            new_state_dict['patch_recon.proj.bias'] = value
            print(f"  Converted: head.bias -> patch_recon.proj.bias")
        else:
            new_state_dict[key] = value

    # Add skip_fusion if it doesn't exist (initialize with zeros)
    if 'skip_fusion.weight' not in new_state_dict:
        # Skip fusion is Conv2d(6, 3, 1) for 3-channel images
        new_state_dict['skip_fusion.weight'] = torch.randn(3, 6, 1, 1) * 0.02
        new_state_dict['skip_fusion.bias'] = torch.zeros(3)
        print(f"  Added: skip_fusion layers (initialized)")

    # Wrap if original was wrapped
    if is_wrapped:
        new_checkpoint = checkpoint.copy()
        new_checkpoint['model_state_dict'] = new_state_dict
    else:
        new_checkpoint = new_state_dict

    # Save converted checkpoint
    try:
        torch.save(new_checkpoint, output_path)
        print(f"✓ Saved converted checkpoint to: {output_path}")

        # Print file sizes
        input_size_mb = input_path.stat().st_size / (1024 * 1024)
        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Input size: {input_size_mb:.2f} MB")
        print(f"  Output size: {output_size_mb:.2f} MB")

        return True
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_kaggle_checkpoint.py <checkpoint_path> [output_path]")
        print("\nExample:")
        print("  python convert_kaggle_checkpoint.py checkpoints/kaggle/final.pth")
        print("  python convert_kaggle_checkpoint.py checkpoints/kaggle/desti.pth checkpoints/kaggle/desti_v2.pth")
        print("\nThis will convert old checkpoint formats to work with the current model architecture.")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 70)
    print("KAGGLE CHECKPOINT CONVERTER")
    print("=" * 70)
    print()

    success = convert_checkpoint(input_path, output_path)

    print()
    if success:
        print("=" * 70)
        print("✅ Conversion successful!")
        print("=" * 70)
        if output_path is None:
            output_path = Path(input_path).parent / f"{Path(input_path).stem}_converted.pth"
        print(f"\nYou can now use the converted model:")
        print(f"  python inference.py --checkpoint {output_path} --input data/datasets/samples/")
    else:
        print("=" * 70)
        print("❌ Conversion failed")
        print("=" * 70)


if __name__ == '__main__':
    main()

