"""
Inference script for processing manuscripts using trained models
"""

import torch
import argparse
from pathlib import Path
import cv2
import numpy as np

from models.vit_restorer import create_vit_restorer
from utils.metrics import ImageMetrics


def restore_image(model, image_path, device='cuda', img_size=256):
    """
    Restore a single image using trained model

    Args:
        model: Trained ViT restoration model
        image_path: Path to input image
        device: Device to use
        img_size: Input size for model

    Returns:
        Restored image (numpy array)
    """
    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]

    # Resize to model input size
    img_resized = cv2.resize(img, (img_size, img_size))

    # Convert to tensor
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # Restore
    model.eval()
    with torch.no_grad():
        restored_tensor = model(img_tensor)

    # Convert back to numpy
    restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored = (restored * 255).clip(0, 255).astype(np.uint8)

    # Resize back to original dimensions
    restored = cv2.resize(restored, (original_w, original_h))

    return img, restored


def main():
    parser = argparse.ArgumentParser(description='Inference for image restoration')

    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', type=str, default='output/restored',
                       help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model size')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Model input size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--calculate_metrics', action='store_true',
                       help='Calculate quality metrics (requires clean reference)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = create_vit_restorer(args.model_size, img_size=args.img_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded")

    # Initialize metrics if needed
    if args.calculate_metrics:
        metrics_calc = ImageMetrics(device=device)

    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.tif'))

    print(f"\nProcessing {len(image_files)} images...")

    # Process images
    for img_path in image_files:
        print(f"\nProcessing {img_path.name}...")

        # Restore
        original, restored = restore_image(
            model, img_path, device=device, img_size=args.img_size
        )

        # Save
        output_path = output_dir / f"restored_{img_path.name}"
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
        )
        print(f"✓ Saved to {output_path}")

        # Calculate metrics if reference image exists
        if args.calculate_metrics:
            # Assume clean version is in 'clean' subdirectory
            clean_path = img_path.parent / 'clean' / img_path.name
            if clean_path.exists():
                clean_img = cv2.imread(str(clean_path))
                clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

                # Resize to match
                h, w = restored.shape[:2]
                clean_img = cv2.resize(clean_img, (w, h))

                # Calculate metrics
                metrics = metrics_calc.calculate_all(
                    restored / 255.0,
                    clean_img / 255.0
                )

                print("Metrics:")
                for name, value in metrics.items():
                    print(f"  {name.upper()}: {value:.4f}")

    print(f"\n✓ All images processed! Outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()

