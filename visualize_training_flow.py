"""
Visualize the Training Flow for Sanskrit Manuscript Restoration
This script demonstrates the exact flow shown in TRAINING_PROCESS.md
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random

# Import project modules
from utils.dataset_loader import ManuscriptDataset
from models.vit_restorer import create_vit_restorer


def visualize_training_flow(image_path, output_dir='output/training_flow'):
    """
    Demonstrate the complete training flow:
    Clean Image → Synthetic Degradation → ViT Model → Restored Image → Loss Calculation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("TRAINING FLOW VISUALIZATION")
    print("=" * 80)
    print()

    # ========================================================================
    # STEP 1: Load Clean Image (from dataset)
    # ========================================================================
    print("STEP 1: Load Clean Image (from dataset)")
    print("-" * 80)

    clean_img = cv2.imread(str(image_path))
    if clean_img is None:
        clean_img = np.array(Image.open(image_path).convert('RGB'))
    else:
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

    print(f"✓ Loaded image: {image_path}")
    print(f"  Original size: {clean_img.shape[1]}×{clean_img.shape[0]}")
    print(f"  Data type: {clean_img.dtype}")
    print()

    # Resize for training
    img_size = 256
    clean_resized = cv2.resize(clean_img, (img_size, img_size))
    print(f"✓ Resized to training size: {img_size}×{img_size}")
    print()

    # Save
    cv2.imwrite(str(output_dir / '1_clean_original.jpg'),
                cv2.cvtColor(clean_resized, cv2.COLOR_RGB2BGR))

    # ========================================================================
    # STEP 2: Apply Synthetic Degradation
    # ========================================================================
    print("STEP 2: Apply Synthetic Degradation (noise, blur, contrast reduction)")
    print("-" * 80)

    degraded_img = apply_degradation_verbose(clean_resized.copy())

    # Save
    cv2.imwrite(str(output_dir / '2_degraded_input.jpg'),
                cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR))
    print()

    # ========================================================================
    # STEP 3: Prepare input for ViT Model
    # ========================================================================
    print("STEP 3: Degraded Image (input to model)")
    print("-" * 80)

    # Convert to tensor
    degraded_tensor = torch.from_numpy(
        degraded_img.astype(np.float32).transpose(2, 0, 1) / 255.0
    ).unsqueeze(0)  # Add batch dimension

    clean_tensor = torch.from_numpy(
        clean_resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    ).unsqueeze(0)

    print(f"✓ Converted to tensor")
    print(f"  Degraded tensor shape: {degraded_tensor.shape}")
    print(f"  Value range: [{degraded_tensor.min():.3f}, {degraded_tensor.max():.3f}]")
    print(f"  Data type: {degraded_tensor.dtype}")
    print()

    # ========================================================================
    # STEP 4: ViT Restoration Model
    # ========================================================================
    print("STEP 4: ViT Restoration Model")
    print("-" * 80)

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = create_vit_restorer(model_size='base', img_size=img_size, patch_size=16)
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Created ViT Restorer model")
    print(f"  Model size: base")
    print(f"  Image size: {img_size}×{img_size}")
    print(f"  Patch size: 16×16")
    print(f"  Total parameters: {total_params:,}")
    print()

    print("Processing through transformer layers...")
    print("  - Patch Embedding (split into 16×16 patches)")
    print("  - Positional Encoding")
    print("  - 12 Transformer Encoder layers")
    print("  - Multi-Head Self-Attention")
    print("  - Feed-Forward Networks")
    print("  - Decoder (upsampling + reconstruction)")
    print()

    # ========================================================================
    # STEP 5: Restored Image (output)
    # ========================================================================
    print("STEP 5: Restored Image (output)")
    print("-" * 80)

    with torch.no_grad():
        degraded_tensor = degraded_tensor.to(device)
        restored_tensor = model(degraded_tensor)

    # Convert back to numpy
    restored_img = restored_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    restored_img = (restored_img * 255).clip(0, 255).astype(np.uint8)

    print(f"✓ Forward pass complete")
    print(f"  Restored tensor shape: {restored_tensor.shape}")
    print(f"  Output value range: [{restored_tensor.min():.3f}, {restored_tensor.max():.3f}]")
    print()

    # Save
    cv2.imwrite(str(output_dir / '3_restored_output.jpg'),
                cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))

    # ========================================================================
    # STEP 6: Compare with Original Clean Image (loss calculation)
    # ========================================================================
    print("STEP 6: Compare with Original Clean Image (loss calculation)")
    print("-" * 80)

    # Calculate losses
    restored_tensor = restored_tensor.to(device)
    clean_tensor = clean_tensor.to(device)

    # L1 Loss
    l1_loss = torch.nn.functional.l1_loss(restored_tensor, clean_tensor)
    print(f"✓ L1 Loss (pixel-wise): {l1_loss.item():.6f}")

    # MSE Loss
    mse_loss = torch.nn.functional.mse_loss(restored_tensor, clean_tensor)
    print(f"✓ MSE Loss: {mse_loss.item():.6f}")

    # PSNR (Peak Signal-to-Noise Ratio)
    psnr = 10 * torch.log10(1.0 / mse_loss)
    print(f"✓ PSNR: {psnr.item():.2f} dB")

    # SSIM (would require additional library)
    print(f"✓ SSIM: (requires training to calculate)")
    print()

    print("Loss backward pass (during training):")
    print("  1. Calculate gradient: ∂Loss/∂weights")
    print("  2. Clip gradients (prevent explosion)")
    print("  3. Optimizer step: update weights using AdamW")
    print("  4. Learning rate scheduler step")
    print()

    # ========================================================================
    # Create Comparison Visualization
    # ========================================================================
    print("Creating comparison visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(clean_resized)
    axes[0, 0].set_title('1. Clean Image (Original)\nFrom Dataset', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(degraded_img)
    axes[0, 1].set_title('2. Degraded Image (Input)\nSynthetic Degradation Applied', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(restored_img)
    axes[1, 0].set_title('3. Restored Image (Output)\nViT Model Prediction', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Show difference map
    diff = np.abs(clean_resized.astype(float) - restored_img.astype(float))
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
    axes[1, 1].imshow(diff_normalized)
    axes[1, 1].set_title(f'4. Difference Map\nL1 Loss: {l1_loss.item():.4f} | PSNR: {psnr.item():.2f} dB',
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    plt.suptitle('Training Flow: Clean → Degraded → ViT Model → Restored → Loss',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    comparison_path = output_dir / 'training_flow_comparison.jpg'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison: {comparison_path}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("SUMMARY: Training Flow Complete!")
    print("=" * 80)
    print()
    print("Flow executed:")
    print("  1. ✓ Clean Image (from dataset)")
    print("  2. ✓ Apply Synthetic Degradation (noise, blur, contrast)")
    print("  3. ✓ Degraded Image (input to model)")
    print("  4. ✓ ViT Restoration Model (forward pass)")
    print("  5. ✓ Restored Image (output)")
    print("  6. ✓ Compare with Original Clean Image (loss calculation)")
    print()
    print("During training, this process repeats for:")
    print(f"  - 415 training images per epoch")
    print(f"  - 16 images per batch = ~26 batches")
    print(f"  - 100 epochs = 41,500 training steps")
    print(f"  - Weights updated after each batch")
    print()
    print(f"Output saved to: {output_dir}/")
    print("=" * 80)


def apply_degradation_verbose(img):
    """
    Apply synthetic degradation with verbose output
    Shows what degradations are applied
    """
    img = img.astype(np.float32) / 255.0
    degradations_applied = []

    # 1. Gaussian noise
    if random.random() > 0.3:
        noise_sigma = 0.05
        noise = np.random.normal(0, noise_sigma, img.shape)
        img = img + noise
        degradations_applied.append(f"Gaussian noise (σ={noise_sigma})")

    # 2. Gaussian blur
    if random.random() > 0.3:
        kernel_size = 5
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        degradations_applied.append(f"Gaussian blur (kernel={kernel_size})")

    # 3. Reduce contrast
    if random.random() > 0.4:
        alpha = 0.7  # contrast
        beta = 0.1   # brightness
        img = alpha * img + beta
        degradations_applied.append(f"Contrast reduction (α={alpha}, β={beta})")

    # 4. Salt and pepper noise
    if random.random() > 0.5:
        noise_ratio = 0.005
        mask = np.random.random(img.shape[:2]) < noise_ratio
        if mask.sum() > 0:
            img[mask] = np.random.choice([0, 1], size=(mask.sum(), 3))
        degradations_applied.append(f"Salt & pepper noise (ratio={noise_ratio})")

    # 5. Yellowish tint (aging)
    if random.random() > 0.4:
        yellow_tint = np.array([1.0, 0.95, 0.8])
        img = img * yellow_tint
        degradations_applied.append("Yellow tint (aging effect)")

    print("Degradations applied:")
    for i, deg in enumerate(degradations_applied, 1):
        print(f"  {i}. {deg}")

    # Clip values
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


if __name__ == "__main__":
    import sys

    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use first training image
        train_dir = Path('data/raw/train')
        images = list(train_dir.glob('*.jpg'))
        if not images:
            print("Error: No training images found!")
            print(f"Please add images to {train_dir}")
            sys.exit(1)
        image_path = images[0]

    visualize_training_flow(image_path)

