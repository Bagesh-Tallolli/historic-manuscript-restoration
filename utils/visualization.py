"""
Visualization utilities for manuscript processing
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path


def visualize_restoration(original, restored, save_path=None):
    """
    Visualize original vs restored images

    Args:
        original: Original image
        restored: Restored image
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original)
    axes[0].set_title('Original Manuscript', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(restored)
    axes[1].set_title('Restored Manuscript', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_pipeline_stages(results, save_path=None):
    """
    Visualize all pipeline stages

    Args:
        results: Results dictionary from pipeline
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))

    # Images
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(results['original_image'])
    ax1.set_title('1. Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(results['restored_image'])
    ax2.set_title('2. Restored Image', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Text results
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    text_info = f"""
    3. OCR & Normalization
    
    Raw OCR:
    {results['ocr_text_raw'][:100]}...
    
    Cleaned Devanagari:
    {results['ocr_text_cleaned'][:100]}...
    
    Words: {results['word_count']}
    """
    ax3.text(0.1, 0.5, text_info, fontsize=10, family='monospace',
             verticalalignment='center')

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    trans_info = f"""
    4. Translation
    
    English:
    {results['translation'][:150]}...
    
    Romanized (IAST):
    {results['romanized'][:100]}...
    """
    ax4.text(0.1, 0.5, trans_info, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved pipeline visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(history_path, save_path=None):
    """
    Plot training history

    Args:
        history_path: Path to training_history.json
        save_path: Path to save plot
    """
    import json

    with open(history_path, 'r') as f:
        history = json.load(f)

    train_history = history['train']
    val_history = history.get('val', [])

    # Extract metrics
    epochs = range(1, len(train_history) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [h['loss'] for h in train_history], label='Train', linewidth=2)
    if val_history:
        ax.plot(epochs, [h['loss'] for h in val_history], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PSNR
    ax = axes[0, 1]
    if 'psnr' in train_history[0]:
        ax.plot(epochs, [h.get('psnr', 0) for h in train_history],
                label='Train', linewidth=2)
        if val_history and 'psnr' in val_history[0]:
            ax.plot(epochs, [h.get('psnr', 0) for h in val_history],
                    label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('PSNR', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # SSIM
    ax = axes[1, 0]
    if 'ssim' in train_history[0]:
        ax.plot(epochs, [h.get('ssim', 0) for h in train_history],
                label='Train', linewidth=2)
        if val_history and 'ssim' in val_history[0]:
            ax.plot(epochs, [h.get('ssim', 0) for h in val_history],
                    label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SSIM')
        ax.set_title('SSIM', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Combined metrics
    ax = axes[1, 1]
    ax.axis('off')

    # Show final metrics
    final_train = train_history[-1]
    info_text = "Final Training Metrics:\n\n"
    for key, value in final_train.items():
        info_text += f"{key}: {value:.4f}\n"

    if val_history:
        final_val = val_history[-1]
        info_text += "\nFinal Validation Metrics:\n\n"
        for key, value in final_val.items():
            info_text += f"{key}: {value:.4f}\n"

    ax.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.suptitle('Training History', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_attention_maps(model, image, save_path=None):
    """
    Visualize attention maps from ViT model

    Args:
        model: ViT restoration model
        image: Input image
        save_path: Path to save visualization
    """
    import torch

    # Get attention maps
    with torch.no_grad():
        attention_maps = model.get_attention_maps(image)

    # Visualize first few layers
    num_layers = min(4, len(attention_maps))

    fig, axes = plt.subplots(2, num_layers // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(num_layers):
        attn = attention_maps[i][0, 0].numpy()  # First head of first batch

        axes[i].imshow(attn, cmap='viridis')
        axes[i].set_title(f'Layer {i+1}', fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('Self-Attention Maps', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def create_demo_figure(results, save_path='demo_output.png'):
    """
    Create a comprehensive demo figure showing all results

    Args:
        results: Results from pipeline
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Sanskrit Manuscript Processing Pipeline',
                 fontsize=18, fontweight='bold', y=0.98)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['original_image'])
    ax1.set_title('Original', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Restored image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['restored_image'])
    ax2.set_title('Restored', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Difference
    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.abs(results['original_image'].astype(float) -
                  results['restored_image'].astype(float))
    ax3.imshow(diff.astype(np.uint8))
    ax3.set_title('Difference', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # OCR text
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    ocr_text = f"""
    OCR Output (Devanagari):
    {results['ocr_text_cleaned']}
    
    Romanized (IAST):
    {results['romanized']}
    """
    ax4.text(0.05, 0.5, ocr_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Translation
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    trans_text = f"""
    English Translation:
    {results['translation']}
    
    Statistics: {results['word_count']} words, {len(results['sentences'])} sentences
    """
    ax5.text(0.05, 0.5, trans_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved demo figure to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities loaded")
    print("Use these functions to visualize pipeline results")

