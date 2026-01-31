"""
Enhanced Image Restoration Utilities
Implements patch-based processing to maintain high quality
"""

import torch
import cv2
import numpy as np


class EnhancedRestoration:
    """
    Enhanced restoration using patch-based processing

    Benefits:
    - Maintains original resolution
    - Processes in smaller patches to preserve details
    - Applies post-processing for better quality
    - Handles overlapping regions smoothly
    """

    def __init__(self, model, device='cuda', patch_size=256, overlap=32):
        """
        Initialize enhanced restoration

        Args:
            model: Trained ViT restoration model
            device: Device to use ('cuda' or 'cpu')
            patch_size: Size of patches to process (should match model input)
            overlap: Overlap between patches for smooth blending
        """
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.model.eval()

    def restore_image(self, image: np.ndarray, use_patches: bool = True,
                     apply_postprocess: bool = True) -> np.ndarray:
        """
        Restore image with enhanced quality

        Args:
            image: Input image (H, W, 3) in RGB format
            use_patches: If True, use patch-based processing (better quality)
                        If False, use simple resize (faster)
            apply_postprocess: Apply sharpening and enhancement

        Returns:
            Restored image (same size as input)
        """
        if use_patches and (image.shape[0] > self.patch_size or image.shape[1] > self.patch_size):
            # Large image - use patch-based processing
            return self._restore_patches(image, apply_postprocess)
        else:
            # Small image - use simple method
            return self._restore_simple(image, apply_postprocess)

    def _restore_simple(self, image: np.ndarray, apply_postprocess: bool) -> np.ndarray:
        """Simple restoration with resize (faster but lower quality for large images)"""
        h, w = image.shape[:2]

        # Resize to model input size
        img_resized = cv2.resize(image, (self.patch_size, self.patch_size))

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Restore
        with torch.no_grad():
            restored_tensor = self.model(img_tensor)

        # Convert back to numpy
        restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Check if output is in valid range, otherwise normalize
        if restored.max() < 1.0 and restored.min() > -1.0 and (restored.max() - restored.min()) < 0.5:
            # Output seems to be residual or in wrong range - normalize to full range
            restored = (restored - restored.min()) / (restored.max() - restored.min() + 1e-8)

        restored = (restored * 255).clip(0, 255).astype(np.uint8)

        # Resize back to original size
        restored = cv2.resize(restored, (w, h))

        # Apply post-processing
        if apply_postprocess:
            restored = self._apply_postprocess(restored)

        return restored

    def _restore_patches(self, image: np.ndarray, apply_postprocess: bool) -> np.ndarray:
        """
        Patch-based restoration for maintaining high quality

        Process:
        1. Divide image into overlapping patches
        2. Restore each patch independently
        3. Blend overlapping regions
        4. Apply post-processing
        """
        h, w = image.shape[:2]
        stride = self.patch_size - self.overlap

        # Calculate number of patches
        n_patches_h = max(1, (h - self.overlap) // stride)
        n_patches_w = max(1, (w - self.overlap) // stride)

        print(f"   Processing in {n_patches_h}x{n_patches_w} = {n_patches_h * n_patches_w} patches...")

        # Initialize output and weight accumulator
        restored = np.zeros_like(image, dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        # Process each patch
        patch_count = 0
        total_patches = n_patches_h * n_patches_w

        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates
                y1 = i * stride
                x1 = j * stride
                y2 = min(y1 + self.patch_size, h)
                x2 = min(x1 + self.patch_size, w)

                # Adjust if at boundary
                if y2 - y1 < self.patch_size:
                    y1 = max(0, y2 - self.patch_size)
                if x2 - x1 < self.patch_size:
                    x1 = max(0, x2 - self.patch_size)

                # Extract patch
                patch = image[y1:y2, x1:x2]

                # Pad if needed (shouldn't happen, but just in case)
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    patch = cv2.resize(patch, (self.patch_size, self.patch_size))

                # Restore patch
                restored_patch = self._restore_patch(patch)

                # Resize back if needed
                if restored_patch.shape[0] != (y2 - y1) or restored_patch.shape[1] != (x2 - x1):
                    restored_patch = cv2.resize(restored_patch, (x2 - x1, y2 - y1))

                # Add to output with weighting for smooth blending
                weight = self._get_blend_weights(y2 - y1, x2 - x1, self.overlap)
                restored[y1:y2, x1:x2] += restored_patch * weight[:, :, np.newaxis]
                weight_map[y1:y2, x1:x2] += weight

                patch_count += 1
                if patch_count % 10 == 0 or patch_count == total_patches:
                    print(f"   Processed {patch_count}/{total_patches} patches ({100*patch_count//total_patches}%)")

        # Normalize by weight
        weight_map = np.maximum(weight_map, 1e-6)  # Avoid division by zero
        restored = restored / weight_map[:, :, np.newaxis]
        restored = restored.clip(0, 255).astype(np.uint8)

        # Apply post-processing
        if apply_postprocess:
            restored = self._apply_postprocess(restored)

        return restored

    def _restore_patch(self, patch: np.ndarray) -> np.ndarray:
        """Restore a single patch"""
        # Convert to tensor
        patch_tensor = torch.from_numpy(patch).float() / 255.0
        patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Restore
        with torch.no_grad():
            restored_tensor = self.model(patch_tensor)

        # Convert back to numpy
        restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Check if output is in valid range, otherwise normalize
        if restored.max() < 1.0 and restored.min() > -1.0 and (restored.max() - restored.min()) < 0.5:
            # Output seems to be residual or in wrong range - normalize to full range
            restored = (restored - restored.min()) / (restored.max() - restored.min() + 1e-8)

        restored = (restored * 255).clip(0, 255).astype(np.uint8)

        return restored

    def _get_blend_weights(self, h: int, w: int, overlap: int) -> np.ndarray:
        """
        Generate blend weights for smooth patch merging

        Uses linear ramp in overlap regions to avoid seams
        """
        weight = np.ones((h, w), dtype=np.float32)

        if overlap > 0:
            # Top edge
            if h > overlap:
                ramp = np.linspace(0, 1, overlap)
                weight[:overlap, :] *= ramp[:, np.newaxis]

            # Bottom edge
            if h > overlap:
                ramp = np.linspace(1, 0, overlap)
                weight[-overlap:, :] *= ramp[:, np.newaxis]

            # Left edge
            if w > overlap:
                ramp = np.linspace(0, 1, overlap)
                weight[:, :overlap] *= ramp[np.newaxis, :]

            # Right edge
            if w > overlap:
                ramp = np.linspace(1, 0, overlap)
                weight[:, -overlap:] *= ramp[np.newaxis, :]

        return weight

    def _apply_postprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply post-processing for better quality

        - Unsharp mask for sharpening
        - Contrast enhancement
        """
        # Convert to float for processing
        img_float = image.astype(np.float32)

        # Unsharp mask (sharpening)
        gaussian = cv2.GaussianBlur(img_float, (0, 0), 2.0)
        sharpened = cv2.addWeighted(img_float, 1.5, gaussian, -0.5, 0)

        # Clip to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened


def create_enhanced_restorer(model, device='cuda', patch_size=256, overlap=32):
    """
    Factory function to create enhanced restorer

    Args:
        model: Trained restoration model
        device: Device to use
        patch_size: Size of patches (should match model input)
        overlap: Overlap for blending

    Returns:
        EnhancedRestoration instance
    """
    return EnhancedRestoration(model, device, patch_size, overlap)

