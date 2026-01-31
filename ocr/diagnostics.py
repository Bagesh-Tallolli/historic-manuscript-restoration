"""Diagnostics utilities for Sanskrit OCR completeness.
Computes image quality metrics to help decide fallback strategies.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Any, Tuple
from PIL import Image


def _to_cv(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def compute_image_quality(image) -> Dict[str, Any]:
    """Compute basic quality metrics.
    Returns dict with:
        width, height, megapixels, blur (variance of Laplacian),
        contrast (std-dev grayscale), brightness (mean), aspect_ratio
    """
    img_cv = _to_cv(image)
    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Blur score
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur = lap.var()
    # Contrast
    contrast = float(gray.std())
    # Brightness
    brightness = float(gray.mean())
    megapixels = (w * h) / 1_000_000.0
    aspect_ratio = round(w / h, 3)
    return {
        'width': w,
        'height': h,
        'megapixels': megapixels,
        'blur': blur,
        'contrast': contrast,
        'brightness': brightness,
        'aspect_ratio': aspect_ratio,
    }


def summarize_metrics(metrics: Dict[str, Any]) -> str:
    return (f"{metrics['width']}x{metrics['height']}px | {metrics['megapixels']:.2f}MP | "
            f"Blur={metrics['blur']:.1f} | Contrast={metrics['contrast']:.1f} | "
            f"Brightness={metrics['brightness']:.1f}")


def is_low_quality(metrics: Dict[str, Any]) -> bool:
    # Heuristic thresholds tuned loosely; can refine later
    return (metrics['blur'] < 80 or metrics['contrast'] < 30 or metrics['brightness'] < 40)


__all__ = [
    'compute_image_quality', 'summarize_metrics', 'is_low_quality'
]

