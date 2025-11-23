"""
OCR package for Sanskrit manuscript text extraction
"""

from ocr.preprocess import OCRPreprocessor, resize_with_aspect_ratio
from ocr.run_ocr import SanskritOCR, EnsembleOCR

__all__ = [
    'OCRPreprocessor',
    'resize_with_aspect_ratio',
    'SanskritOCR',
    'EnsembleOCR'
]

