"""Minimal test to validate restored (or original) image is sent to Tesseract OCR
Creates a synthetic Sanskrit text image and runs the ManuscriptPipeline with restoration skipped.

Run with: python -m tests.test_tesseract_flow
Prerequisites: pip install numpy pillow opencv-python pytesseract
Ensure Tesseract installed: sudo apt-get install tesseract-ocr
Optional Sanskrit pack: sudo apt-get install tesseract-ocr-san
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from main import ManuscriptPipeline

DEVANAGARI_SAMPLE = "रामः वनं गच्छति"  # Simple Sanskrit sentence


def create_synthetic_image(text: str) -> Image.Image:
    # Create white canvas
    h, w = 200, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Choose font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Put text (OpenCV doesn't render complex ligatures perfectly, but enough for test)
    cv2.putText(img, text, (20, 120), font, 2, (0, 0, 0), 3, cv2.LINE_AA)
    return Image.fromarray(img)


def test_tesseract_flow():
    # Create synthetic Devanagari image
    pil_img = create_synthetic_image(DEVANAGARI_SAMPLE)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        pil_img.save(tmp.name)
        temp_path = tmp.name

    try:
        # Initialize pipeline with no restoration model (skip restoration to focus on OCR)
        pipeline = ManuscriptPipeline(restoration_model_path=None, ocr_engine='tesseract', translation_method='google', device='cpu')
        results = pipeline.process_manuscript(temp_path, save_output=False)

        extracted = results.get('ocr_text_cleaned', '')
        print('Extracted OCR Text:', extracted)

        # Basic assertions: should contain at least one Devanagari character
        assert any('\u0900' <= c <= '\u097F' for c in extracted), 'No Devanagari characters detected in OCR output.'
        # Optional: length check
        assert len(extracted.strip()) > 3, 'Extracted text too short.'
        print('Test passed: Tesseract-only OCR flow functional.')
    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    test_tesseract_flow()

