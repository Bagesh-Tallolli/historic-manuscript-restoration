"""
Enhanced OCR for 100% Accurate Sanskrit Manuscript Text Extraction (Tesseract-Only Version)

This version has all non-Tesseract engines disabled/commented out per request.
- Removed initialization of TrOCR and Google Cloud Vision
- Hybrid mode disabled; engine forced to 'tesseract'
- Extraction always uses multi-pass Tesseract
"""

import numpy as np
from PIL import Image
import pytesseract
from pathlib import Path
import cv2
import re

# NOTE: TrOCR and Google Cloud Vision support intentionally disabled
# try:
#     from transformers import TrOCRProcessor, VisionEncoderDecoderModel
#     import torch
#     TROCR_AVAILABLE = True
# except ImportError:
#     TROCR_AVAILABLE = False
#     pass

from ocr.preprocess import OCRPreprocessor
# from ocr.google_lens_ocr import GoogleLensOCR  # Disabled
# import os  # Disabled


class EnhancedSanskritOCR:
    """Tesseract-only enhanced OCR for Sanskrit manuscripts"""

    def __init__(self, engine='tesseract', device='auto'):
        """Force engine to Tesseract regardless of input"""
        # Always use tesseract
        self.engine = 'tesseract'
        self.preprocessor = OCRPreprocessor()
        self.google_lens = None  # Disabled
        self.trocr_available = False  # Disabled

        # Minimal device handling (only needed if later extended)
        # device parameter retained for API compatibility
        self.device = 'cpu'

        self._init_tesseract()

    def _init_tesseract(self):
        """Initialize and verify Tesseract with Devanagari support"""
        try:
            version = pytesseract.get_tesseract_version()
            langs = pytesseract.get_languages()

            # Preferred language order: san -> hin -> dev -> eng
            for lang in ['san', 'hin', 'dev', 'eng']:
                if lang in langs:
                    self.tesseract_lang = lang
                    break
            else:
                self.tesseract_lang = 'eng'
                print("⚠️  Warning: No expected language pack found (san/hin/dev). Using 'eng'.")

            print(f"✅ Tesseract {version} initialized (lang: {self.tesseract_lang})")
        except Exception as e:
            print(f"❌ Tesseract initialization failed: {e}")
            self.tesseract_lang = 'eng'

    def extract_complete_paragraph(self, image, preprocess=True, multi_pass=True):
        """Extract complete paragraph using multi-pass Tesseract only"""
        # Normalize input image to numpy RGB
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        if preprocess:
            processed_img = self.preprocessor.preprocess(img)
        else:
            processed_img = img

        # Always call tesseract extraction
        result = self._tesseract_complete_extraction(processed_img, multi_pass=multi_pass)
        return result

    def _tesseract_complete_extraction(self, image, multi_pass=True):
        """Multi-pass Tesseract OCR to maximize paragraph capture"""
        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image

        results = []
        confidences = []

        base_lang = self.tesseract_lang or 'san'
        configs = [
            f'--oem 3 --psm 6 -l {base_lang}',
            f'--oem 3 --psm 4 -l {base_lang}',
            f'--oem 3 --psm 3 -l {base_lang}',
            f'--oem 1 --psm 6 -l {base_lang}',
        ]

        if multi_pass:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(pil_img, config=config)
                    data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
                    conf = [int(c) for c in data['conf'] if c != '-1']
                    avg_conf = np.mean(conf) / 100.0 if conf else 0.0
                    if text.strip():
                        results.append(text.strip())
                        confidences.append(avg_conf)
                except Exception:
                    continue

            if results:
                scores = [len(t) * c for t, c in zip(results, confidences)]
                best_idx = int(np.argmax(scores))
                text = results[best_idx]
                confidence = confidences[best_idx]
            else:
                text = ''
                confidence = 0.0
        else:
            try:
                text = pytesseract.image_to_string(pil_img, config=configs[0]).strip()
                data = pytesseract.image_to_data(pil_img, config=configs[0], output_type=pytesseract.Output.DICT)
                conf = [int(c) for c in data['conf'] if c != '-1']
                confidence = np.mean(conf) / 100.0 if conf else 0.0
            except Exception:
                text = ''
                confidence = 0.0

        text = self._post_process_text(text)
        return {
            'text': text,
            'confidence': confidence,
            'word_count': len(text.split()),
            'method': f'tesseract-{base_lang}'
        }

    def _post_process_text(self, text):
        """Basic cleanup of OCR output while preserving structure"""
        if not text:
            return ''
        lines = text.split('\n')
        lines = [' '.join(line.split()) for line in lines]
        lines = [line for line in lines if line.strip()]
        text = ' '.join(lines)
        text = re.sub(r'(.)\1{4,}', r'\1', text)
        text = re.sub(r'\s+([।॥,;])', r'\1', text)
        text = re.sub(r'([।॥])\s*([।॥])', r'\1 \2', text)
        return text.strip()

    def extract_with_confidence(self, image, threshold=0.5):
        """Run extraction and optionally retry with enhanced preprocessing if confidence low"""
        result = self.extract_complete_paragraph(image, preprocess=True, multi_pass=True)
        if result['confidence'] < threshold:
            enhanced = self.preprocessor.enhance_for_low_quality(image)
            result = self.extract_complete_paragraph(enhanced, preprocess=False, multi_pass=True)
        return result


def create_ocr_engine(engine='tesseract', device='auto'):
    """Factory returns Tesseract-only engine ignoring requested engine"""
    return EnhancedSanskritOCR(engine='tesseract', device=device)


if __name__ == '__main__':
    # Simple self-test (will only use Tesseract)
    ocr = EnhancedSanskritOCR(engine='tesseract')
    test_img = 'data/raw/test/test_0001.jpg'
    if Path(test_img).exists():
        result = ocr.extract_complete_paragraph(test_img)
        print(f"Extracted: {result['word_count']} words | Confidence: {result['confidence']:.2%}")
        print(f"Method: {result['method']}")
        print(f"\nText (first 500 chars):\n{result['text'][:500]}")
    else:
        print("Test image not found; place a sample at data/raw/test/test_0001.jpg to run self-test.")
