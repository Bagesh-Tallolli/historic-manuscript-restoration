"""
OCR pipeline for Sanskrit/Devanagari text extraction
"""

import numpy as np
from PIL import Image
import pytesseract
from pathlib import Path
import cv2

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("Warning: transformers not installed. TrOCR will not be available.")

from ocr.preprocess import OCRPreprocessor, resize_with_aspect_ratio


class SanskritOCR:
    """OCR engine for Sanskrit/Devanagari manuscripts"""

    def __init__(self, engine='tesseract', preprocessor=None):
        """
        Args:
            engine: 'tesseract' or 'trocr'
            preprocessor: OCRPreprocessor instance (creates new if None)
        """
        self.engine = engine
        self.preprocessor = preprocessor or OCRPreprocessor()

        if engine == 'trocr':
            self._init_trocr()
        elif engine == 'tesseract':
            self._check_tesseract()

    def _check_tesseract(self):
        """Check if Tesseract is installed and has Devanagari support"""
        try:
            # Test tesseract
            pytesseract.get_tesseract_version()

            # Check for Devanagari language pack
            langs = pytesseract.get_languages()
            if 'san' not in langs and 'hin' not in langs:
                print("Warning: Devanagari language pack not found in Tesseract")
                print("Available languages:", langs)
                print("Install with: sudo apt-get install tesseract-ocr-san")
        except Exception as e:
            print(f"Warning: Tesseract not properly configured: {e}")

    def _init_trocr(self):
        """Initialize TrOCR model"""
        if not TROCR_AVAILABLE:
            raise ImportError("transformers library required for TrOCR")

        print("Loading TrOCR model...")
        # Using base TrOCR model - for best results, fine-tune on Devanagari
        self.trocr_processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-base-handwritten'
        )
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-base-handwritten'
        )
        print("TrOCR model loaded")

    def extract_text(self, image, preprocess=True, lang='san'):
        """
        Extract text from image

        Args:
            image: Image (numpy array, PIL Image, or path)
            preprocess: Whether to apply preprocessing
            lang: Language code ('san' for Sanskrit, 'hin' for Hindi)

        Returns:
            Extracted text string
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Preprocess
        if preprocess:
            image = self.preprocessor.preprocess(image)

        # Run OCR based on engine
        if self.engine == 'tesseract':
            text = self._tesseract_ocr(image, lang)
        elif self.engine == 'trocr':
            text = self._trocr_ocr(image)
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine}")

        return text

    def _tesseract_ocr(self, image, lang='san'):
        """Run Tesseract OCR"""
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Configure Tesseract
        # Try Sanskrit first, fallback to Hindi if not available
        try:
            config = '--psm 6'  # Assume uniform block of text
            text = pytesseract.image_to_string(image, lang=lang, config=config)
        except pytesseract.TesseractError:
            # Try Hindi as fallback
            print(f"Warning: Language '{lang}' not found, trying 'hin'")
            text = pytesseract.image_to_string(image, lang='hin', config=config)

        return text.strip()

    def _trocr_ocr(self, image):
        """Run TrOCR"""
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            # TrOCR expects RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)

        # Prepare image
        pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values

        # Generate text
        generated_ids = self.trocr_model.generate(pixel_values)
        generated_text = self.trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def extract_text_with_layout(self, image, preprocess=True, lang='san'):
        """
        Extract text with layout information (lines, words, confidence)

        Returns:
            Dictionary with structured OCR data
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if preprocess:
            image = self.preprocessor.preprocess(image, apply_all=False)
            image = self.preprocessor.grayscale(image)

        # Convert to PIL
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Get detailed data from Tesseract
        try:
            data = pytesseract.image_to_data(
                pil_image, lang=lang, output_type=pytesseract.Output.DICT
            )
        except:
            data = pytesseract.image_to_data(
                pil_image, lang='hin', output_type=pytesseract.Output.DICT
            )

        # Organize by lines
        lines = {}
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                line_num = data['line_num'][i]
                if line_num not in lines:
                    lines[line_num] = []

                lines[line_num].append({
                    'text': data['text'][i],
                    'confidence': data['conf'][i],
                    'bbox': (data['left'][i], data['top'][i],
                            data['width'][i], data['height'][i])
                })

        # Combine into full text with line breaks
        full_text = []
        for line_num in sorted(lines.keys()):
            line_words = [word['text'] for word in lines[line_num]]
            full_text.append(' '.join(line_words))

        return {
            'text': '\n'.join(full_text),
            'lines': lines,
            'raw_data': data
        }

    def extract_from_lines(self, image, preprocess=True, lang='san'):
        """
        Segment image into lines and extract text from each

        Returns:
            List of (line_image, text) tuples
        """
        # Load and preprocess
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if preprocess:
            processed = self.preprocessor.preprocess(image)
        else:
            processed = self.preprocessor.grayscale(image)

        # Segment into lines
        lines = self.preprocessor.line_segmentation(processed)

        # Extract text from each line
        results = []
        for line_img in lines:
            text = self._tesseract_ocr(line_img, lang)
            if text.strip():
                results.append((line_img, text))

        return results


class EnsembleOCR:
    """Combine multiple OCR engines for better accuracy"""

    def __init__(self):
        self.engines = []

        # Add Tesseract
        try:
            self.engines.append(SanskritOCR(engine='tesseract'))
        except:
            pass

        # Add TrOCR if available
        if TROCR_AVAILABLE:
            try:
                self.engines.append(SanskritOCR(engine='trocr'))
            except:
                pass

        if not self.engines:
            raise RuntimeError("No OCR engines available")

        print(f"Initialized ensemble with {len(self.engines)} engines")

    def extract_text(self, image, preprocess=True, voting='majority'):
        """
        Extract text using ensemble of OCR engines

        Args:
            image: Input image
            preprocess: Whether to preprocess
            voting: 'majority' or 'concat' (concatenate results)

        Returns:
            Combined OCR result
        """
        results = []

        for engine in self.engines:
            try:
                text = engine.extract_text(image, preprocess=preprocess)
                results.append(text)
            except Exception as e:
                print(f"Warning: Engine failed with error: {e}")

        if not results:
            return ""

        if voting == 'concat':
            # Return concatenation with separator
            return " | ".join(results)
        else:
            # Return most common result (simple majority voting)
            from collections import Counter
            counter = Counter(results)
            return counter.most_common(1)[0][0]


if __name__ == "__main__":
    print("Testing Sanskrit OCR...")

    # Test with sample text
    test_img = np.ones((200, 800, 3), dtype=np.uint8) * 255
    cv2.putText(
        test_img, "Sanskrit Text Example",
        (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2
    )

    # Test Tesseract
    try:
        ocr = SanskritOCR(engine='tesseract')
        text = ocr.extract_text(test_img, preprocess=False, lang='eng')
        print(f"\nTesseract result: {text}")
    except Exception as e:
        print(f"Tesseract error: {e}")

    print("\nOCR module ready!")

