"""
Google Cloud Vision API OCR Engine
Provides OCR using Google's Cloud Vision API (Google Lens technology)
"""

import base64
import requests
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


class GoogleLensOCR:
    """
    OCR engine using Google Cloud Vision API
    This uses the same technology as Google Lens
    """

    def __init__(self, api_key):
        """
        Initialize Google Vision API OCR

        Args:
            api_key: Google Cloud API key
        """
        self.api_key = api_key
        self.api_url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"

    def extract_text(self, image, preprocess=False, lang='san'):
        """
        Extract text from image using Google Cloud Vision API

        Args:
            image: Image (numpy array, PIL Image, or path)
            preprocess: Whether to apply preprocessing (optional for Google Vision)
            lang: Language hint (not strictly required for Google Vision)

        Returns:
            Extracted text string
        """
        # Convert image to base64
        image_base64 = self._prepare_image(image)

        # Prepare API request
        request_data = {
            "requests": [
                {
                    "image": {
                        "content": image_base64
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION",
                            "maxResults": 1
                        }
                    ],
                    "imageContext": {
                        "languageHints": [lang, "hi", "en"]  # Sanskrit, Hindi, English
                    }
                }
            ]
        }

        try:
            # Make API request
            response = requests.post(
                self.api_url,
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                # Extract text from response
                if 'responses' in result and len(result['responses']) > 0:
                    response_data = result['responses'][0]

                    if 'textAnnotations' in response_data and len(response_data['textAnnotations']) > 0:
                        # First annotation contains full text
                        text = response_data['textAnnotations'][0].get('description', '')
                        return text.strip()
                    elif 'fullTextAnnotation' in response_data:
                        text = response_data['fullTextAnnotation'].get('text', '')
                        return text.strip()

                return ""
            else:
                error_msg = f"Google Vision API error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                raise Exception(error_msg)

        except requests.exceptions.Timeout:
            raise Exception("Google Vision API request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Google Vision API request failed: {e}")

    def extract_text_with_confidence(self, image):
        """
        Extract text with confidence scores

        Returns:
            Dictionary with text and confidence information
        """
        image_base64 = self._prepare_image(image)

        request_data = {
            "requests": [
                {
                    "image": {
                        "content": image_base64
                    },
                    "features": [
                        {
                            "type": "DOCUMENT_TEXT_DETECTION",
                            "maxResults": 1
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(
                self.api_url,
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                if 'responses' in result and len(result['responses']) > 0:
                    response_data = result['responses'][0]

                    if 'fullTextAnnotation' in response_data:
                        full_text = response_data['fullTextAnnotation']
                        text = full_text.get('text', '')

                        # Calculate average confidence from pages
                        confidences = []
                        for page in full_text.get('pages', []):
                            if 'confidence' in page:
                                confidences.append(page['confidence'])

                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                        return {
                            'text': text.strip(),
                            'confidence': avg_confidence,
                            'full_response': response_data
                        }

                return {'text': '', 'confidence': 0.0, 'full_response': None}
            else:
                raise Exception(f"Google Vision API error: {response.status_code}")

        except Exception as e:
            raise Exception(f"Google Vision API request failed: {e}")

    def _prepare_image(self, image):
        """
        Convert image to base64 encoded string

        Args:
            image: Image (numpy array, PIL Image, or path)

        Returns:
            Base64 encoded image string
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            with open(image, 'rb') as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')

        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume OpenCV BGR format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image.astype('uint8'))

        # Convert PIL Image to bytes
        if isinstance(image, Image.Image):
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')

        raise ValueError(f"Unsupported image type: {type(image)}")

    def is_available(self):
        """
        Check if the API key is valid and service is available

        Returns:
            Boolean indicating if service is available
        """
        # Simply check if API key is set
        # We don't want to make a real API call here as it costs money
        return bool(self.api_key)


# Convenience function
def create_google_lens_ocr(api_key):
    """
    Factory function to create Google Lens OCR engine

    Args:
        api_key: Google Cloud API key

    Returns:
        GoogleLensOCR instance
    """
    return GoogleLensOCR(api_key)

