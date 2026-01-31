"""
ManuscriptVision-Agent — Complete Pipeline
Sanskrit manuscript restoration, OCR, correction, and multilingual translation
Uses API-based vision and language models only (NO custom trained models)
"""
import io
import json
import os
import base64
from typing import Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from google import genai
from google.genai import types

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAxTN1rfdyJQVE3ecy5d8Zqkl5I431nBh0")
DEFAULT_MODEL = "gemini-2.0-flash-exp"

# COMPLETE AGENT PROMPT
AGENT_SYSTEM_PROMPT = """You are **ManuscriptVision-Agent**, an expert agent for **Sanskrit manuscript image restoration, OCR extraction, text correction, and multilingual translation**.

Your task is to analyze the provided Sanskrit manuscript image and execute the complete pipeline in strict order.

## **STEP 1 — READABILITY-FIRST IMAGE RESTORATION ANALYSIS**
Analyze the image quality and describe what restoration improvements are needed:
* Brightness, contrast, and sharpness issues
* Noise, stains, blur, folds, background discoloration
* Text readability and clarity

## **STEP 2 — OCR EXTRACTION**
Extract Sanskrit text in **pure Unicode Devanagari**:
* Preserve matras, ligatures (संयुक्ताक्षर), anusvāra, visarga, virāma
* Extract exactly what you see in the image

## **STEP 3 — OCR TEXT CORRECTION**
Correct OCR errors only:
* Fix broken characters, missing matras, split ligatures
* Normalize to valid Sanskrit grammar and Unicode Devanagari
* ❌ Do NOT invent missing words or lines
* ❌ Do NOT change meaning

## **STEP 4 — MULTILINGUAL TRANSLATION**
Translate the corrected Sanskrit text into:
* **English** (accurate, literal)
* **Hindi** (अर्थ सुरक्षित रखते हुए)
* **Kannada** (ಅರ್ಥವನ್ನು ಕಾಪಾಡುವಂತೆ)

Rules:
* Preserve original meaning
* No poetic rewriting
* No hallucination

## **STEP 5 — VERIFICATION**
* Ensure OCR text matches visible image
* Ensure translations align with corrected Sanskrit
* Generate confidence score (0.0 to 1.0)

## **OUTPUT FORMAT (STRICT — JSON ONLY)**
Respond ONLY with valid JSON in this exact format:
```json
{
  "restored_image_quality": "description of image quality and needed improvements",
  "ocr_extracted_text": "raw OCR text in Devanagari",
  "corrected_sanskrit_text": "corrected Sanskrit text in Devanagari",
  "english_translation": "accurate English translation",
  "hindi_translation": "Hindi translation in Devanagari",
  "kannada_translation": "Kannada translation in Kannada script",
  "confidence_score": "0.0-1.0"
}
```

**IMPORTANT**: Output ONLY the JSON object. No other text before or after."""


class ManuscriptVisionAgent:
    """Complete pipeline agent for Sanskrit manuscript processing"""

    def __init__(self, api_key: str = None):
        """Initialize the agent with Gemini API"""
        self.api_key = api_key or GEMINI_API_KEY
        self.client = genai.Client(api_key=self.api_key)
        self.model = DEFAULT_MODEL

    def restore_image_api_based(self, image: Image.Image) -> Image.Image:
        """
        STEP 1: API-based image restoration (readability-first)
        Improve brightness, contrast, sharpness without custom models
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Auto-contrast and brightness adjustment
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast

        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)  # Increase brightness

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # Increase sharpness

        # Denoise using PIL filters
        image = image.filter(ImageFilter.MedianFilter(size=3))

        # Normalize background to light parchment tone
        img_array = np.array(image)

        # Convert to LAB color space for better processing
        # Simple background normalization: brighten darker areas
        img_array = img_array.astype(np.float32)

        # Increase brightness of darker pixels (background)
        gray = np.mean(img_array, axis=2)
        mask = gray < 200
        img_array[mask] = np.clip(img_array[mask] * 1.3, 0, 255)

        # Ensure text remains dark (pixels below threshold)
        text_mask = gray < 100
        img_array[text_mask] = np.clip(img_array[text_mask] * 0.8, 0, 255)

        restored_image = Image.fromarray(img_array.astype(np.uint8))

        return restored_image

    def extract_and_process_with_gemini(self, image: Image.Image) -> Dict[str, Any]:
        """
        STEPS 2-5: Use Gemini Vision to extract, correct, translate, and verify
        """
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Prepare content with image and agent prompt
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=AGENT_SYSTEM_PROMPT),
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/png"
                    ),
                ],
            ),
        ]

        # Configure generation
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,  # Low temperature for accuracy
            response_mime_type="application/json",
        )

        # Generate content
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            )

            # Parse JSON response
            result_text = response.text.strip()

            # Remove markdown code blocks if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            result_text = result_text.strip()

            result = json.loads(result_text)
            return result

        except json.JSONDecodeError as e:
            # Fallback: return raw text in structured format
            return {
                "restored_image_quality": "Unable to parse response",
                "ocr_extracted_text": response.text if hasattr(response, 'text') else "",
                "corrected_sanskrit_text": "",
                "english_translation": "",
                "hindi_translation": "",
                "kannada_translation": "",
                "confidence_score": "0.0",
                "error": f"JSON parse error: {str(e)}"
            }
        except Exception as e:
            return {
                "restored_image_quality": "Error during processing",
                "ocr_extracted_text": "",
                "corrected_sanskrit_text": "",
                "english_translation": "",
                "hindi_translation": "",
                "kannada_translation": "",
                "confidence_score": "0.0",
                "error": str(e)
            }

    def process_manuscript(self, image: Image.Image) -> Dict[str, Any]:
        """
        Execute the complete pipeline on a manuscript image

        Args:
            image: PIL Image of Sanskrit manuscript

        Returns:
            Dictionary with all pipeline results including restored image
        """
        # STEP 1: Restore image (API-based, no custom models)
        restored_image = self.restore_image_api_based(image)

        # STEPS 2-5: Extract, correct, translate, verify with Gemini
        results = self.extract_and_process_with_gemini(restored_image)

        # Add restored image to results
        results['restored_image'] = restored_image
        results['original_image'] = image

        return results

    def process_manuscript_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Process manuscript from file path

        Args:
            image_path: Path to manuscript image file

        Returns:
            Dictionary with all pipeline results
        """
        image = Image.open(image_path)
        return self.process_manuscript(image)


def main():
    """Command-line interface for the agent"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python manuscript_vision_agent.py <image_path>")
        print("Example: python manuscript_vision_agent.py sample.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("ManuscriptVision-Agent — Complete Pipeline")
    print("=" * 60)
    print(f"\n📄 Processing: {image_path}\n")

    # Initialize agent
    agent = ManuscriptVisionAgent()

    # Process manuscript
    print("🔧 Step 1: API-based image restoration...")
    print("📖 Step 2: OCR extraction...")
    print("✏️  Step 3: Text correction...")
    print("🌍 Step 4: Multilingual translation...")
    print("✓  Step 5: Verification...\n")

    results = agent.process_manuscript_from_path(image_path)

    # Save restored image
    if 'restored_image' in results:
        output_path = image_path.rsplit('.', 1)[0] + '_restored.png'
        results['restored_image'].save(output_path)
        print(f"💾 Restored image saved: {output_path}\n")
        del results['restored_image']  # Remove from JSON output
        del results['original_image']

    # Display results as formatted JSON
    print("=" * 60)
    print("RESULTS (JSON FORMAT)")
    print("=" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("=" * 60)


if __name__ == "__main__":
    main()

