"""
Backend functions for Sanskrit Manuscript Processing
DO NOT MODIFY - Contains all processing logic
"""
import io
import numpy as np
import cv2
from PIL import Image
from google import genai
from google.genai import types


# API Configuration
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_MODEL = "gemini-2.5-flash"

# OCR System Prompt
OCR_PROMPT = """You are an expert Sanskrit scholar specializing in manuscript digitization.

From the provided manuscript image, extract all visible Sanskrit text accurately in Devanagari script.
Correct any obvious transcription errors.

Output only the Sanskrit text in Devanagari, nothing else."""

# Translation Prompt
TRANSLATION_PROMPT = """You are an expert Sanskrit-to-English, Sanskrit-to-Hindi, and Sanskrit-to-Kannada translator.

Translate the following Sanskrit text into English, Hindi, and Kannada.
Preserve the scholarly and poetic meaning. Avoid literal word-by-word translation.
If verses are incomplete, provide context where possible.

Sanskrit Text:
{sanskrit_text}

Output format (STRICT):

**Extracted Sanskrit Text:**
{sanskrit_text}

**English Meaning:**
[English translation]

**हिंदी अर्थ:**
[Hindi translation in Devanagari]

**ಕನ್ನಡ ಅರ್ಥ:**
[Kannada translation]
"""


def enhance_manuscript_simple(image_pil):
    """
    Enhance manuscript using CLAHE + Unsharp Mask

    Args:
        image_pil: PIL Image

    Returns:
        Enhanced PIL Image
    """
    try:
        # Convert to numpy
        img = np.array(image_pil.convert("RGB"))

        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # Apply unsharp mask for sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Convert back to PIL
        return Image.fromarray(sharpened)
    except Exception as e:
        return image_pil


def perform_ocr_translation(client, image_pil, prompt_text, model_name, temperature=0.3):
    """
    Perform OCR and translation using Gemini

    Returns:
        Translated text or None if failed
    """
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Prepare content
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_text),
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/png"
                    ),
                ],
            ),
        ]

        # Configure generation
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
        )

        # Generate content with streaming
        result_text = ""
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if hasattr(chunk, 'text') and chunk.text:
                result_text += chunk.text

        return result_text.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

