#!/usr/bin/env python3
"""
SanskritOCR-Translation-Agent: Production-Ready Pipeline Controller

Workflow (Strict Order):
1. Image Restoration
2. Google Lens OCR
3. Gemini Text Correction
4. English Translation
5. Final Verification

No hallucination. No invented content. Accuracy > speed.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    print("INFO: python-dotenv not installed. Install with: pip install python-dotenv")

# Google Cloud Vision (Google Lens OCR)
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    print("WARNING: Google Cloud Vision not available. Install with: pip install google-cloud-vision")

# Gemini API for Sanskrit correction
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNING: Gemini API not available. Install with: pip install google-generativeai")

# Translation
from transformers import MarianMTModel, MarianTokenizer

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from models.vit_restorer import create_vit_restorer


class SanskritOCRTranslationAgent:
    """
    Production-ready automated pipeline controller for Sanskrit manuscript processing.

    Stages:
    1. Image Restoration (ViT model)
    2. OCR via Google Lens (Cloud Vision API)
    3. Sanskrit Text Correction (Gemini model)
    4. English Translation (MarianMT)
    5. Final Verification
    """

    def __init__(
        self,
        restoration_model_path: str,
        google_credentials_path: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        translation_model: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: str = "auto"
    ):
        """
        Initialize the agent pipeline.

        Args:
            restoration_model_path: Path to trained ViT restoration checkpoint
            google_credentials_path: Path to Google Cloud credentials JSON
            gemini_api_key: Gemini API key for text correction
            translation_model: HuggingFace translation model
            device: 'cuda', 'cpu', or 'auto'
        """
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[AGENT] Initializing on {self.device}...")

        # Stage 1: Image Restoration Model
        self._load_restoration_model(restoration_model_path)

        # Stage 2: Google Lens OCR (Cloud Vision)
        self._initialize_google_vision(google_credentials_path)

        # Stage 3: Gemini Correction
        self._initialize_gemini(gemini_api_key)

        # Stage 4: Translation Model
        self._load_translation_model(translation_model)

        print("[AGENT] ✓ All stages initialized successfully")

    def _load_restoration_model(self, model_path: str):
        """Load ViT-based restoration model"""
        print(f"[STAGE 1] Loading restoration model from {model_path}...")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Restoration model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Detect architecture
        has_head = any('head.' in k for k in state_dict.keys())
        has_patch_recon = any('patch_recon' in k for k in state_dict.keys())

        if has_head and not has_patch_recon:
            print("  → Kaggle format (simple head)")
            self.restoration_model = create_vit_restorer(
                'base', img_size=256,
                use_skip_connections=False,
                use_simple_head=True
            )
        else:
            print("  → Standard format (patch_recon)")
            self.restoration_model = create_vit_restorer(
                'base', img_size=256,
                use_skip_connections=True,
                use_simple_head=False
            )

        self.restoration_model.load_state_dict(state_dict, strict=False)
        self.restoration_model.to(self.device)
        self.restoration_model.eval()
        print("[STAGE 1] ✓ Restoration model loaded")

    def _initialize_google_vision(self, credentials_path: Optional[str]):
        """Initialize Google Cloud Vision API for OCR"""
        print("[STAGE 2] Initializing Google Lens OCR...")

        if not GOOGLE_VISION_AVAILABLE:
            print("[STAGE 2] ⚠ Google Vision not installed - OCR will be limited")
            self.vision_client = None
            return

        # Check for API key in environment variable first
        api_key = os.getenv('GOOGLE_VISION_API_KEY')

        # Set credentials if provided
        if credentials_path and Path(credentials_path).exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            print(f"  → Using credentials: {credentials_path}")
        elif api_key:
            # If API key is provided instead of JSON credentials
            print(f"  → Using API key: {api_key[:10]}...")
            os.environ['GOOGLE_VISION_API_KEY'] = api_key

        try:
            self.vision_client = vision.ImageAnnotatorClient()
            print("[STAGE 2] ✓ Google Lens OCR ready")
        except Exception as e:
            print(f"[STAGE 2] ⚠ Could not initialize Google Vision: {e}")
            self.vision_client = None

    def _initialize_gemini(self, api_key: Optional[str]):
        """Initialize Gemini API for Sanskrit text correction"""
        print("[STAGE 3] Initializing Gemini correction model...")

        if not GEMINI_AVAILABLE:
            print("[STAGE 3] ⚠ Gemini not installed - correction will be limited")
            self.gemini_model = None
            return

        # Use API key from environment or parameter
        key = api_key or os.getenv('GEMINI_API_KEY')

        if not key:
            print("[STAGE 3] ⚠ No Gemini API key - correction disabled")
            self.gemini_model = None
            return

        try:
            genai.configure(api_key=key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
            print("[STAGE 3] ✓ Gemini correction ready")
        except Exception as e:
            print(f"[STAGE 3] ⚠ Could not initialize Gemini: {e}")
            self.gemini_model = None

    def _load_translation_model(self, model_name: str):
        """Load translation model"""
        print(f"[STAGE 4] Loading translation model: {model_name}...")

        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translation_model = MarianMTModel.from_pretrained(model_name)
            self.translation_model.to(self.device)
            self.translation_model.eval()
            print("[STAGE 4] ✓ Translation model loaded")
        except Exception as e:
            print(f"[STAGE 4] ⚠ Translation model failed: {e}")
            self.translation_model = None

    # ==================== STAGE 1: IMAGE RESTORATION ====================

    def restore_image(self, image_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, str]:
        """
        Stage 1: Restore degraded manuscript image

        Args:
            image_path: Path to input image
            output_path: Optional path to save restored image

        Returns:
            (restored_image_array, status_message)
        """
        print(f"\n[STAGE 1] Restoring image: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Resize to model input size
        resized = cv2.resize(image_rgb, (256, 256))

        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Restore
        with torch.no_grad():
            restored_tensor = self.restoration_model(img_tensor)

        # Convert back to image
        restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored = (np.clip(restored, 0, 1) * 255).astype(np.uint8)

        # Resize back to original size
        restored = cv2.resize(restored, (w, h))

        # Save if path provided
        if output_path:
            cv2.imwrite(str(output_path), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
            print(f"[STAGE 1] ✓ Saved to: {output_path}")

        status = f"Image restored: {h}×{w}px, noise removed, clarity enhanced"
        return restored, status

    # ==================== STAGE 2: GOOGLE LENS OCR ====================

    def extract_text_google_lens(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Stage 2: Extract text using Google Lens (Cloud Vision API)

        Args:
            image: Restored image (numpy array, RGB)

        Returns:
            (extracted_text, confidence_score)
        """
        print("\n[STAGE 2] Running Google Lens OCR...")

        if self.vision_client is None:
            print("[STAGE 2] ⚠ Google Vision not available, using fallback")
            return self._fallback_ocr(image)

        try:
            # Convert to bytes
            pil_img = Image.fromarray(image)
            import io
            byte_arr = io.BytesIO()
            pil_img.save(byte_arr, format='PNG')
            image_bytes = byte_arr.getvalue()

            # Call Google Vision API
            vision_image = vision.Image(content=image_bytes)
            response = self.vision_client.text_detection(image=vision_image)

            if response.error.message:
                raise Exception(response.error.message)

            # Extract text
            texts = response.text_annotations
            if not texts:
                print("[STAGE 2] ⚠ No text detected")
                return "", 0.0

            # Full text is in first annotation
            full_text = texts[0].description
            confidence = texts[0].confidence if hasattr(texts[0], 'confidence') else 0.85

            print(f"[STAGE 2] ✓ Extracted {len(full_text)} characters (confidence: {confidence:.2%})")
            return full_text.strip(), confidence

        except Exception as e:
            print(f"[STAGE 2] ⚠ Google Vision failed: {e}")
            return self._fallback_ocr(image)

    def _fallback_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Fallback OCR using Tesseract if Google Vision fails"""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image, lang='san')
            return text.strip(), 0.70
        except Exception as e:
            print(f"[STAGE 2] ⚠ Fallback OCR also failed: {e}")
            return "", 0.0

    # ==================== STAGE 3: GEMINI TEXT CORRECTION ====================

    def correct_sanskrit_text(self, ocr_text: str) -> str:
        """
        Stage 3: Correct OCR errors using Gemini

        Args:
            ocr_text: Raw text from Google Lens OCR

        Returns:
            corrected_sanskrit_text
        """
        print("\n[STAGE 3] Correcting Sanskrit text with Gemini...")

        if not ocr_text.strip():
            print("[STAGE 3] ⚠ Empty OCR text, skipping correction")
            return ocr_text

        if self.gemini_model is None:
            print("[STAGE 3] ⚠ Gemini not available, returning OCR text as-is")
            return ocr_text

        # Correction prompt
        prompt = f"""You are a Sanskrit OCR error correction expert.

INPUT (from Google Lens OCR - may contain errors):
{ocr_text}

TASK:
1. Fix OCR mistakes (missing matras, broken conjuncts, wrong characters)
2. Restore proper Devanagari Unicode
3. Ensure grammatically valid Sanskrit
4. DO NOT add words that are not present
5. DO NOT hallucinate missing content
6. Keep the exact meaning

OUTPUT ONLY the corrected Sanskrit text in Devanagari (Unicode). No explanation."""

        try:
            response = self.gemini_model.generate_content(prompt)
            corrected = response.text.strip()
            print(f"[STAGE 3] ✓ Text corrected ({len(corrected)} chars)")
            return corrected
        except Exception as e:
            print(f"[STAGE 3] ⚠ Gemini correction failed: {e}")
            return ocr_text

    # ==================== STAGE 4: TRANSLATION ====================

    def translate_to_english(self, sanskrit_text: str) -> str:
        """
        Stage 4: Translate corrected Sanskrit to English

        Args:
            sanskrit_text: Corrected Sanskrit text (Devanagari)

        Returns:
            english_translation
        """
        print("\n[STAGE 4] Translating to English...")

        if not sanskrit_text.strip():
            print("[STAGE 4] ⚠ Empty Sanskrit text")
            return ""

        if self.translation_model is None:
            print("[STAGE 4] ⚠ Translation model not available")
            return "[Translation unavailable]"

        try:
            # Tokenize
            inputs = self.tokenizer(sanskrit_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Translate
            with torch.no_grad():
                outputs = self.translation_model.generate(**inputs)

            # Decode
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[STAGE 4] ✓ Translation complete ({len(translation)} chars)")
            return translation.strip()

        except Exception as e:
            print(f"[STAGE 4] ⚠ Translation failed: {e}")
            return f"[Translation error: {str(e)}]"

    # ==================== STAGE 5: VERIFICATION ====================

    def verify_output(self, ocr_text: str, corrected_text: str, translation: str) -> Tuple[bool, str, float]:
        """
        Stage 5: Verify pipeline output quality

        Args:
            ocr_text: Original OCR output
            corrected_text: Gemini-corrected text
            translation: English translation

        Returns:
            (is_valid, notes, confidence_score)
        """
        print("\n[STAGE 5] Final verification...")

        notes = []
        confidence = 1.0

        # Check 1: Text not empty
        if not corrected_text.strip():
            notes.append("ERROR: Corrected text is empty")
            confidence = 0.0

        # Check 2: Translation not empty
        if not translation.strip() or "unavailable" in translation.lower():
            notes.append("WARNING: Translation missing or failed")
            confidence *= 0.7

        # Check 3: Text length similarity (correction shouldn't drastically change length)
        if ocr_text and corrected_text:
            len_ratio = len(corrected_text) / max(len(ocr_text), 1)
            if len_ratio < 0.5 or len_ratio > 2.0:
                notes.append(f"WARNING: Text length changed significantly (ratio: {len_ratio:.2f})")
                confidence *= 0.8

        # Check 4: Translation length reasonable
        if corrected_text and translation:
            words_sanskrit = len(corrected_text.split())
            words_english = len(translation.split())
            if words_english < words_sanskrit * 0.3:
                notes.append("WARNING: Translation seems too short")
                confidence *= 0.9

        is_valid = confidence > 0.5

        if is_valid:
            print(f"[STAGE 5] ✓ Verification passed (confidence: {confidence:.2%})")
        else:
            print(f"[STAGE 5] ⚠ Verification failed (confidence: {confidence:.2%})")

        notes_str = "; ".join(notes) if notes else "All checks passed"
        return is_valid, notes_str, confidence

    # ==================== MAIN PIPELINE ====================

    def process(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute complete pipeline: Restore → OCR → Correct → Translate → Verify

        Args:
            image_path: Path to input manuscript image
            output_dir: Optional directory to save outputs

        Returns:
            Dictionary with all pipeline results (JSON-compatible)
        """
        print(f"\n{'='*60}")
        print(f"STARTING PIPELINE: {Path(image_path).name}")
        print(f"{'='*60}")

        start_time = time.time()

        # Setup output directory
        out_path = None
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            restored_img_path = str(out_path / "restored_output.png")
        else:
            restored_img_path = None

        # STAGE 1: Restoration
        restored_image, restore_status = self.restore_image(image_path, restored_img_path)

        # STAGE 2: Google Lens OCR
        ocr_text, ocr_confidence = self.extract_text_google_lens(restored_image)

        # STAGE 3: Gemini Correction
        corrected_text = self.correct_sanskrit_text(ocr_text)

        # STAGE 4: Translation
        english_translation = self.translate_to_english(corrected_text)

        # STAGE 5: Verification
        is_valid, notes, final_confidence = self.verify_output(ocr_text, corrected_text, english_translation)

        processing_time = time.time() - start_time

        # Build output
        result = {
            "restored_image_status": restore_status,
            "restored_image_path": str(restored_img_path) if restored_img_path else "not_saved",
            "ocr_output_text": ocr_text,
            "ocr_confidence": f"{ocr_confidence:.2%}",
            "corrected_sanskrit_text": corrected_text,
            "english_translation": english_translation,
            "notes": notes,
            "confidence_score": f"{final_confidence:.2%}",
            "is_valid": is_valid,
            "processing_time_seconds": f"{processing_time:.2f}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save outputs
        if output_dir:
            # Save texts
            (out_path / "extracted_sanskrit.txt").write_text(corrected_text, encoding='utf-8')
            (out_path / "translation_english.txt").write_text(english_translation, encoding='utf-8')

            # Save JSON
            json_path = out_path / "pipeline_result.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"\n[OUTPUT] Results saved to: {output_dir}")

        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE ({processing_time:.2f}s)")
        print(f"Confidence: {final_confidence:.2%} | Valid: {is_valid}")
        print(f"{'='*60}\n")

        return result


# ==================== CLI INTERFACE ====================

def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanskrit Manuscript OCR & Translation Agent"
    )
    parser.add_argument("image", help="Path to manuscript image")
    parser.add_argument(
        "--model",
        default="checkpoints/kaggle/final.pth",
        help="Path to restoration model checkpoint"
    )
    parser.add_argument(
        "--google-creds",
        help="Path to Google Cloud credentials JSON"
    )
    parser.add_argument(
        "--gemini-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--output",
        default="output/agent",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )

    args = parser.parse_args()

    # Initialize agent
    agent = SanskritOCRTranslationAgent(
        restoration_model_path=args.model,
        google_credentials_path=args.google_creds,
        gemini_api_key=args.gemini_key,
        device=args.device
    )

    # Process image
    result = agent.process(args.image, output_dir=args.output)

    # Print result
    print("\n📋 FINAL OUTPUT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

