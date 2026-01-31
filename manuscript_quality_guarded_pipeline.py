"""
ManuscriptVision-QualityAgent — QUALITY-GUARDED PIPELINE
Complete pipeline with ViT restoration model and Gemini API
ENSURES RESTORATION NEVER DEGRADES IMAGE QUALITY
"""

import io
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from google import genai
from google.genai import types
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import ViT restoration model
from models.vit_restorer import create_vit_restorer

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAxTN1rfdyJQVE3ecy5d8Zqkl5I431nBh0")
DEFAULT_MODEL = "gemini-2.0-flash-exp"
VIT_CHECKPOINT = "checkpoints/kaggle/final_converted.pth"  # Default to kaggle converted checkpoint

# QUALITY-GUARDED AGENT PROMPT
QUALITY_GUARDED_AGENT_PROMPT = """You are **ManuscriptVision-QualityAgent**, an expert agent for Sanskrit manuscript processing using **API-based vision and language models only**.

Your **primary responsibility** is to ensure that **image restoration NEVER degrades readability**. You are analyzing an image that has ALREADY been quality-checked.

## **YOUR TASKS (STRICT ORDER)**

### **TASK 1 — ANALYZE IMAGE QUALITY**
Analyze the provided Sanskrit manuscript image for:
* Text sharpness and clarity
* Character legibility
* Contrast between text and background
* Background uniformity
* Overall readability

Provide a quality assessment score (0.0 to 1.0) and description.

### **TASK 2 — OCR EXTRACTION**
Extract Sanskrit text in **pure Unicode Devanagari**:
* Preserve matras, ligatures (संयुक्ताक्षर), anusvāra, visarga, virāma
* Extract exactly what you see in the image
* Handle damaged or unclear characters with [?]

### **TASK 3 — OCR TEXT CORRECTION**
Correct OCR errors only:
* Fix broken characters, missing matras, split ligatures
* Normalize to valid Sanskrit grammar and Unicode Devanagari
* ❌ Do NOT invent missing words or lines
* ❌ Do NOT change meaning
* Keep [?] markers for truly unreadable text

### **TASK 4 — MULTILINGUAL TRANSLATION**
Translate the corrected Sanskrit text into:
* **English** (accurate, literal translation)
* **Hindi** (अर्थ सुरक्षित रखते हुए)
* **Kannada** (ಅರ್ಥವನ್ನು ಕಾಪಾಡುವಂತೆ)

Rules:
* Preserve original meaning
* No poetic rewriting
* No hallucination
* Mark uncertain sections

### **TASK 5 — CONFIDENCE VERIFICATION**
* Ensure OCR text matches visible image
* Ensure translations align with corrected Sanskrit
* Generate overall confidence score (0.0 to 1.0)

## **OUTPUT FORMAT (STRICT — JSON ONLY)**

Respond ONLY with valid JSON in this exact format:
```json
{
  "image_quality_assessment": {
    "score": 0.85,
    "description": "detailed quality assessment"
  },
  "ocr_extracted_text": "raw OCR text in Devanagari",
  "corrected_sanskrit_text": "corrected Sanskrit text in Devanagari",
  "english_translation": "accurate English translation",
  "hindi_translation": "Hindi translation in Devanagari",
  "kannada_translation": "Kannada translation in Kannada script",
  "confidence_score": 0.85,
  "processing_notes": "any relevant notes about quality or issues"
}
```

**CRITICAL**: Output ONLY the JSON object. No other text before or after.
"""


class ImageQualityAnalyzer:
    """Analyzes and compares image quality to ensure restoration doesn't degrade"""

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range (typical values are 0-1000+)
        return min(laplacian_var / 500.0, 1.0)

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate RMS contrast"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        return gray.std() / 255.0

    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Calculate mean brightness (normalized to 0-1)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        return gray.mean() / 255.0

    @staticmethod
    def calculate_text_clarity(image: np.ndarray) -> float:
        """Estimate text clarity using edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Normalize (typical good values are 0.05-0.15)
        return min(edge_density / 0.15, 1.0)

    @staticmethod
    def calculate_overall_quality(image: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive quality metrics"""
        metrics = {
            'sharpness': ImageQualityAnalyzer.calculate_sharpness(image),
            'contrast': ImageQualityAnalyzer.calculate_contrast(image),
            'brightness': ImageQualityAnalyzer.calculate_brightness(image),
            'text_clarity': ImageQualityAnalyzer.calculate_text_clarity(image),
        }

        # Overall score (weighted average)
        metrics['overall'] = (
            metrics['sharpness'] * 0.35 +
            metrics['contrast'] * 0.25 +
            metrics['text_clarity'] * 0.30 +
            (1.0 - abs(metrics['brightness'] - 0.7)) * 0.10  # Prefer brightness around 0.7
        )

        return metrics

    @staticmethod
    def compare_images(original: np.ndarray, restored: np.ndarray) -> Dict[str, Any]:
        """
        Compare original and restored images
        Returns decision on which image to use
        """
        # Ensure same size
        if original.shape != restored.shape:
            restored = cv2.resize(restored, (original.shape[1], original.shape[0]))

        # Convert to grayscale for comparison
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            rest_gray = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original
            rest_gray = restored

        # Calculate quality metrics
        orig_metrics = ImageQualityAnalyzer.calculate_overall_quality(original)
        rest_metrics = ImageQualityAnalyzer.calculate_overall_quality(restored)

        # Calculate SSIM (structural similarity)
        ssim_score = ssim(orig_gray, rest_gray, data_range=255)

        # Calculate PSNR (peak signal-to-noise ratio)
        psnr_score = psnr(orig_gray, rest_gray, data_range=255)

        # Decision logic
        improvement = rest_metrics['overall'] - orig_metrics['overall']

        # Quality gate thresholds
        MIN_IMPROVEMENT = 0.05  # Restoration must improve by at least 5%
        MIN_SSIM = 0.70  # Restored image must be at least 70% similar

        use_restored = False
        reason = ""

        if improvement < MIN_IMPROVEMENT:
            reason = f"Insufficient improvement ({improvement:.3f} < {MIN_IMPROVEMENT})"
        elif ssim_score < MIN_SSIM:
            reason = f"Structural similarity too low ({ssim_score:.3f} < {MIN_SSIM})"
        elif rest_metrics['sharpness'] < orig_metrics['sharpness'] * 0.9:
            reason = "Restoration reduced sharpness"
        elif rest_metrics['contrast'] < orig_metrics['contrast'] * 0.9:
            reason = "Restoration reduced contrast"
        else:
            use_restored = True
            reason = f"Quality improved by {improvement:.3f}"

        return {
            'use_restored': use_restored,
            'reason': reason,
            'original_quality': orig_metrics,
            'restored_quality': rest_metrics,
            'improvement': improvement,
            'ssim': ssim_score,
            'psnr': psnr_score,
        }


class ManuscriptQualityGuardedPipeline:
    """
    Complete quality-guarded pipeline for Sanskrit manuscript processing
    Ensures restoration never degrades quality
    """

    def __init__(self, api_key: str = None, vit_checkpoint: str = VIT_CHECKPOINT):
        """Initialize the pipeline with Gemini API and ViT model"""
        self.api_key = api_key or GEMINI_API_KEY
        self.gemini_client = genai.Client(api_key=self.api_key)
        self.model_name = DEFAULT_MODEL

        # Initialize ViT restoration model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vit_model = None
        self.vit_checkpoint = vit_checkpoint

        # Load ViT model if checkpoint exists
        if Path(vit_checkpoint).exists():
            self._load_vit_model()
        else:
            print(f"⚠️  ViT checkpoint not found at {vit_checkpoint}")
            print("⚠️  Will use fallback PIL-based restoration")

        self.quality_analyzer = ImageQualityAnalyzer()

    def _load_vit_model(self):
        """Load the ViT restoration model"""
        try:
            print(f"Loading ViT restoration model from {self.vit_checkpoint}...")
            self.vit_model = create_vit_restorer('base', img_size=256)
            checkpoint = torch.load(self.vit_checkpoint, map_location=self.device, weights_only=False)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.vit_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.vit_model.load_state_dict(checkpoint['state_dict'])
            else:
                self.vit_model.load_state_dict(checkpoint)

            self.vit_model.to(self.device)
            self.vit_model.eval()
            print(f"✓ ViT model loaded successfully on {self.device}")
        except Exception as e:
            print(f"⚠️  Failed to load ViT model: {e}")
            print("⚠️  Will use fallback PIL-based restoration")
            self.vit_model = None

    def restore_with_vit(self, image: Image.Image) -> Image.Image:
        """Restore image using ViT model"""
        # Convert PIL to numpy
        img_np = np.array(image)
        if img_np.shape[2] == 4:  # RGBA
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        original_h, original_w = img_np.shape[:2]

        # Resize to model input size
        img_resized = cv2.resize(img_np, (256, 256))

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Restore
        with torch.no_grad():
            restored_tensor = self.vit_model(img_tensor)

        # Convert back to numpy
        restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored = (restored * 255).clip(0, 255).astype(np.uint8)

        # Resize back to original dimensions
        restored = cv2.resize(restored, (original_w, original_h))

        return Image.fromarray(restored)

    def restore_with_pil_fallback(self, image: Image.Image) -> Image.Image:
        """Fallback PIL-based restoration (conservative)"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Conservative enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)  # Mild contrast boost

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)  # Mild sharpening

        # Denoise
        image = image.filter(ImageFilter.MedianFilter(size=3))

        return image

    def restore_image_with_quality_gate(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        STEP 1-3: Restore image with quality gate
        Returns: (best_image, quality_report)
        """
        print("\n" + "="*60)
        print("STEP 1: ORIGINAL IMAGE ANALYSIS")
        print("="*60)

        # Convert to numpy for analysis
        original_np = np.array(image)

        # Analyze original
        original_metrics = self.quality_analyzer.calculate_overall_quality(original_np)
        print(f"\n📊 Original Image Quality:")
        print(f"   • Sharpness:    {original_metrics['sharpness']:.3f}")
        print(f"   • Contrast:     {original_metrics['contrast']:.3f}")
        print(f"   • Text Clarity: {original_metrics['text_clarity']:.3f}")
        print(f"   • Overall:      {original_metrics['overall']:.3f}")

        print("\n" + "="*60)
        print("STEP 2: ATTEMPTING RESTORATION")
        print("="*60)

        # Attempt restoration
        if self.vit_model is not None:
            print("Using ViT restoration model...")
            restored_image = self.restore_with_vit(image)
        else:
            print("Using PIL-based fallback restoration...")
            restored_image = self.restore_with_pil_fallback(image)

        restored_np = np.array(restored_image)

        print("\n" + "="*60)
        print("STEP 3: QUALITY COMPARISON & GATE")
        print("="*60)

        # Compare quality
        comparison = self.quality_analyzer.compare_images(original_np, restored_np)

        print(f"\n📊 Restored Image Quality:")
        print(f"   • Sharpness:    {comparison['restored_quality']['sharpness']:.3f}")
        print(f"   • Contrast:     {comparison['restored_quality']['contrast']:.3f}")
        print(f"   • Text Clarity: {comparison['restored_quality']['text_clarity']:.3f}")
        print(f"   • Overall:      {comparison['restored_quality']['overall']:.3f}")

        print(f"\n🔍 Comparison Metrics:")
        print(f"   • Improvement:  {comparison['improvement']:+.3f}")
        print(f"   • SSIM:         {comparison['ssim']:.3f}")
        print(f"   • PSNR:         {comparison['psnr']:.2f} dB")

        print(f"\n🚦 QUALITY GATE DECISION:")

        if comparison['use_restored']:
            print(f"   ✅ USING RESTORED IMAGE")
            print(f"   Reason: {comparison['reason']}")
            selected_image = restored_image
            image_used = "restored"
        else:
            print(f"   ⚠️  USING ORIGINAL IMAGE")
            print(f"   Reason: {comparison['reason']}")
            selected_image = image
            image_used = "original"

        quality_report = {
            'image_used': image_used,
            'restoration_attempted': True,
            'restoration_applied': comparison['use_restored'],
            'decision_reason': comparison['reason'],
            'original_metrics': original_metrics,
            'restored_metrics': comparison['restored_quality'] if comparison['use_restored'] else None,
            'improvement': comparison['improvement'],
            'ssim': comparison['ssim'],
            'psnr': comparison['psnr'],
        }

        return selected_image, quality_report

    def extract_and_process_with_gemini(self, image: Image.Image) -> Dict[str, Any]:
        """
        STEPS 4-7: Use Gemini Vision for OCR, correction, translation, verification
        """
        print("\n" + "="*60)
        print("STEP 4-7: GEMINI API PROCESSING")
        print("="*60)

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Prepare content with image and agent prompt
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=QUALITY_GUARDED_AGENT_PROMPT),
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/png"
                    ),
                ],
            ),
        ]

        # Configure generation
        generate_content_config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
            response_mime_type="application/json"
        )

        try:
            print("\n🔄 Calling Gemini Vision API...")
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config
            )

            # Parse JSON response
            result_text = response.text.strip()

            # Clean up any markdown formatting
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            result = json.loads(result_text)

            print("✅ Gemini API processing complete")
            return result

        except Exception as e:
            print(f"❌ Error calling Gemini API: {e}")
            return {
                'error': str(e),
                'image_quality_assessment': {'score': 0.0, 'description': 'Error occurred'},
                'ocr_extracted_text': '',
                'corrected_sanskrit_text': '',
                'english_translation': '',
                'hindi_translation': '',
                'kannada_translation': '',
                'confidence_score': 0.0,
                'processing_notes': f'Error: {e}'
            }

    def process_manuscript(self, image: Image.Image) -> Dict[str, Any]:
        """
        Execute complete quality-guarded pipeline
        """
        print("\n" + "🔰"*30)
        print("🔰 QUALITY-GUARDED MANUSCRIPT PIPELINE")
        print("🔰"*30)

        # Step 1-3: Restore with quality gate
        selected_image, quality_report = self.restore_image_with_quality_gate(image)

        # Step 4-7: OCR, correction, translation
        gemini_result = self.extract_and_process_with_gemini(selected_image)

        # Combine results
        final_result = {
            **quality_report,
            **gemini_result,
        }

        # Add images for UI display
        final_result['original_image'] = image
        final_result['selected_image'] = selected_image

        print("\n" + "✅"*30)
        print("✅ PIPELINE COMPLETE")
        print("✅"*30 + "\n")

        return final_result


# Convenience function
def process_manuscript_file(image_path: str, api_key: str = None) -> Dict[str, Any]:
    """
    Process a manuscript image file
    """
    image = Image.open(image_path)
    pipeline = ManuscriptQualityGuardedPipeline(api_key=api_key)
    return pipeline.process_manuscript(image)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python manuscript_quality_guarded_pipeline.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = process_manuscript_file(image_path)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

