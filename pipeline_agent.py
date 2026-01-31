#!/usr/bin/env python3
"""
AI Pipeline Agent for Historic Sanskrit Manuscript Processing
3-Stage Workflow: RESTORE → OCR → TRANSLATE

Requirements:
- Stage 1: User's trained restoration model
- Stage 2: HuggingFace OCR model (TrOCR/DETR)
- Stage 3: HuggingFace Translation model (mBART/Helsinki)
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json
import argparse
from datetime import datetime
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models.vit_restorer import create_vit_restorer


class SanskritManuscriptAgent:
    """
    AI Agent for processing historic Sanskrit manuscripts
    Executes: Restoration → OCR → Translation
    """

    def __init__(
        self,
        restoration_model_path,
        ocr_model='microsoft/trocr-base-handwritten',
        translation_model='Helsinki-NLP/opus-mt-sa-en',
        device='auto'
    ):
        """
        Initialize the 3-stage pipeline

        Args:
            restoration_model_path: Path to trained ViT restoration model
            ocr_model: HuggingFace OCR model name
            translation_model: HuggingFace translation model name
            device: 'cuda', 'cpu', or 'auto'
        """
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"🤖 Initializing AI Pipeline Agent on {self.device}...")
        print("=" * 70)

        # Stage 1: Restoration Model
        self._init_restoration_model(restoration_model_path)

        # Stage 2: OCR Model
        self._init_ocr_model(ocr_model)

        # Stage 3: Translation Model
        self._init_translation_model(translation_model)

        print("=" * 70)
        print("✅ Pipeline Agent Ready!\n")

    def _init_restoration_model(self, model_path):
        """Initialize Stage 1: Image Restoration"""
        print("\n1️⃣  STAGE 1: Image Restoration")
        print("   Loading user's trained restoration model...")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Detect model format
        has_head = any('head.' in k for k in state_dict.keys())
        has_skip = any('skip_fusion' in k for k in state_dict.keys())

        # Create model with appropriate architecture
        self.restoration_model = create_vit_restorer(
            'base',
            img_size=256,
            use_skip_connections=has_skip,
            use_simple_head=has_head
        )

        # Load weights
        self.restoration_model.load_state_dict(state_dict, strict=False)
        self.restoration_model.to(self.device)
        self.restoration_model.eval()

        print(f"   ✅ Restoration model loaded ({model_path})")

    def _init_ocr_model(self, model_name):
        """Initialize Stage 2: OCR with HuggingFace"""
        print("\n2️⃣  STAGE 2: OCR Text Extraction")
        print(f"   Loading HuggingFace model: {model_name}")

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self.ocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.ocr_model.to(self.device)
            self.ocr_model.eval()

            self.ocr_type = 'trocr'
            print(f"   ✅ TrOCR model loaded")
        except Exception as e:
            print(f"   ⚠️  TrOCR failed: {e}")
            print("   📌 Falling back to Tesseract (Devanagari)...")

            # Fallback to Tesseract
            try:
                import pytesseract
                self.ocr_type = 'tesseract'
                self.tesseract = pytesseract
                print("   ✅ Tesseract OCR loaded")
            except ImportError:
                print("   ❌ Tesseract not available")
                self.ocr_type = None

    def _init_translation_model(self, model_name):
        """Initialize Stage 3: Translation with HuggingFace"""
        print("\n3️⃣  STAGE 3: Translation to English")
        print(f"   Loading HuggingFace model: {model_name}")

        try:
            from transformers import MarianMTModel, MarianTokenizer

            self.translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translation_model = MarianMTModel.from_pretrained(model_name)
            self.translation_model.to(self.device)
            self.translation_model.eval()

            self.translation_type = 'helsinki'
            print(f"   ✅ Helsinki translation model loaded")
        except Exception as e:
            print(f"   ⚠️  Helsinki model failed: {e}")
            print("   📌 Falling back to Google Translate...")

            # Fallback to Google Translate
            try:
                from googletrans import Translator
                self.google_translator = Translator()
                self.translation_type = 'google'
                print("   ✅ Google Translate loaded")
            except ImportError:
                print("   ❌ Google Translate not available")
                self.translation_type = None

    def stage1_restore_image(self, image_path, output_dir):
        """
        Stage 1: Image Restoration
        Input: Degraded manuscript image
        Output: Enhanced/restored image
        """
        print("\n" + "=" * 70)
        print("🔧 STAGE 1: IMAGE RESTORATION")
        print("=" * 70)

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            img = np.array(Image.open(image_path).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_h, original_w = img.shape[:2]
        print(f"   Input: {image_path}")
        print(f"   Size: {original_w}x{original_h}")

        # Normalize and resize
        img_normalized = img.astype(np.float32) / 255.0
        img_resized = cv2.resize(img_normalized, (256, 256))

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.to(self.device)

        # Restore
        print("   🔄 Processing through ViT restoration model...")
        with torch.no_grad():
            restored_tensor = self.restoration_model(img_tensor)

        # Convert back to numpy
        restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored = np.clip(restored, 0, 1)

        # Resize back to original size
        restored_full = cv2.resize(restored, (original_w, original_h))
        restored_uint8 = (restored_full * 255).astype(np.uint8)

        # Save restored image
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        restored_path = output_path / "restored_output.png"
        cv2.imwrite(str(restored_path), cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2BGR))

        print(f"   ✅ Restored image saved: {restored_path}")
        print(f"   Output size: {original_w}x{original_h}")

        return {
            'restored_image_path': str(restored_path),
            'restored_array': restored_uint8,
            'original_size': (original_w, original_h)
        }

    def stage2_extract_text(self, restored_image, output_dir):
        """
        Stage 2: OCR Text Extraction
        Input: Restored image
        Output: Sanskrit text in Devanagari (UTF-8)
        """
        print("\n" + "=" * 70)
        print("📝 STAGE 2: OCR TEXT EXTRACTION")
        print("=" * 70)

        # Preprocess for OCR
        print("   🔄 Preprocessing image for OCR...")
        gray = cv2.cvtColor(restored_image, cv2.COLOR_RGB2GRAY)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)

        # Extract text
        sanskrit_text = ""

        if self.ocr_type == 'trocr':
            print(f"   🤖 Running TrOCR model...")
            try:
                pil_img = Image.fromarray(restored_image)
                pixel_values = self.ocr_processor(pil_img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)

                generated_ids = self.ocr_model.generate(pixel_values)
                sanskrit_text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(f"   ✅ TrOCR extraction complete")
            except Exception as e:
                print(f"   ⚠️  TrOCR failed: {e}")
                print("   📌 Falling back to Tesseract...")
                self.ocr_type = 'tesseract'

        if self.ocr_type == 'tesseract' or not sanskrit_text:
            print(f"   🔤 Running Tesseract OCR (Devanagari)...")
            try:
                import pytesseract

                # Try multiple configurations
                configs = [
                    '--oem 3 --psm 6 -l san',  # Sanskrit
                    '--oem 3 --psm 6 -l hin',  # Hindi (has Devanagari)
                    '--oem 3 --psm 4 -l san',  # Different page segmentation
                ]

                for i, config in enumerate(configs):
                    try:
                        text = pytesseract.image_to_string(denoised, config=config)
                        if text.strip():
                            sanskrit_text = text
                            print(f"   ✅ Tesseract extraction complete (config {i+1})")
                            break
                    except:
                        continue

                if not sanskrit_text:
                    # Last resort: default config
                    sanskrit_text = pytesseract.image_to_string(denoised)
                    print(f"   ⚠️  Using default Tesseract config")

            except Exception as e:
                print(f"   ❌ OCR failed: {e}")
                sanskrit_text = "[OCR extraction failed]"

        # Clean extracted text
        sanskrit_text = self._clean_text(sanskrit_text)

        # Save extracted text
        output_path = Path(output_dir)
        text_path = output_path / "extracted_sanskrit.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(sanskrit_text)

        word_count = len(sanskrit_text.split())
        print(f"   ✅ Sanskrit text saved: {text_path}")
        print(f"   Extracted: {word_count} words, {len(sanskrit_text)} characters")

        return {
            'sanskrit_text': sanskrit_text,
            'word_count': word_count,
            'text_path': str(text_path)
        }

    def stage3_translate(self, sanskrit_text, output_dir):
        """
        Stage 3: Translation to English
        Input: Sanskrit text
        Output: English translation
        """
        print("\n" + "=" * 70)
        print("🌍 STAGE 3: TRANSLATION TO ENGLISH")
        print("=" * 70)

        if not sanskrit_text or sanskrit_text == "[OCR extraction failed]":
            english_translation = "[No text to translate]"
            print("   ⚠️  No text available for translation")
        else:
            english_translation = ""

            if self.translation_type == 'helsinki':
                print(f"   🤖 Running Helsinki NLP translation...")
                try:
                    # Segment text (max 512 tokens per batch)
                    segments = self._segment_text(sanskrit_text, max_length=400)

                    translations = []
                    for seg in segments:
                        inputs = self.translation_tokenizer(seg, return_tensors="pt", padding=True)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            translated = self.translation_model.generate(**inputs)

                        trans_text = self.translation_tokenizer.decode(translated[0], skip_special_tokens=True)
                        translations.append(trans_text)

                    english_translation = " ".join(translations)
                    print(f"   ✅ Helsinki translation complete")
                except Exception as e:
                    print(f"   ⚠️  Helsinki failed: {e}")
                    print("   📌 Falling back to Google Translate...")
                    self.translation_type = 'google'

            if self.translation_type == 'google' or not english_translation:
                print(f"   🌐 Running Google Translate...")
                try:
                    from googletrans import Translator
                    translator = Translator()

                    # Translate in segments
                    segments = self._segment_text(sanskrit_text, max_length=500)
                    translations = []

                    for seg in segments:
                        result = translator.translate(seg, src='sa', dest='en')
                        translations.append(result.text)

                    english_translation = " ".join(translations)
                    print(f"   ✅ Google translation complete")
                except Exception as e:
                    print(f"   ❌ Translation failed: {e}")
                    english_translation = f"[Translation failed: {str(e)}]"

        # Clean translation
        english_translation = self._clean_translation(english_translation)

        # Save translation
        output_path = Path(output_dir)
        trans_path = output_path / "translation_english.txt"
        with open(trans_path, 'w', encoding='utf-8') as f:
            f.write(english_translation)

        print(f"   ✅ Translation saved: {trans_path}")
        print(f"   Length: {len(english_translation)} characters")

        return {
            'english_translation': english_translation,
            'translation_path': str(trans_path)
        }

    def _clean_text(self, text):
        """Clean OCR output text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove repeated characters (common OCR error)
        import re
        text = re.sub(r'(.)\1{3,}', r'\1', text)

        return text.strip()

    def _clean_translation(self, text):
        """Clean translation output"""
        # Capitalize sentences
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences]
        text = '. '.join(sentences)

        return text.strip()

    def _segment_text(self, text, max_length=500):
        """Segment text into chunks for translation"""
        words = text.split()
        segments = []
        current = []
        current_len = 0

        for word in words:
            if current_len + len(word) + 1 > max_length:
                if current:
                    segments.append(' '.join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1

        if current:
            segments.append(' '.join(current))

        return segments if segments else [text]

    def execute_pipeline(self, image_path, output_dir='output/pipeline_agent'):
        """
        Execute complete 3-stage pipeline

        Args:
            image_path: Path to manuscript image
            output_dir: Output directory

        Returns:
            dict: Final JSON output with all results
        """
        print("\n" + "=" * 70)
        print("🏛️  SANSKRIT MANUSCRIPT AI PIPELINE AGENT")
        print("=" * 70)
        print(f"Input: {image_path}")
        print(f"Output: {output_dir}")
        print(f"Device: {self.device}")

        start_time = datetime.now()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize results log
        log = []
        log.append(f"Pipeline started: {start_time}")

        # STAGE 1: Restoration
        try:
            stage1_result = self.stage1_restore_image(image_path, output_dir)
            log.append("Stage 1: SUCCESS - Image restored")
        except Exception as e:
            print(f"\n❌ Stage 1 failed: {e}")
            log.append(f"Stage 1: FAILED - {e}")
            raise

        # STAGE 2: OCR
        try:
            stage2_result = self.stage2_extract_text(stage1_result['restored_array'], output_dir)
            log.append(f"Stage 2: SUCCESS - {stage2_result['word_count']} words extracted")
        except Exception as e:
            print(f"\n❌ Stage 2 failed: {e}")
            log.append(f"Stage 2: FAILED - {e}")
            stage2_result = {'sanskrit_text': '[OCR failed]', 'word_count': 0}

        # STAGE 3: Translation
        try:
            stage3_result = self.stage3_translate(stage2_result['sanskrit_text'], output_dir)
            log.append("Stage 3: SUCCESS - Translation completed")
        except Exception as e:
            print(f"\n❌ Stage 3 failed: {e}")
            log.append(f"Stage 3: FAILED - {e}")
            stage3_result = {'english_translation': '[Translation failed]'}

        # Create final JSON output
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        final_output = {
            "restored_image_path": stage1_result['restored_image_path'],
            "sanskrit_text": stage2_result['sanskrit_text'],
            "english_translation": stage3_result['english_translation'],
            "metadata": {
                "input_image": str(image_path),
                "output_directory": str(output_dir),
                "processing_time_seconds": processing_time,
                "device": self.device,
                "timestamp": end_time.isoformat(),
                "stages": {
                    "restoration": "completed",
                    "ocr": f"{stage2_result.get('word_count', 0)} words",
                    "translation": "completed"
                }
            }
        }

        # Save JSON output
        json_path = output_path / "pipeline_output.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        # Save processing log
        log.append(f"Pipeline completed: {end_time}")
        log.append(f"Total time: {processing_time:.2f}s")

        log_path = output_path / "pipeline.log"
        with open(log_path, 'w') as f:
            f.write('\n'.join(log))

        # Print summary
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"\n📁 Output Files:")
        print(f"   • {stage1_result['restored_image_path']}")
        print(f"   • {stage2_result.get('text_path', 'N/A')}")
        print(f"   • {stage3_result.get('translation_path', 'N/A')}")
        print(f"   • {json_path}")
        print(f"   • {log_path}")
        print(f"\n⏱️  Processing Time: {processing_time:.2f}s")
        print(f"📊 Words Extracted: {stage2_result.get('word_count', 0)}")
        print("=" * 70)

        return final_output


def main():
    parser = argparse.ArgumentParser(
        description='AI Pipeline Agent for Sanskrit Manuscript Processing'
    )
    parser.add_argument(
        '--image_path',
        required=True,
        help='Path to manuscript image'
    )
    parser.add_argument(
        '--restoration_model',
        default='checkpoints/kaggle/final.pth',
        help='Path to trained restoration model'
    )
    parser.add_argument(
        '--ocr_model',
        default='microsoft/trocr-base-handwritten',
        help='HuggingFace OCR model name'
    )
    parser.add_argument(
        '--translation_model',
        default='Helsinki-NLP/opus-mt-sa-en',
        help='HuggingFace translation model name'
    )
    parser.add_argument(
        '--output_dir',
        default='output/pipeline_agent',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Initialize agent
    agent = SanskritManuscriptAgent(
        restoration_model_path=args.restoration_model,
        ocr_model=args.ocr_model,
        translation_model=args.translation_model,
        device=args.device
    )

    # Execute pipeline
    result = agent.execute_pipeline(
        image_path=args.image_path,
        output_dir=args.output_dir
    )

    # Print final JSON
    print("\n📋 FINAL OUTPUT JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


if __name__ == '__main__':
    main()

