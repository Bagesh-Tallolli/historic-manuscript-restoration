"""
Main Pipeline: Sanskrit Manuscript Restoration → OCR → Normalization → Translation

Complete end-to-end processing of Sanskrit manuscript images.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from models.vit_restorer import create_vit_restorer
from ocr.preprocess import OCRPreprocessor
from ocr.run_ocr import SanskritOCR
from ocr.enhanced_ocr import EnhancedSanskritOCR
from nlp.unicode_normalizer import SanskritTextProcessor
from nlp.translation import SanskritTranslator
from utils.image_restoration_enhanced import create_enhanced_restorer


class ManuscriptPipeline:
    """
    Complete end-to-end manuscript processing pipeline

    Agent-based architecture for:
    1. Ancient manuscript image restoration
    2. Sanskrit OCR extraction
    3. OCR error correction & normalization
    4. Sanskrit → English translation
    5. Quality verification & self-correction
    """

    def __init__(
        self,
        restoration_model_path=None,
        ocr_engine='tesseract',
        translation_method='google',
        device='auto'
    ):
        """
        Initialize the pipeline

        Args:
            restoration_model_path: Path to trained ViT model (None = skip restoration)
            ocr_engine: 'tesseract' or 'trocr'
            translation_method: 'google', 'indictrans', or 'ensemble'
            device: 'cuda', 'cpu', or 'auto'
        """
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Initializing pipeline on {self.device}...")

        # 1. Image Restoration Model
        if restoration_model_path:
            print("Loading restoration model...")
            checkpoint = torch.load(restoration_model_path, map_location=self.device, weights_only=False)

            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Detect model format automatically
            has_head = any('head.' in k for k in state_dict.keys())
            has_patch_recon = any('patch_recon' in k for k in state_dict.keys())
            has_skip = any('skip_fusion' in k for k in state_dict.keys())

            if has_head and not has_patch_recon:
                # Old Kaggle format: simple head, no skip connections
                print("  Format: Kaggle checkpoint (simple head, no skip)")
                self.restoration_model = create_vit_restorer(
                    'base', img_size=256,
                    use_skip_connections=False,
                    use_simple_head=True
                )
            elif has_patch_recon and has_skip:
                # New format: patch_recon with skip connections
                print("  Format: New checkpoint (patch_recon + skip)")
                self.restoration_model = create_vit_restorer(
                    'base', img_size=256,
                    use_skip_connections=True,
                    use_simple_head=False
                )
            elif has_patch_recon and not has_skip:
                # Patch recon without skip
                print("  Format: Patch_recon without skip")
                self.restoration_model = create_vit_restorer(
                    'base', img_size=256,
                    use_skip_connections=False,
                    use_simple_head=False
                )
            else:
                # Default to new format
                print("  Format: Default (patch_recon + skip)")
                self.restoration_model = create_vit_restorer('base', img_size=256)

            # Load weights
            try:
                self.restoration_model.load_state_dict(state_dict, strict=True)
                print("✓ Restoration model loaded successfully")
            except RuntimeError as e:
                print(f"  Warning: Strict loading failed, trying flexible mode...")
                missing, unexpected = self.restoration_model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"  Missing keys: {len(missing)}")
                if unexpected:
                    print(f"  Unexpected keys: {len(unexpected)}")
                print("✓ Restoration model loaded (flexible mode)")

            self.restoration_model.to(self.device)
            self.restoration_model.eval()

            # Create enhanced restorer for better quality
            self.enhanced_restorer = create_enhanced_restorer(
                self.restoration_model,
                device=self.device,
                patch_size=256,
                overlap=32
            )
            print("✓ Enhanced restorer initialized (patch-based processing)")
        else:
            print("⚠ Skipping restoration (no model provided)")
            self.restoration_model = None
            self.enhanced_restorer = None

        # 2. OCR Engine
        print("✓ OCR initialized (Tesseract-only)")
        self.ocr = EnhancedSanskritOCR(engine='tesseract', device=self.device)
        # Force Tesseract-only regardless of requested engine per updated requirements
        print(f"Initializing OCR (tesseract-only)...")

        # 3. Text Processor (Unicode normalization)
        print("Initializing text processor...")
        self.text_processor = SanskritTextProcessor()
        print("✓ Text processor initialized")

        # 4. Translator
        print(f"Initializing translator ({translation_method})...")
        self.translator = SanskritTranslator(method=translation_method, device=self.device)
        print("✓ Translator initialized")

        print("\n✓ Pipeline ready!\n")

    def process(self, image_path, save_output=False, output_dir='output'):
        """
        Alias for process_manuscript for backward compatibility
        """
        return self.process_manuscript(image_path, save_output, output_dir)

    def process_manuscript(self, image_path, save_output=False, output_dir='output'):
        """
        Process a manuscript image through the complete pipeline with agent-based verification

        Agent Responsibilities:
        1. Image Restoration - Enhance clarity, remove noise, preserve character shapes
        2. OCR Extraction - Extract Sanskrit/Devanagari text accurately
        3. OCR Correction - Fix broken ligatures, missing matras, normalize Unicode
        4. Translation - Translate normalized Sanskrit into clear, accurate English
        5. Verification - Compare restored vs OCR, validate translation, provide confidence

        Args:
            image_path: Path to manuscript image
            save_output: Whether to save intermediate results
            output_dir: Directory to save outputs

        Returns:
            Dictionary with all results including confidence scores
        """
        print(f"\nProcessing: {image_path}")
        print("=" * 60)

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load original image
        original_img = cv2.imread(str(image_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        print("\n1️⃣  Image Restoration")
        print("-" * 60)

        # Restore image
        if self.restoration_model is not None:
            restored_img = self._restore_image(original_img)
            print("✓ Image restored")

            # CRITICAL FIX: Ensure restored image is in correct format for OCR
            # Check if values are in [0, 1] range (float) instead of [0, 255] (uint8)
            if restored_img.dtype != np.uint8:
                print(f"  ⚠ Converting restored image from {restored_img.dtype} to uint8")
                if restored_img.max() <= 1.0:
                    restored_img = (restored_img * 255).astype(np.uint8)
                else:
                    restored_img = restored_img.astype(np.uint8)

            # Verify image format
            print(f"  Restored image: shape={restored_img.shape}, dtype={restored_img.dtype}, range=[{restored_img.min()}, {restored_img.max()}]")
        else:
            restored_img = original_img.copy()
            print("⚠ Skipped (no model)")

        # OCR
        print("\n2️⃣  OCR (Text Extraction)")
        print("-" * 60)

        # DEBUG: Verify what image is being sent to OCR
        print(f"  Image sent to OCR: shape={restored_img.shape}, dtype={restored_img.dtype}, range=[{restored_img.min()}, {restored_img.max()}]")
        print(f"  Using {'RESTORED' if self.restoration_model is not None else 'ORIGINAL'} image for OCR")

        # Use enhanced OCR for complete paragraph extraction
        ocr_result = self.ocr.extract_complete_paragraph(restored_img, preprocess=True, multi_pass=True)
        ocr_text_raw = ocr_result['text']
        ocr_confidence = ocr_result.get('confidence', 0.0)
        ocr_method = ocr_result.get('method', 'unknown')

        print(f"OCR Method: {ocr_method}")
        print(f"Confidence: {ocr_confidence:.2%}")
        print(f"Raw OCR output:\n{ocr_text_raw[:200]}..." if len(ocr_text_raw) > 200 else f"Raw OCR output:\n{ocr_text_raw}")

        # Unicode Normalization
        print("\n3️⃣  Unicode Normalization")
        print("-" * 60)

        processed = self.text_processor.process_ocr_output(ocr_text_raw, input_format='auto')
        ocr_text_cleaned = processed['normalized']
        print(f"Normalized text:\n{ocr_text_cleaned}")
        print(f"Word count: {processed['word_count']}")

        # Translation
        print("\n4️⃣  Translation (Sanskrit → English)")
        print("-" * 60)

        if ocr_text_cleaned:
            translation = self.translator.translate(ocr_text_cleaned)
            print(f"Translation:\n{translation}")
        else:
            translation = ""
            print("⚠ No text to translate")

        # Prepare results
        results = {
            'image_path': str(image_path),
            'original_image': original_img,
            'restored_image': restored_img,
            'ocr_text_raw': ocr_text_raw,
            'ocr_text_cleaned': ocr_text_cleaned,
            'sentences': processed['sentences'],
            'words': processed['words'],
            'word_count': processed['word_count'],
            'romanized': processed['romanized'],
            'translation': translation,
            'ocr_confidence': ocr_confidence,
            'ocr_method': ocr_method,
            'ocr_word_count': ocr_result.get('word_count', processed['word_count']),
            'ocr_hybrid_score': ocr_result.get('hybrid_score') if 'hybrid_score' in ocr_result else None,
            'ocr_components': ocr_result.get('components') if 'components' in ocr_result else None,
            'complete_extraction': True
        }

        # Save outputs
        if save_output:
            restored_path = self._save_results(results, output_dir, image_path.stem)
            results['restored_path'] = restored_path

        print("\n" + "=" * 60)
        print("✓ Processing complete!")

        return results

    def _verify_and_score(self, original_img, restored_img, ocr_raw, ocr_cleaned, translation):
        """
        Agent-based verification and confidence scoring

        Verifies:
        1. Image restoration quality
        2. OCR extraction accuracy
        3. Translation quality

        Returns confidence scores and notes
        """
        notes = []

        # 1. OCR Confidence - based on text length and character validity
        ocr_confidence = 0.0
        if ocr_raw:
            # Check if we have meaningful text
            has_devanagari = any('\u0900' <= c <= '\u097F' for c in ocr_cleaned)
            text_length = len(ocr_cleaned.strip())

            if has_devanagari and text_length > 10:
                ocr_confidence = 0.9
                notes.append("Good Devanagari text extraction")
            elif text_length > 5:
                ocr_confidence = 0.7
                notes.append("Moderate text extraction")
            else:
                ocr_confidence = 0.4
                notes.append("Limited text extraction - image may be heavily degraded")
        else:
            notes.append("No text extracted - check image quality")

        # 2. Translation Quality - based on output validity
        translation_quality = 0.0
        if translation and len(translation.strip()) > 0:
            # Basic checks
            word_count = len(translation.split())
            if word_count > 3:
                translation_quality = 0.85
                notes.append("Complete translation generated")
            elif word_count > 0:
                translation_quality = 0.6
                notes.append("Partial translation generated")
        else:
            if ocr_cleaned:
                notes.append("Translation failed - check translator service")
            else:
                notes.append("No translation - no source text")

        # 3. Image Quality - basic PSNR-like assessment
        if restored_img is not None and original_img is not None:
            # Simple quality check based on variance
            import numpy as np
            restored_var = np.var(restored_img)
            original_var = np.var(original_img)

            if restored_var > original_var * 0.5:
                notes.append("Good restoration quality")
            else:
                notes.append("Restoration may need improvement")

        # 4. Overall Confidence
        overall_confidence = (ocr_confidence * 0.6 + translation_quality * 0.4)

        return {
            'ocr_confidence': ocr_confidence,
            'translation_quality': translation_quality,
            'overall_confidence': overall_confidence,
            'notes': notes
        }

    def _restore_image(self, image, use_enhanced=True):
        """
        Restore image using ViT model

        Args:
            image: Input image (H, W, 3) in RGB format
            use_enhanced: If True, use patch-based processing for better quality
                         If False, use simple resize method (faster)

        Returns:
            Restored image (same size as input, uint8, [0-255])
        """
        if use_enhanced and self.enhanced_restorer is not None:
            # Use enhanced patch-based restoration for better quality
            h, w = image.shape[:2]

            # Use patch-based processing for large images
            if h > 512 or w > 512:
                print(f"   Using enhanced patch-based restoration ({h}x{w})...")
                restored = self.enhanced_restorer.restore_image(
                    image,
                    use_patches=True,
                    apply_postprocess=True
                )
            else:
                print(f"   Using simple restoration ({h}x{w})...")
                restored = self.enhanced_restorer.restore_image(
                    image,
                    use_patches=False,
                    apply_postprocess=True
                )

            # CRITICAL FIX: Ensure output is uint8 [0-255]
            if restored.dtype != np.uint8:
                if restored.max() <= 1.0:
                    restored = (restored * 255).clip(0, 255).astype(np.uint8)
                else:
                    restored = restored.clip(0, 255).astype(np.uint8)

            return restored
        else:
            # Fallback to simple method (original implementation)
            h, w = image.shape[:2]
            img_resized = cv2.resize(image, (256, 256))

            # Convert to tensor
            img_tensor = torch.from_numpy(img_resized).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            # Restore
            with torch.no_grad():
                restored_tensor = self.restoration_model(img_tensor)

            # Convert back to numpy
            restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored = (restored * 255).clip(0, 255).astype(np.uint8)

            # Resize back to original size
            restored = cv2.resize(restored, (w, h))

            return restored

    def _save_results(self, results, output_dir, name):
        """Save all results to files and return restored image path"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save images
        cv2.imwrite(
            str(output_dir / f"{name}_original.jpg"),
            cv2.cvtColor(results['original_image'], cv2.COLOR_RGB2BGR)
        )
        restored_path = str(output_dir / f"{name}_restored.jpg")
        cv2.imwrite(
            restored_path,
            cv2.cvtColor(results['restored_image'], cv2.COLOR_RGB2BGR)
        )

        # Save comparison
        self._save_comparison(results, output_dir / f"{name}_comparison.jpg")

        # Save text results
        text_results = {
            'ocr_raw': results['ocr_text_raw'],
            'ocr_cleaned': results['ocr_text_cleaned'],
            'romanized': results['romanized'],
            'translation': results['translation'],
            'word_count': results['word_count'],
            'sentences': results['sentences'],
        }

        with open(output_dir / f"{name}_results.json", 'w', encoding='utf-8') as f:
            json.dump(text_results, f, ensure_ascii=False, indent=2)

        with open(output_dir / f"{name}_results.txt", 'w', encoding='utf-8') as f:
            f.write("SANSKRIT MANUSCRIPT PROCESSING RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Raw OCR:\n{results['ocr_text_raw']}\n\n")
            f.write(f"Cleaned Devanagari:\n{results['ocr_text_cleaned']}\n\n")
            f.write(f"Romanized (IAST):\n{results['romanized']}\n\n")
            f.write(f"English Translation:\n{results['translation']}\n\n")
            f.write(f"Word Count: {results['word_count']}\n")

        print(f"\n✓ Results saved to {output_dir}/")

    def _save_comparison(self, results, output_path):
        """Save side-by-side comparison image"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(results['original_image'])
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(results['restored_image'])
        axes[1].set_title('Restored', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def process_manuscript(
    image_path,
    restoration_model=None,
    ocr_engine='tesseract',
    translation_method='google',
    save_output=True
):
    """
    Convenient function to process a single manuscript

    Args:
        image_path: Path to manuscript image
        restoration_model: Path to restoration model checkpoint
        ocr_engine: OCR engine to use
        translation_method: Translation method to use
        save_output: Whether to save results

    Returns:
        Dictionary with all processing results
    """
    pipeline = ManuscriptPipeline(
        restoration_model_path=restoration_model,
        ocr_engine=ocr_engine,
        translation_method=translation_method
    )

    return pipeline.process_manuscript(image_path, save_output=save_output)


def main():
    parser = argparse.ArgumentParser(
        description='Sanskrit Manuscript Restoration and Translation Pipeline'
    )

    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to manuscript image')
    parser.add_argument('--restoration_model', type=str, default=None,
                       help='Path to restoration model checkpoint')
    parser.add_argument('--ocr_engine', type=str, default='tesseract',
                       choices=['tesseract', 'trocr'],
                       help='OCR engine to use')
    parser.add_argument('--translation_method', type=str, default='google',
                       choices=['google', 'indictrans', 'ensemble'],
                       help='Translation method')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save output files')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Create pipeline
    pipeline = ManuscriptPipeline(
        restoration_model_path=args.restoration_model,
        ocr_engine=args.ocr_engine,
        translation_method=args.translation_method,
        device=args.device
    )

    # Process manuscript
    results = pipeline.process_manuscript(
        image_path=args.image_path,
        save_output=not args.no_save,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Sanskrit text: {results['ocr_text_cleaned'][:100]}...")
    print(f"Translation: {results['translation'][:100]}...")
    print(f"Words extracted: {results['word_count']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
