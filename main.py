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

from models.vit_restorer import create_vit_restorer
from ocr.preprocess import OCRPreprocessor
from ocr.run_ocr import SanskritOCR
from nlp.unicode_normalizer import SanskritTextProcessor
from nlp.translation import SanskritTranslator


class ManuscriptPipeline:
    """Complete end-to-end manuscript processing pipeline"""

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
            self.restoration_model = create_vit_restorer('base', img_size=256)
            checkpoint = torch.load(restoration_model_path, map_location=self.device)
            self.restoration_model.load_state_dict(checkpoint['model_state_dict'])
            self.restoration_model.to(self.device)
            self.restoration_model.eval()
            print("✓ Restoration model loaded")
        else:
            print("⚠ Skipping restoration (no model provided)")
            self.restoration_model = None

        # 2. OCR Engine
        print(f"Initializing OCR ({ocr_engine})...")
        self.ocr = SanskritOCR(engine=ocr_engine)
        print("✓ OCR initialized")

        # 3. Text Processor (Unicode normalization)
        print("Initializing text processor...")
        self.text_processor = SanskritTextProcessor()
        print("✓ Text processor initialized")

        # 4. Translator
        print(f"Initializing translator ({translation_method})...")
        self.translator = SanskritTranslator(method=translation_method, device=self.device)
        print("✓ Translator initialized")

        print("\n✓ Pipeline ready!\n")

    def process_manuscript(self, image_path, save_output=False, output_dir='output'):
        """
        Process a manuscript image through the complete pipeline

        Args:
            image_path: Path to manuscript image
            save_output: Whether to save intermediate results
            output_dir: Directory to save outputs

        Returns:
            Dictionary with all results
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
        else:
            restored_img = original_img.copy()
            print("⚠ Skipped (no model)")

        # OCR
        print("\n2️⃣  OCR (Text Extraction)")
        print("-" * 60)

        ocr_text_raw = self.ocr.extract_text(restored_img, preprocess=True, lang='san')
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
        }

        # Save outputs
        if save_output:
            self._save_results(results, output_dir, image_path.stem)

        print("\n" + "=" * 60)
        print("✓ Processing complete!")

        return results

    def _restore_image(self, image):
        """Restore image using ViT model"""
        # Resize to model input size
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
        """Save all results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save images
        cv2.imwrite(
            str(output_dir / f"{name}_original.jpg"),
            cv2.cvtColor(results['original_image'], cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            str(output_dir / f"{name}_restored.jpg"),
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
