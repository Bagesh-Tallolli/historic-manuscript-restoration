#!/usr/bin/env python3
"""
Simple CLI test for the quality-guarded pipeline
Tests ViT restoration + Gemini API
"""

import sys
import os
from pathlib import Path
from PIL import Image
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_quality_guarded_pipeline import ManuscriptQualityGuardedPipeline

def test_pipeline_cli():
    """Test the pipeline with a sample image"""

    print("\n" + "="*70)
    print("🔰 QUALITY-GUARDED PIPELINE - CLI TEST")
    print("="*70 + "\n")

    # Find a test image
    test_images = [
        "debug_original.png",
        "outputs/test_restoration.png",
        "output/test_sample_original.jpg",
    ]

    image_path = None
    for img in test_images:
        if Path(img).exists():
            image_path = img
            break

    if not image_path:
        print("❌ No test images found. Please provide an image path:")
        print("   Usage: python test_quality_guarded_pipeline.py <image_path>")
        return

    print(f"📷 Using test image: {image_path}\n")

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  Warning: GEMINI_API_KEY not found in environment")
        print("   The pipeline will use the default key\n")

    # Check checkpoint
    checkpoint = "checkpoints/kaggle/final_converted.pth"
    if Path(checkpoint).exists():
        print(f"✅ ViT checkpoint found: {checkpoint}")
    else:
        print(f"⚠️  ViT checkpoint not found: {checkpoint}")
        print("   Will use PIL-based fallback restoration")

    print("\n" + "-"*70)
    print("🚀 STARTING PIPELINE")
    print("-"*70 + "\n")

    try:
        # Load image
        image = Image.open(image_path)
        print(f"✅ Image loaded: {image.size[0]}x{image.size[1]} pixels\n")

        # Initialize pipeline
        pipeline = ManuscriptQualityGuardedPipeline(api_key=api_key)

        # Process manuscript
        print("Processing manuscript...\n")
        results = pipeline.process_manuscript(image)

        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE - RESULTS")
        print("="*70 + "\n")

        # Display key results
        print(f"Image Used for OCR: {results.get('image_used', 'N/A')}")
        print(f"Restoration Applied: {results.get('restoration_applied', 'N/A')}")
        print(f"Decision: {results.get('decision_reason', 'N/A')}")
        print(f"Confidence Score: {results.get('confidence_score', 'N/A')}")

        if 'improvement' in results:
            print(f"\nQuality Metrics:")
            print(f"  • Improvement: {results['improvement']:+.3f}")
            print(f"  • SSIM: {results.get('ssim', 'N/A'):.3f}")
            print(f"  • PSNR: {results.get('psnr', 'N/A'):.2f} dB")

        print("\n" + "-"*70)

        if results.get('corrected_sanskrit_text'):
            print("\n📜 CORRECTED SANSKRIT TEXT:")
            print("-"*70)
            print(results['corrected_sanskrit_text'])
            print("-"*70)

        if results.get('english_translation'):
            print("\n🇬🇧 ENGLISH TRANSLATION:")
            print("-"*70)
            print(results['english_translation'])
            print("-"*70)

        if results.get('hindi_translation'):
            print("\n🇮🇳 HINDI TRANSLATION:")
            print("-"*70)
            print(results['hindi_translation'])
            print("-"*70)

        if results.get('kannada_translation'):
            print("\n🇮🇳 KANNADA TRANSLATION:")
            print("-"*70)
            print(results['kannada_translation'])
            print("-"*70)

        # Save results
        output_file = "test_pipeline_results.json"
        results_export = {k: v for k, v in results.items() if k not in ['original_image', 'selected_image']}

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_export, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Full results saved to: {output_file}")

        print("\n" + "="*70)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")

    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR OCCURRED")
        print("="*70)
        print(f"\n{str(e)}\n")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use provided image path
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"❌ Error: Image not found: {image_path}")
            sys.exit(1)

        # Temporarily replace test image logic
        from PIL import Image
        import json

        print(f"\n📷 Using provided image: {image_path}\n")

        image = Image.open(image_path)
        pipeline = ManuscriptQualityGuardedPipeline()
        results = pipeline.process_manuscript(image)

        print("\n✅ Processing complete!")
        print(f"Image used: {results.get('image_used')}")
        print(f"Confidence: {results.get('confidence_score')}")

        # Save results
        results_export = {k: v for k, v in results.items() if k not in ['original_image', 'selected_image']}
        with open("test_pipeline_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_export, f, indent=2, ensure_ascii=False)

        print("\n💾 Results saved to: test_pipeline_results.json")
    else:
        sys.exit(test_pipeline_cli())

