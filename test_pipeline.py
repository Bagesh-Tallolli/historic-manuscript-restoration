#!/usr/bin/env python3
"""
Test script for Quality-Guarded Manuscript Pipeline
Demonstrates command-line usage without the web UI
"""

import sys
from pathlib import Path
from PIL import Image
import json
from manuscript_quality_guarded_pipeline import ManuscriptQualityGuardedPipeline

def test_pipeline(image_path: str):
    """Test the pipeline with a given image"""

    print("\n" + "="*70)
    print("🔰 QUALITY-GUARDED MANUSCRIPT PIPELINE - CLI TEST")
    print("="*70)

    # Check if image exists
    if not Path(image_path).exists():
        print(f"\n❌ Error: Image not found at {image_path}")
        print("\nUsage:")
        print("  python test_pipeline.py <image_path>")
        print("\nExample:")
        print("  python test_pipeline.py samples/manuscript.jpg")
        return

    # Load image
    print(f"\n📂 Loading image: {image_path}")
    try:
        image = Image.open(image_path)
        print(f"✅ Image loaded: {image.size[0]} × {image.size[1]} pixels, mode: {image.mode}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Initialize pipeline
    print("\n🔧 Initializing pipeline...")
    try:
        pipeline = ManuscriptQualityGuardedPipeline()
        print("✅ Pipeline initialized")
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        return

    # Process manuscript
    print("\n🚀 Processing manuscript...")
    print("-" * 70)

    try:
        result = pipeline.process_manuscript(image)

        print("\n" + "="*70)
        print("✅ PROCESSING COMPLETE - RESULTS")
        print("="*70)

        # Quality Decision
        print(f"\n🚦 QUALITY GATE DECISION:")
        print(f"   Image Used: {result['image_used'].upper()}")
        print(f"   Restoration Applied: {'Yes' if result['restoration_applied'] else 'No'}")
        print(f"   Reason: {result['decision_reason']}")

        # Quality Metrics
        print(f"\n📊 QUALITY METRICS:")
        orig = result['original_metrics']
        print(f"   Original Quality:")
        print(f"      • Sharpness:    {orig['sharpness']:.3f}")
        print(f"      • Contrast:     {orig['contrast']:.3f}")
        print(f"      • Text Clarity: {orig['text_clarity']:.3f}")
        print(f"      • Overall:      {orig['overall']:.3f}")

        if result['restoration_applied'] and result['restored_metrics']:
            rest = result['restored_metrics']
            print(f"\n   Restored Quality:")
            print(f"      • Sharpness:    {rest['sharpness']:.3f}")
            print(f"      • Contrast:     {rest['contrast']:.3f}")
            print(f"      • Text Clarity: {rest['text_clarity']:.3f}")
            print(f"      • Overall:      {rest['overall']:.3f}")

            print(f"\n   Comparison:")
            print(f"      • Improvement:  {result['improvement']:+.3f}")
            print(f"      • SSIM:         {result['ssim']:.3f}")
            print(f"      • PSNR:         {result['psnr']:.2f} dB")

        # OCR Results
        if 'ocr_extracted_text' in result:
            print(f"\n📝 OCR EXTRACTION:")
            ocr_text = result['ocr_extracted_text']
            if len(ocr_text) > 100:
                print(f"   {ocr_text[:100]}...")
            else:
                print(f"   {ocr_text}")

        # Corrected Text
        if 'corrected_sanskrit_text' in result:
            print(f"\n✅ CORRECTED SANSKRIT:")
            corrected = result['corrected_sanskrit_text']
            if len(corrected) > 100:
                print(f"   {corrected[:100]}...")
            else:
                print(f"   {corrected}")

        # Translations
        if 'english_translation' in result:
            print(f"\n🇬🇧 ENGLISH TRANSLATION:")
            english = result['english_translation']
            if len(english) > 150:
                print(f"   {english[:150]}...")
            else:
                print(f"   {english}")

        if 'hindi_translation' in result:
            print(f"\n🇮🇳 HINDI TRANSLATION:")
            hindi = result['hindi_translation']
            if len(hindi) > 100:
                print(f"   {hindi[:100]}...")
            else:
                print(f"   {hindi}")

        if 'kannada_translation' in result:
            print(f"\n🇮🇳 KANNADA TRANSLATION:")
            kannada = result['kannada_translation']
            if len(kannada) > 100:
                print(f"   {kannada[:100]}...")
            else:
                print(f"   {kannada}")

        # Confidence Score
        if 'confidence_score' in result:
            confidence = result['confidence_score']
            if isinstance(confidence, str):
                confidence = float(confidence) if confidence else 0.0

            print(f"\n🎯 CONFIDENCE SCORE: {confidence:.2f}")
            if confidence >= 0.8:
                print("   Quality: 🟢 HIGH")
            elif confidence >= 0.6:
                print("   Quality: 🟡 MEDIUM")
            else:
                print("   Quality: 🔴 LOW")

        # Processing Notes
        if 'processing_notes' in result and result['processing_notes']:
            print(f"\n📋 PROCESSING NOTES:")
            print(f"   {result['processing_notes']}")

        # Save results
        output_file = Path(image_path).stem + "_results.json"
        export_data = {k: v for k, v in result.items() if k not in ['original_image', 'selected_image']}
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 RESULTS SAVED:")
        print(f"   {output_file}")

        # Save selected image
        if 'selected_image' in result:
            output_image = Path(image_path).stem + f"_selected_{result['image_used']}.png"
            result['selected_image'].save(output_image)
            print(f"   {output_image}")

        print("\n" + "="*70)
        print("✅ TEST COMPLETE")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error processing manuscript: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("🔰 QUALITY-GUARDED MANUSCRIPT PIPELINE - CLI TEST")
        print("="*70)
        print("\nUsage:")
        print("  python test_pipeline.py <image_path>")
        print("\nExample:")
        print("  python test_pipeline.py samples/manuscript.jpg")
        print("  python test_pipeline.py debug_original.png")
        print("\n" + "="*70 + "\n")
        sys.exit(1)

    image_path = sys.argv[1]
    test_pipeline(image_path)

