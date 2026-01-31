#!/usr/bin/env python3
"""
Quick test script for the clean pipeline
Tests: ViT Restoration → Google Cloud Vision → Gemini
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import clean pipeline
from pipeline_clean import SanskritManuscriptPipeline

def test_pipeline():
    """Test the clean pipeline with a sample image"""

    # Configuration
    model_path = "checkpoints/kaggle/final.pth"
    google_api_key = os.getenv('GOOGLE_CLOUD_API_KEY', 'd48382987f9cddac6b042e3703797067fd46f2b0')
    import os
    from dotenv import load_dotenv
    load_dotenv()

    gemini_api_key = os.getenv('GEMINI_API_KEY', '')

    # Find a test image
    test_images = list(Path('data/raw/test').glob('*.jpg'))
    if not test_images:
        print("❌ No test images found in data/raw/test/")
        return

    test_image = str(test_images[0])
    print(f"📸 Using test image: {test_image}")

    # Check model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Initialize pipeline
    print("\n" + "="*60)
    pipeline = SanskritManuscriptPipeline(
        restoration_model_path=model_path,
        google_api_key=google_api_key,
        gemini_api_key=gemini_api_key,
        device='auto'
    )

    # Process
    output_dir = 'outputs/test_clean'
    result = pipeline.process(test_image, output_dir)

    # Display results
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    print(f"\n📜 Sanskrit Text (first 200 chars):")
    print(result['sanskrit_corrected'][:200])
    print(f"\n🌍 English Translation (first 200 chars):")
    print(result['translation_english'][:200])
    print(f"\n✅ All outputs saved to: {output_dir}/")
    print("="*60)

if __name__ == '__main__':
    test_pipeline()

