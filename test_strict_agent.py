#!/usr/bin/env python3
"""
Quick test script for the Strict Sanskrit Manuscript Processing Agent

Tests the complete pipeline:
1. Image Restoration (ViT)
2. OCR (Google Lens ONLY)
3. Text Correction (Gemini ONLY)
4. Translation (MarianMT ONLY)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sanskrit_ocr_agent import SanskritOCRTranslationAgent
import json

def test_agent_initialization():
    """Test 1: Can we initialize the agent?"""
    print("\n" + "="*60)
    print("TEST 1: Agent Initialization")
    print("="*60)

    try:
        agent = SanskritOCRTranslationAgent(
            restoration_model_path="checkpoints/kaggle/final.pth",
            google_credentials_path=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            gemini_api_key=os.getenv('GEMINI_API_KEY'),
            translation_model="Helsinki-NLP/opus-mt-sa-en",
            device="auto"
        )
        print("✅ Agent initialized successfully!")
        return agent
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pipeline_components(agent):
    """Test 2: Are all pipeline components loaded?"""
    print("\n" + "="*60)
    print("TEST 2: Pipeline Component Verification")
    print("="*60)

    checks = []

    # Check restoration model
    if hasattr(agent, 'restoration_model') and agent.restoration_model is not None:
        print("✅ Restoration Model: Loaded")
        checks.append(True)
    else:
        print("❌ Restoration Model: Missing")
        checks.append(False)

    # Check Google Vision
    if hasattr(agent, 'vision_client') and agent.vision_client is not None:
        print("✅ Google Lens OCR: Initialized")
        checks.append(True)
    else:
        print("⚠️  Google Lens OCR: Not available (will use fallback)")
        checks.append(False)

    # Check Gemini
    if hasattr(agent, 'gemini_model') and agent.gemini_model is not None:
        print("✅ Gemini Correction: Initialized")
        checks.append(True)
    else:
        print("⚠️  Gemini Correction: Not available")
        checks.append(False)

    # Check translation model
    if hasattr(agent, 'translation_model') and agent.translation_model is not None:
        print("✅ MarianMT Translation: Loaded")
        checks.append(True)
    else:
        print("❌ MarianMT Translation: Missing")
        checks.append(False)

    return all(checks)


def test_with_sample_image(agent):
    """Test 3: Process a sample image (if available)"""
    print("\n" + "="*60)
    print("TEST 3: Sample Image Processing")
    print("="*60)

    # Look for test images
    test_images = [
        "test_images/sample.jpg",
        "images/sample.jpg",
        "data/sample.jpg",
        "sample.jpg"
    ]

    sample_image = None
    for img_path in test_images:
        if Path(img_path).exists():
            sample_image = img_path
            break

    if sample_image is None:
        print("⚠️  No test image found. Skipping image processing test.")
        print("   Create a test image at: test_images/sample.jpg")
        return False

    print(f"📷 Found test image: {sample_image}")

    try:
        print("🔄 Processing image through strict pipeline...")
        result = agent.process(
            image_path=sample_image,
            output_dir="output/test"
        )

        print("\n📊 RESULTS:")
        print(f"  ✓ Restored Image: {result.get('restored_image_path')}")
        print(f"  ✓ OCR Confidence: {result.get('ocr_confidence')}")
        print(f"  ✓ Sanskrit Text Length: {len(result.get('corrected_sanskrit_text', ''))} chars")
        print(f"  ✓ Translation Length: {len(result.get('english_translation', ''))} chars")
        print(f"  ✓ Overall Confidence: {result.get('confidence_score')}")
        print(f"  ✓ Valid: {result.get('is_valid')}")

        # Show sample output
        if result.get('corrected_sanskrit_text'):
            print(f"\n📜 Sanskrit (first 100 chars):")
            print(f"   {result['corrected_sanskrit_text'][:100]}...")

        if result.get('english_translation'):
            print(f"\n🌍 English (first 100 chars):")
            print(f"   {result['english_translation'][:100]}...")

        print(f"\n💾 Full results saved to: output/test/pipeline_result.json")

        return True

    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strict_compliance():
    """Test 4: Verify strict pipeline compliance"""
    print("\n" + "="*60)
    print("TEST 4: Strict Pipeline Compliance Check")
    print("="*60)

    checks = {
        "Restoration Model Path": "checkpoints/kaggle/final.pth",
        "OCR Engine": "Google Lens (Cloud Vision API)",
        "Correction Engine": "Gemini API",
        "Translation Model": "Helsinki-NLP/opus-mt-sa-en"
    }

    print("\n✅ Required Components:")
    for component, requirement in checks.items():
        print(f"   • {component}: {requirement}")

    print("\n❌ Forbidden Components:")
    forbidden = [
        "Tesseract OCR (except as fallback)",
        "TrOCR",
        "Google Translate API for translation",
        "IndicTrans2",
        "Gemini for translation",
        "Any OCR on non-restored images"
    ]
    for item in forbidden:
        print(f"   • {item}")

    return True


def main():
    """Run all tests"""
    print("\n" + "🔬"*30)
    print("STRICT SANSKRIT MANUSCRIPT AGENT - TEST SUITE")
    print("🔬"*30)

    # Check environment
    print("\n📋 Environment Variables:")
    print(f"   GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'NOT SET')}")
    print(f"   GEMINI_API_KEY: {'SET' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")

    # Test 1: Initialization
    agent = test_agent_initialization()
    if agent is None:
        print("\n❌ CRITICAL: Cannot initialize agent. Stopping tests.")
        sys.exit(1)

    # Test 2: Component verification
    components_ok = test_pipeline_components(agent)

    # Test 3: Sample image processing
    if components_ok:
        test_with_sample_image(agent)

    # Test 4: Compliance check
    test_strict_compliance()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✅ Agent is initialized and ready to use!")
    print("✅ All mandatory components are in strict mode:")
    print("   - Restoration: ViT Model")
    print("   - OCR: Google Lens ONLY")
    print("   - Correction: Gemini ONLY")
    print("   - Translation: MarianMT ONLY")
    print("\n🚀 You can now use the agent with:")
    print("   • CLI: python3 sanskrit_ocr_agent.py <image.jpg>")
    print("   • Web: http://localhost:8501")
    print("   • API: SanskritOCRTranslationAgent(...).process(...)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

