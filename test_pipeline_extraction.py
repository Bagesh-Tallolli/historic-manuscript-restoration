"""
Quick test to verify the complete pipeline extracts text correctly
"""
import sys
sys.path.insert(0, '/home/bagesh/EL-project')

from pathlib import Path

print("=" * 70)
print("TESTING COMPLETE PIPELINE - TEXT EXTRACTION")
print("=" * 70)

# Find a test image
test_image = Path('data/raw/test/test_0009.jpg')
if not test_image.exists():
    images = list(Path('data/raw').rglob('*.jpg'))
    if images:
        test_image = images[0]
    else:
        print("❌ No test images found")
        sys.exit(1)

print(f"\nTest image: {test_image}")

# Test the pipeline
print("\n" + "=" * 70)
print("INITIALIZING PIPELINE")
print("=" * 70)

try:
    from main import ManuscriptPipeline

    # Initialize WITHOUT restoration model for speed
    pipeline = ManuscriptPipeline(
        restoration_model_path=None,  # Skip restoration
        ocr_engine='tesseract',
        translation_method='google',
        device='cpu'
    )
    print("✅ Pipeline initialized")

except Exception as e:
    print(f"❌ Pipeline initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Process the image
print("\n" + "=" * 70)
print("PROCESSING IMAGE")
print("=" * 70)

try:
    results = pipeline.process_manuscript(str(test_image), save_output=False)
    print("\n✅ Processing complete!")

except Exception as e:
    print(f"❌ Processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check results
print("\n" + "=" * 70)
print("RESULTS ANALYSIS")
print("=" * 70)

print(f"\nResult keys available: {list(results.keys())}")

# Check OCR text
ocr_raw = results.get('ocr_text_raw', '')
ocr_cleaned = results.get('ocr_text_cleaned', '')
romanized = results.get('romanized', '')
translation = results.get('translation', '')
word_count = results.get('word_count', 0)

print(f"\n📊 Text Lengths:")
print(f"  OCR Raw:       {len(str(ocr_raw))} characters")
print(f"  OCR Cleaned:   {len(str(ocr_cleaned))} characters")
print(f"  Romanized:     {len(str(romanized))} characters")
print(f"  Translation:   {len(str(translation))} characters")
print(f"  Word Count:    {word_count}")

# Show previews
print(f"\n📜 OCR Raw (first 200 chars):")
print("-" * 70)
print(str(ocr_raw)[:200])
print("-" * 70)

print(f"\n📜 OCR Cleaned (first 200 chars):")
print("-" * 70)
print(str(ocr_cleaned)[:200])
print("-" * 70)

print(f"\n🔤 Romanized (first 200 chars):")
print("-" * 70)
print(str(romanized)[:200])
print("-" * 70)

print(f"\n🌍 Translation (first 200 chars):")
print("-" * 70)
print(str(translation)[:200])
print("-" * 70)

# Verdict
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if len(str(ocr_cleaned)) > 100:
    print("✅ TEXT EXTRACTION WORKS!")
    print(f"   - Extracted {len(str(ocr_cleaned))} characters")
    print(f"   - {word_count} words detected")
    print("   - OCR is functioning correctly")
    print("\n⚠️  If Streamlit doesn't show text, the issue is in the DISPLAY code, not OCR!")
    print("   - Check app.py display_results_table() function")
    print("   - Check that results are passed correctly to display function")
else:
    print("❌ TEXT EXTRACTION FAILED!")
    print("   - OCR is not extracting text")
    print("   - Check Tesseract installation")
    print("   - Check image quality")

# Test what Streamlit display function would receive
print("\n" + "=" * 70)
print("SIMULATING STREAMLIT DISPLAY FUNCTION")
print("=" * 70)

# This is what the display function extracts
if 'ocr_text' in results and isinstance(results['ocr_text'], dict):
    # Old format
    sanskrit_text = results['ocr_text'].get('cleaned', '')
    romanized_text = results['ocr_text'].get('romanized', '')
else:
    # New format
    sanskrit_text = results.get('ocr_text_cleaned', '')
    romanized_text = results.get('romanized', '')

translation_data = results.get('translation', '')
if isinstance(translation_data, dict):
    translation_text = translation_data.get('english', '')
else:
    translation_text = translation_data

print(f"Sanskrit text length: {len(str(sanskrit_text))}")
print(f"Romanized text length: {len(str(romanized_text))}")
print(f"Translation text length: {len(str(translation_text))}")

if len(str(sanskrit_text)) > 0:
    print("\n✅ Display function WILL receive text")
    print(f"   Preview: {str(sanskrit_text)[:100]}...")
else:
    print("\n❌ Display function will NOT receive text")
    print("   Check key names in results dictionary")
    print(f"   Available keys: {list(results.keys())}")

