#!/usr/bin/env python3
"""
Quick verification test - Check if pipeline initializes correctly
"""

import sys
sys.path.insert(0, '/home/bagesh/EL-project')

from pipeline_clean import SanskritManuscriptPipeline

# Test initialization with correct parameters
try:
    print("Testing pipeline initialization...")
    print("-" * 60)

    pipeline = SanskritManuscriptPipeline(
        restoration_model_path='checkpoints/kaggle/final.pth',
        google_api_key='d48382987f9cddac6b042e3703797067fd46f2b0',
        gemini_api_key='AIzaSyBIORWk0PZThY5m3yCudftd3sssnZADi_A',
        device='cpu'  # Use CPU for quick test
    )

    print("-" * 60)
    print("✅ SUCCESS! Pipeline initialized correctly.")
    print(f"   Device: {pipeline.device}")
    print(f"   Google API Key: {pipeline.google_api_key[:20]}...")
    print(f"   Model loaded: {pipeline.restoration_model is not None}")
    print("-" * 60)

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

