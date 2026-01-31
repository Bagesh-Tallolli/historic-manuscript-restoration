#!/usr/bin/env python3
"""
Quick verification test - Check if pipeline initializes correctly
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '/home/bagesh/EL-project')

from pipeline_clean import SanskritManuscriptPipeline

# Test initialization with correct parameters
try:
    print("Testing pipeline initialization...")
    print("-" * 60)

    pipeline = SanskritManuscriptPipeline(
        restoration_model_path='checkpoints/kaggle/final.pth',
        google_api_key=os.getenv('GOOGLE_API_KEY', ''),
        gemini_api_key=os.getenv('GEMINI_API_KEY', ''),
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

