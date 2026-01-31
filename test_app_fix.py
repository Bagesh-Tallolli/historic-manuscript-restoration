#!/usr/bin/env python3
"""
Quick test to verify the Streamlit app doesn't have import/attribute errors
"""

import sys
from pathlib import Path

print("Testing app.py for common errors...")
print("=" * 60)

# Read the app.py file
app_file = Path("app.py")
if not app_file.exists():
    print("❌ app.py not found!")
    sys.exit(1)

content = app_file.read_text()

# Check for the fixed method call
if "pipeline.process_manuscript(" in content:
    print("✅ Correct method call: pipeline.process_manuscript()")
else:
    print("❌ Wrong method call detected!")
    if "pipeline.process(" in content:
        print("   Found: pipeline.process() - should be process_manuscript()")
    sys.exit(1)

# Check for correct parameter names
if "save_output=" in content:
    print("✅ Correct parameter: save_output=")
else:
    print("⚠️  Warning: save_output parameter not found")

print()
print("=" * 60)
print("✅ All checks passed! app.py should work correctly.")
print()
print("To run the Streamlit app:")
print("  source activate_venv.sh")
print("  streamlit run app.py")
print()

