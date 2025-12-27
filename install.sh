#!/bin/bash
# Quick installation script for Sanskrit Manuscript Restoration Pipeline
# This script sets up the environment with all fixed dependencies

set -e  # Exit on error

echo "=========================================="
echo "Sanskrit Manuscript Restoration Pipeline"
echo "Installation Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo ""
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider creating one with: python3 -m venv venv && source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Clear pip cache to save space
echo ""
echo "Clearing pip cache..."
python3 -m pip cache purge

# Install requirements
echo ""
echo "Installing dependencies from requirements.txt..."
echo "This may take several minutes..."
python3 -m pip install -r requirements.txt

# Check installation
echo ""
echo "Verifying installation..."

# Test key imports
python3 << 'EOF'
import sys
success = True

print("Testing imports...")

try:
    import torch
    print("✓ PyTorch")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    success = False

try:
    import cv2
    print("✓ OpenCV")
except ImportError as e:
    print(f"✗ OpenCV: {e}")
    success = False

try:
    import numpy
    print("✓ NumPy")
except ImportError as e:
    print(f"✗ NumPy: {e}")
    success = False

try:
    from deep_translator import GoogleTranslator
    print("✓ deep-translator")
except ImportError as e:
    print(f"✗ deep-translator: {e}")
    success = False

try:
    from roboflow import Roboflow
    print("✓ roboflow")
except ImportError as e:
    print(f"✗ roboflow: {e}")
    success = False

try:
    from transformers import AutoTokenizer
    print("✓ transformers")
except ImportError as e:
    print(f"✗ transformers: {e}")
    success = False

if not success:
    print("\n⚠️  Some imports failed. Check the errors above.")
    sys.exit(1)
else:
    print("\n✓ All key dependencies installed successfully!")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation verification failed"
    echo "   Please check the error messages above"
    exit 1
fi

# Check for Tesseract
echo ""
echo "Checking for Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    tesseract_version=$(tesseract --version 2>&1 | head -n 1)
    echo "✓ $tesseract_version"
else
    echo "⚠️  Tesseract OCR not found"
    echo "   Install with:"
    echo "   Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-san"
    echo "   macOS: brew install tesseract tesseract-lang"
fi

# Run test setup if it exists
if [ -f "test_setup.py" ]; then
    echo ""
    echo "Running setup tests..."
    python3 test_setup.py
fi

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Install Tesseract if not already installed (see above)"
echo "2. Run test: python test_setup.py"
echo "3. Process a manuscript: python main.py --image_path <path>"
echo "4. Or open the demo notebook: jupyter notebook demo.ipynb"
echo ""
echo "For more information, see:"
echo "- INSTALLATION_GUIDE.md"
echo "- GETTING_STARTED.md"
echo "- README.md"
echo ""
