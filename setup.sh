#!/bin/bash

# Setup script for Sanskrit Manuscript Restoration Pipeline

echo "=========================================="
echo "Sanskrit Manuscript Pipeline Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install Tesseract (Linux only)
echo ""
echo "=========================================="
echo "Installing Tesseract OCR"
echo "=========================================="
echo ""

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    echo "Installing Tesseract with Sanskrit support..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-san tesseract-ocr-hin
    echo "✓ Tesseract installed"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    echo "Installing Tesseract..."
    brew install tesseract tesseract-lang
    echo "✓ Tesseract installed"
else
    echo "Please install Tesseract manually:"
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-san"
    echo "  macOS: brew install tesseract tesseract-lang"
    echo "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Create directories
echo ""
echo "Creating directory structure..."
python dataset_downloader.py

# Test installation
echo ""
echo "=========================================="
echo "Testing Installation"
echo "=========================================="
echo ""

python -c "
import torch
import cv2
import numpy as np
print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ OpenCV version:', cv2.__version__)
print('✓ NumPy version:', np.__version__)

try:
    import pytesseract
    print('✓ Tesseract version:', pytesseract.get_tesseract_version())
except:
    print('✗ Tesseract not found')

try:
    from transformers import AutoTokenizer
    print('✓ Transformers available')
except:
    print('⚠ Transformers not available (optional)')

print('\n✓ All core dependencies installed!')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Add manuscript images to data/raw/"
echo ""
echo "3. Train the restoration model:"
echo "   python train.py --train_dir data/raw --epochs 100"
echo ""
echo "4. Process manuscripts:"
echo "   python main.py --image_path data/raw/your_image.jpg"
echo ""
echo "5. Or use the Jupyter notebook:"
echo "   jupyter notebook demo.ipynb"
echo ""
echo "=========================================="

