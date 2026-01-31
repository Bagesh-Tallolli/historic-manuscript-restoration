#!/bin/bash

# Quick Setup and Launch Script for Sanskrit OCR Application

echo "=========================================="
echo "Sanskrit OCR - Setup & Launch"
echo "=========================================="
echo ""

# Check if Tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Tesseract OCR is not installed!"
    echo ""
    echo "Please install Tesseract first:"
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-san"
    echo "  macOS: brew install tesseract tesseract-lang"
    echo ""
    exit 1
else
    echo "✅ Tesseract OCR is installed"
    tesseract --version | head -n 1
fi

echo ""

# Check if Sanskrit language is available
if tesseract --list-langs 2>/dev/null | grep -q "san"; then
    echo "✅ Sanskrit language data is available"
else
    echo "⚠️  Sanskrit language data not found!"
    echo ""
    echo "Install it with:"
    echo "  sudo apt-get install tesseract-ocr-san"
    echo ""
    echo "Continuing anyway... (may not work properly)"
fi

echo ""
echo "Available Tesseract languages:"
tesseract --list-langs 2>&1 | tail -n +2 | head -n 10

echo ""
echo "=========================================="
echo "Checking Python dependencies..."
echo "=========================================="
echo ""

# Check Python packages
python3 -c "import streamlit" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Streamlit is installed"
else
    echo "⚠️  Streamlit not found. Installing..."
    pip install streamlit
fi

python3 -c "import pytesseract" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ pytesseract is installed"
else
    echo "⚠️  pytesseract not found. Installing..."
    pip install pytesseract
fi

python3 -c "from PIL import Image" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Pillow is installed"
else
    echo "⚠️  Pillow not found. Installing..."
    pip install Pillow
fi

echo ""
echo "=========================================="
echo "Launching Sanskrit OCR Application..."
echo "=========================================="
echo ""
echo "The application will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Launch the app
streamlit run app_sanskrit_ocr.py --server.port 8501 --server.address localhost

