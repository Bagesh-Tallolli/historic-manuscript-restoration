# 🕉️ Sanskrit OCR - Quick Start Guide

## What is this?

A simple Streamlit web application that extracts Sanskrit text from images using Tesseract OCR.

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
./setup_and_run_ocr.sh
```

This script will:
- Check if Tesseract is installed
- Check for Sanskrit language data
- Install missing Python packages
- Launch the application

### Option 2: Manual Setup

1. **Install Tesseract with Sanskrit support:**
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr tesseract-ocr-san
   ```

2. **Install Python dependencies:**
   ```bash
   pip install streamlit pytesseract Pillow
   ```

3. **Run the application:**
   ```bash
   streamlit run app_sanskrit_ocr.py
   ```

## How to Use

1. Open your browser to `http://localhost:8501`
2. Upload an image (drag & drop or browse)
3. Click "Extract Sanskrit Text"
4. View and download the extracted text

## Files Created

- `app_sanskrit_ocr.py` - Main Streamlit application
- `run_sanskrit_ocr.sh` - Simple run script
- `setup_and_run_ocr.sh` - Setup and run script with checks
- `SANSKRIT_OCR_README.md` - Detailed documentation
- `QUICK_START_OCR.md` - This file

## Features

✨ **Key Features:**
- Drag and drop image upload
- Real-time text extraction
- Adjustable OCR settings
- Text statistics
- Download extracted text
- Copy to clipboard

📊 **OCR Settings (in sidebar):**
- Language selection (Sanskrit, Hindi, English combinations)
- OCR Engine Mode (Legacy, LSTM, or both)
- Page Segmentation Mode (automatic, single column, etc.)

## Supported Image Formats

- PNG
- JPG / JPEG

## Tips

✅ **For best results:**
- Use high-resolution images (300 DPI+)
- Ensure good lighting and contrast
- Use clear, printed text
- Keep images properly oriented

## Troubleshooting

### Error: Tesseract not found
```bash
# Install Tesseract
sudo apt-get install tesseract-ocr tesseract-ocr-san
```

### Error: No text detected
- Try adjusting OCR settings in the sidebar
- Check image quality and resolution
- Ensure the image contains Devanagari script

### Error: Language data not found
```bash
# Install Sanskrit language data
sudo apt-get install tesseract-ocr-san

# Verify installation
tesseract --list-langs
```

## Need More Help?

See `SANSKRIT_OCR_README.md` for detailed documentation.

---

**Quick Commands:**

```bash
# Run the app
./setup_and_run_ocr.sh

# Or manually
streamlit run app_sanskrit_ocr.py

# Check Tesseract version
tesseract --version

# List available languages
tesseract --list-langs
```

---

**Created**: November 2025  
**Author**: AI Assistant  
**Purpose**: Sanskrit text extraction from images

