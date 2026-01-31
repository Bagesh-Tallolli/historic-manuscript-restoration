# Sanskrit OCR Text Extraction Application

A simple Streamlit web application for extracting Sanskrit text from images using Tesseract OCR.

## Features

- 📤 **Easy Image Upload**: Drag and drop or browse to upload images
- 🔍 **Text Extraction**: Extract Sanskrit text using Tesseract OCR
- 📝 **Display Results**: View extracted text in a clean, readable format
- ⬇️ **Download**: Save extracted text as a .txt file
- ⚙️ **Customizable Settings**: Adjust OCR engine and page segmentation modes
- 📊 **Text Statistics**: View word count, character count, and line count

## Prerequisites

### 1. Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-san  # Sanskrit language data
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang  # Includes Sanskrit
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install Python Dependencies

```bash
pip install streamlit pytesseract Pillow
```

Or use the existing requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Using the Run Script

```bash
./run_sanskrit_ocr.sh
```

### Method 2: Direct Streamlit Command

```bash
streamlit run app_sanskrit_ocr.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## How to Use the Application

1. **Upload an Image**: 
   - Click on the upload area or drag and drop an image
   - Supported formats: PNG, JPG, JPEG

2. **Configure Settings** (Optional):
   - Use the sidebar to adjust OCR settings
   - Choose language (Sanskrit, Sanskrit+English, Hindi, etc.)
   - Select OCR engine mode
   - Select page segmentation mode

3. **Extract Text**:
   - Click the "Extract Sanskrit Text" button
   - Wait for processing (usually takes a few seconds)

4. **View Results**:
   - See the extracted text displayed on the right side
   - View text statistics (lines, words, characters)

5. **Download or Copy**:
   - Use the download button to save as .txt file
   - Or copy text directly from the expandable section

## Tips for Best Results

✅ **Do:**
- Use high-resolution images (300 DPI or higher)
- Ensure good lighting and contrast
- Use clear, printed text
- Keep the image properly oriented (not rotated)

❌ **Avoid:**
- Low-resolution or blurry images
- Skewed or rotated text
- Poor lighting or low contrast
- Handwritten text (works better with printed text)

## Troubleshooting

### Tesseract Not Found Error

If you get an error about Tesseract not being found:

1. **Verify Installation:**
   ```bash
   tesseract --version
   ```

2. **Check Sanskrit Language Data:**
   ```bash
   tesseract --list-langs
   ```
   You should see "san" in the list.

3. **Install Sanskrit Language Data:**
   ```bash
   sudo apt-get install tesseract-ocr-san
   ```

### No Text Detected

If no text is detected from your image:

1. Check image quality and resolution
2. Try different OCR settings in the sidebar
3. Ensure the image contains Devanagari script
4. Try adjusting the page segmentation mode

### Poor Recognition Quality

To improve recognition:

1. Use higher resolution images
2. Increase contrast in the image
3. Try different PSM (Page Segmentation Mode) settings
4. Use "LSTM only" engine mode for better accuracy

## OCR Settings Explained

### Languages
- **san**: Sanskrit (Devanagari script)
- **san+eng**: Sanskrit + English (for mixed text)
- **hin**: Hindi (similar script)
- **hin+eng**: Hindi + English

### OCR Engine Mode (OEM)
- **0**: Legacy engine only (faster, less accurate)
- **1**: Neural nets LSTM only (slower, more accurate)
- **2**: Legacy + LSTM
- **3**: Default (recommended)

### Page Segmentation Mode (PSM)
- **3**: Fully automatic (recommended for most cases)
- **4**: Single column of text
- **6**: Single uniform block of text
- **11**: Sparse text (for images with scattered text)
- **12**: Sparse text with OSD (orientation detection)

## Application Structure

```
app_sanskrit_ocr.py          # Main Streamlit application
run_sanskrit_ocr.sh          # Quick start script
SANSKRIT_OCR_README.md       # This file
```

## Technical Details

- **Framework**: Streamlit
- **OCR Engine**: Tesseract 4.x or higher
- **Language Support**: Sanskrit (Devanagari), Hindi, English
- **Image Processing**: PIL (Pillow)
- **Python**: 3.8+

## License

This application is part of the EL-project and follows the same license.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify Tesseract and language data installation
3. Review the Tesseract documentation: https://github.com/tesseract-ocr/tesseract

---

**Created**: November 2025  
**Purpose**: Simple Sanskrit text extraction from images using Tesseract OCR

