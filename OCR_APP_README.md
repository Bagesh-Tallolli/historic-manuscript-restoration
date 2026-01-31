# Sanskrit OCR & Translation Application

## Overview
This Streamlit application uses Google's Gemini 3 Pro Preview AI model to:
- Extract Sanskrit text from manuscript images
- Provide corrected Sanskrit text
- Translate to Hindi
- Translate to English

## Features
- **Advanced AI Model**: Uses Gemini 3 Pro Preview with HIGH thinking level
- **Clean Interface**: No model selection clutter - uses the best model by default
- **Background Processing**: Model processing happens seamlessly in the background
- **Multi-format Support**: PNG, JPG, JPEG, BMP images
- **Downloadable Results**: Save your translations to text files
- **Adjustable Creativity**: Control the AI's temperature/creativity level

## Installation

### Quick Start (Recommended)
```bash
./run_ocr_app.sh
```

### Manual Installation
1. Install dependencies:
```bash
pip install -r ocr_requirements.txt
```

2. Run the application:
```bash
streamlit run ocr_gemini_streamlit.py
```

## Usage

1. **Upload Image**: Click on the upload area and select a Sanskrit manuscript image
2. **Adjust Settings** (optional): Use the sidebar to adjust creativity level
3. **Analyze**: Click the "Analyze & Translate" button
4. **View Results**: See the extracted Sanskrit text and translations
5. **Download**: Save the results using the download button

## API Configuration

The application uses a hardcoded Gemini API key for convenience:
- API Key: `AIzaSyAxTN1rfdyJQVE3ecy5d8Zqkl5I431nBh0`
- Model: `gemini-3-pro-preview` (default, processed in background)

## Technical Details

### Model Configuration
- **Model**: gemini-3-pro-preview
- **Thinking Level**: HIGH (for better accuracy)
- **Streaming**: Yes (processes results as they arrive)
- **Temperature**: Adjustable (default: 0.3)

### System Prompt
The AI is configured with a specialized prompt for:
- Sanskrit text extraction
- Intelligent verse reconstruction
- Poetic (not literal) translation
- Dual output (Hindi + English)

## Output Format

The application provides results in this format:
```
1. Corrected Sanskrit (if needed)
2. Hindi Meaning:
3. English Meaning:
```

## Troubleshooting

### Model Not Found Error
If you see errors like "404 models/gemini-1.5-pro is not found", don't worry! This application now uses:
- `gemini-3-pro-preview` - the latest preview model
- The new `google-genai` library (not `google-generativeai`)

### Dependencies
Make sure you have installed:
```bash
pip install google-genai streamlit pillow
```

## Browser Access

Once running, the application will be available at:
- **Local**: http://localhost:8501
- **Network**: http://your-ip:8501

## Support

For issues or questions, check:
1. The console output for error messages
2. Ensure your API key is valid
3. Verify image format is supported (PNG, JPG, JPEG, BMP)

