# Enhanced Sanskrit OCR with AI Image Restoration

This enhanced version of the Sanskrit OCR application integrates the Kaggle-trained image restoration model to improve OCR accuracy on degraded manuscripts.

## Features

### 🔧 Image Restoration Pipeline
- **AI-Powered Restoration**: Uses a Vision Transformer (ViT) model trained on historical manuscript data
- **Patch-based Processing**: Maintains original resolution by processing large images in patches
- **Quality Enhancement**: Applies post-processing filters for better text clarity
- **Before/After Comparison**: View original and restored images side-by-side

### 📜 OCR & Translation
- **Sanskrit Text Extraction**: Uses Google Gemini AI for accurate text recognition
- **Multi-language Translation**: Automatic translation to Hindi, English, and Kannada
- **Intelligent Reconstruction**: Completes incomplete verses with context awareness

## Architecture

```
┌─────────────────┐
│  Upload Image   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Image Restoration (Optional)   │
│  - Load ViT Restoration Model   │
│  - Process in Patches           │
│  - Apply Post-processing        │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  OCR with Gemini AI             │
│  - Text Extraction              │
│  - Sanskrit Correction          │
│  - Multi-language Translation   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Display Results│
└─────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
# Using the specific requirements file
pip install -r ocr_gemini_streamlit_requirements.txt

# Or install individually
pip install streamlit google-genai pillow torch torchvision opencv-python numpy einops
```

### 2. Verify Model Checkpoint

Ensure the restoration model checkpoint exists in one of these locations:
- `checkpoints/kaggle/final_converted.pth` (recommended)
- `checkpoints/kaggle/final.pth`
- `models/trained_models/final.pth`

If you don't have the checkpoint, you can train it using the Kaggle training notebook.

## Usage

### Running the Application

```bash
streamlit run ocr_gemini_streamlit.py
```

### Using the Application

1. **Upload Image**: Click "Choose a manuscript image..." and select your Sanskrit manuscript
2. **Enable Restoration** (Optional): Toggle "Enable Image Restoration" in the sidebar
   - Recommended for degraded, faded, or damaged manuscripts
   - Disable for already clear images to save processing time
3. **Adjust Settings**: 
   - Temperature: Controls creativity in translation (0.0-1.0)
   - Lower values: More literal, conservative translations
   - Higher values: More creative, interpretive translations
4. **Analyze**: Click "🔍 Analyze & Translate" button
5. **View Results**:
   - Restored image (if restoration enabled)
   - Extracted Sanskrit text
   - Hindi translation
   - English translation
   - Kannada translation
6. **Download**:
   - Download restored image (PNG)
   - Download analysis results (TXT)

## Model Details

### Image Restoration Model
- **Architecture**: Vision Transformer (ViT) Base
- **Input Size**: 256x256 patches
- **Training Data**: Historical Sanskrit manuscripts with synthetic degradation
- **Capabilities**:
  - Removes noise and artifacts
  - Enhances faded text
  - Reconstructs damaged regions
  - Improves contrast and sharpness

### OCR Model
- **Engine**: Google Gemini AI (gemini-3-pro-preview)
- **Capabilities**:
  - High-accuracy Sanskrit OCR
  - Context-aware text correction
  - Verse reconstruction
  - Multi-language translation

## Configuration

You can customize the application by modifying these constants in `ocr_gemini_streamlit.py`:

```python
# Restoration Model Configuration
RESTORATION_CHECKPOINT_PATHS = [
    "checkpoints/kaggle/final_converted.pth",
    "checkpoints/kaggle/final.pth",
    "models/trained_models/final.pth",
]
RESTORATION_MODEL_SIZE = "base"  # Options: tiny, small, base, large
RESTORATION_IMG_SIZE = 256

# Gemini API Configuration
GEMINI_API_KEY = "your-api-key-here"
DEFAULT_MODEL = "gemini-3-pro-preview"
```

## Performance Optimization

### For Large Images (> 1024x1024)
- The application automatically uses patch-based processing
- Processing time: ~2-5 seconds per patch
- Patches overlap by 32 pixels for smooth blending

### For Small Images (< 512x512)
- Uses direct processing without patches
- Processing time: < 1 second

### GPU Acceleration
- Automatically uses CUDA if available
- Falls back to CPU if no GPU detected
- CPU processing is slower but still functional

## Troubleshooting

### Restoration Model Not Found
```
⚠️ Restoration model checkpoint not found. Restoration feature will be disabled.
```
**Solution**: Download or train the model checkpoint and place it in one of the configured paths.

### Out of Memory Error
```
CUDA out of memory
```
**Solution**: 
- Reduce image size before upload
- Switch to CPU mode by setting `device='cpu'` in the code
- Close other GPU-intensive applications

### Poor OCR Results
**Solutions**:
- Enable image restoration for degraded manuscripts
- Crop image to focus on text areas
- Ensure good lighting and contrast in original image
- Try adjusting temperature setting

## File Structure

```
EL-project/
├── ocr_gemini_streamlit.py              # Main application
├── ocr_gemini_streamlit_requirements.txt # Dependencies
├── models/
│   └── vit_restorer.py                  # ViT model architecture
├── utils/
│   └── image_restoration_enhanced.py    # Restoration utilities
└── checkpoints/
    └── kaggle/
        └── final_converted.pth          # Trained model weights
```

## Advanced Features

### Session State Management
- Restored images are cached in session state
- Prevents re-processing on page refresh
- Clear by refreshing the browser page

### Streaming Results
- OCR results stream in real-time
- See translations as they're generated
- Better user experience for long texts

### Download Options
- **Restored Image**: PNG format, original resolution
- **Analysis Results**: Plain text file with all translations

## Future Enhancements

- [ ] Batch processing for multiple images
- [ ] Custom training data upload
- [ ] Multiple OCR engine options
- [ ] Advanced text post-processing
- [ ] Export to different formats (PDF, DOCX)
- [ ] Historical manuscript database integration

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error messages in the Streamlit interface
3. Check model checkpoint paths and file permissions
4. Verify all dependencies are installed

## License

This project uses:
- Google Gemini AI (subject to Google's terms)
- PyTorch (BSD License)
- Streamlit (Apache 2.0)

## Credits

- **Image Restoration Model**: Trained using the Kaggle workflow
- **OCR Engine**: Google Gemini AI
- **UI Framework**: Streamlit

