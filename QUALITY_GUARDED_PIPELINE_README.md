# 🔰 Quality-Guarded Manuscript Vision Pipeline

## Overview

This is a **production-ready Sanskrit manuscript processing pipeline** that ensures image restoration **NEVER degrades quality**. The pipeline combines:

- **ViT (Vision Transformer) Restoration Model** - AI-based image enhancement
- **Automatic Quality Gating** - Ensures restoration improves or maintains quality
- **Gemini Vision API** - OCR extraction, text correction, and multilingual translation

## 🎯 Key Innovation: Quality Gate

**THE PROBLEM**: Traditional pipelines apply restoration blindly, sometimes making images WORSE.

**OUR SOLUTION**: This pipeline:
1. ✅ Analyzes original image quality
2. ✅ Attempts restoration with ViT model
3. ✅ **Compares restored vs original** using multiple metrics
4. ✅ **Uses restored ONLY if it's better**
5. ✅ **Falls back to original if restoration degrades quality**

## 📊 Quality Metrics

The pipeline evaluates:
- **Sharpness** - Laplacian variance for edge clarity
- **Contrast** - RMS contrast measurement
- **Text Clarity** - Edge detection for character visibility
- **SSIM** - Structural similarity index
- **PSNR** - Peak signal-to-noise ratio

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT IMAGE                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  STEP 1: ANALYZE        │
         │  Original Image Quality  │
         │  • Sharpness            │
         │  • Contrast             │
         │  • Text Clarity         │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │  STEP 2: RESTORE        │
         │  ViT Restoration Model   │
         │  or PIL Fallback        │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │  STEP 3: QUALITY GATE   │
         │  🚦 CRITICAL DECISION    │
         │                         │
         │  Compare:               │
         │  • Restored vs Original │
         │  • SSIM > 0.70?         │
         │  • Improvement > 5%?    │
         │  • Sharpness preserved? │
         │                         │
         │  Decision:              │
         │  ✅ Use Restored        │
         │  ⚠️  Use Original       │
         └────────────┬────────────┘
                      │
                      ▼
              [BEST IMAGE]
                      │
                      ▼
         ┌─────────────────────────┐
         │  STEP 4-7: GEMINI API   │
         │  • OCR Extraction       │
         │  • Text Correction      │
         │  • Translation          │
         │  • Verification         │
         └────────────┬────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │  FINAL OUTPUT           │
         │  • Sanskrit Text        │
         │  • English Translation  │
         │  • Hindi Translation    │
         │  • Kannada Translation  │
         │  • Quality Metrics      │
         │  • Confidence Score     │
         └─────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Setup

1. **Gemini API Key** (Required for OCR/Translation)
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
   Or enter it in the UI sidebar

2. **ViT Model Checkpoint** (Optional, but recommended)
   - Place your trained ViT model at: `models/trained_models/final.pth`
   - If not available, the pipeline uses PIL-based fallback restoration

### Run the Application

#### Option 1: Use the startup script
```bash
./run_quality_guarded_pipeline.sh
```

#### Option 2: Run directly
```bash
streamlit run app_quality_guarded.py
```

The application will open in your browser at: `http://localhost:8501`

## 📁 Project Structure

```
EL-project/
├── manuscript_quality_guarded_pipeline.py  # Core pipeline logic
├── app_quality_guarded.py                  # Streamlit UI
├── run_quality_guarded_pipeline.sh         # Startup script
├── models/
│   ├── vit_restorer.py                     # ViT model architecture
│   └── trained_models/
│       └── final.pth                       # Trained ViT checkpoint
├── requirements.txt                         # Python dependencies
└── QUALITY_GUARDED_PIPELINE_README.md      # This file
```

## 🔧 Usage

### Web Interface (Recommended)

1. **Launch Application**
   ```bash
   ./run_quality_guarded_pipeline.sh
   ```

2. **Upload Image**
   - Click "Browse files" or drag & drop
   - Supports: PNG, JPG, JPEG, BMP

3. **Process**
   - Click "Run Quality-Guarded Pipeline"
   - Watch real-time quality comparison
   - See which image was selected (original or restored)

4. **Review Results**
   - Quality metrics and decision reasoning
   - Side-by-side image comparison
   - OCR text extraction
   - Multilingual translations
   - Confidence scores

5. **Export**
   - Download JSON results
   - Save selected image

### Command Line Interface

```python
from manuscript_quality_guarded_pipeline import process_manuscript_file

# Process a manuscript
result = process_manuscript_file("path/to/manuscript.jpg")

# Print results
import json
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### Programmatic Usage

```python
from PIL import Image
from manuscript_quality_guarded_pipeline import ManuscriptQualityGuardedPipeline

# Initialize pipeline
pipeline = ManuscriptQualityGuardedPipeline(
    api_key="your-gemini-api-key",
    vit_checkpoint="models/trained_models/final.pth"
)

# Load image
image = Image.open("manuscript.jpg")

# Process
result = pipeline.process_manuscript(image)

# Access results
print(f"Image used: {result['image_used']}")
print(f"Decision: {result['decision_reason']}")
print(f"Quality improvement: {result['improvement']:.3f}")
print(f"Sanskrit text: {result['corrected_sanskrit_text']}")
print(f"English: {result['english_translation']}")
```

## 🔒 Quality Gate Parameters

You can adjust these thresholds in `manuscript_quality_guarded_pipeline.py`:

```python
# In ImageQualityAnalyzer.compare_images()
MIN_IMPROVEMENT = 0.05  # Restoration must improve by at least 5%
MIN_SSIM = 0.70        # Restored must be at least 70% similar
```

## 📊 Quality Decision Logic

The pipeline uses restored image ONLY if ALL conditions are met:

1. ✅ **Improvement threshold**: Overall quality improves by ≥5%
2. ✅ **SSIM threshold**: Structural similarity ≥70%
3. ✅ **Sharpness preservation**: Sharpness ≥90% of original
4. ✅ **Contrast preservation**: Contrast ≥90% of original

**If ANY condition fails** → Uses original image

## 🎨 Output Format

```json
{
  "image_used": "original | restored",
  "restoration_applied": true | false,
  "decision_reason": "explanation of quality gate decision",
  "original_metrics": {
    "sharpness": 0.850,
    "contrast": 0.720,
    "text_clarity": 0.680,
    "overall": 0.765
  },
  "restored_metrics": { ... } or null,
  "improvement": 0.045,
  "ssim": 0.825,
  "psnr": 28.3,
  "image_quality_assessment": {
    "score": 0.85,
    "description": "..."
  },
  "ocr_extracted_text": "Sanskrit text in Devanagari",
  "corrected_sanskrit_text": "Corrected Sanskrit",
  "english_translation": "English translation",
  "hindi_translation": "हिन्दी अनुवाद",
  "kannada_translation": "ಕನ್ನಡ ಅನುವಾದ",
  "confidence_score": 0.85,
  "processing_notes": "..."
}
```

## 🛡️ Safety Guarantees

### What This Pipeline GUARANTEES:

✅ **Never degrades image quality**  
✅ **Always uses best available image**  
✅ **Transparent decision-making** (shows why each image was chosen)  
✅ **Automatic fallback** to original if restoration fails  
✅ **Quality metrics** for verification  

### What This Pipeline PREVENTS:

❌ Blind restoration application  
❌ False colors or distortions  
❌ Over-sharpening artifacts  
❌ Quality degradation  
❌ Loss of text clarity  

## 🧪 Testing

Test the pipeline with sample manuscripts:

```bash
# Command line test
python manuscript_quality_guarded_pipeline.py test_images/sample_manuscript.jpg

# Python test
python -c "
from manuscript_quality_guarded_pipeline import process_manuscript_file
result = process_manuscript_file('test_images/sample_manuscript.jpg')
print('Image used:', result['image_used'])
print('Decision:', result['decision_reason'])
"
```

## 🔬 Technical Details

### ViT Restoration Model

- **Architecture**: Vision Transformer (ViT) based image restoration
- **Input**: 256×256 patches
- **Output**: Restored RGB image
- **Training**: Trained on manuscript dataset with paired degraded/clean images

### Quality Analysis

- **Sharpness**: Laplacian variance (normalized 0-1)
- **Contrast**: RMS contrast (standard deviation of grayscale)
- **Text Clarity**: Canny edge detection density
- **Overall Score**: Weighted combination (35% sharpness, 30% clarity, 25% contrast, 10% brightness)

### Gemini API Integration

- **Model**: `gemini-2.0-flash-exp`
- **Temperature**: 0.1 (low for consistency)
- **Output**: JSON-formatted results
- **Features**: Vision understanding, OCR, language models

## ⚙️ Configuration

### Environment Variables

```bash
# Required for OCR/Translation
export GEMINI_API_KEY="your-api-key"

# Optional: Custom model path
export VIT_CHECKPOINT="path/to/model.pth"

# Optional: Device selection
export DEVICE="cuda"  # or "cpu"
```

### In-Code Configuration

Edit `manuscript_quality_guarded_pipeline.py`:

```python
# API Configuration
GEMINI_API_KEY = "your-api-key"
DEFAULT_MODEL = "gemini-2.0-flash-exp"
VIT_CHECKPOINT = "models/trained_models/final.pth"

# Quality Gate Thresholds (in ImageQualityAnalyzer.compare_images)
MIN_IMPROVEMENT = 0.05  # 5% minimum improvement
MIN_SSIM = 0.70         # 70% structural similarity
```

## 🐛 Troubleshooting

### Issue: ViT model not found

```
⚠️  ViT checkpoint not found at models/trained_models/final.pth
⚠️  Will use fallback PIL-based restoration
```

**Solution**: The pipeline works fine with PIL fallback. For better results, train or download a ViT checkpoint.

### Issue: Gemini API errors

```
❌ Error calling Gemini API: ...
```

**Solution**: 
- Check API key is valid
- Check internet connection
- Check API quota/limits

### Issue: Always uses original image

```
⚠️  USING ORIGINAL IMAGE
Reason: Insufficient improvement
```

**Solution**: This is CORRECT behavior! It means restoration didn't improve quality. The pipeline is protecting you from degradation.

### Issue: Low confidence scores

**Solution**: 
- Try a clearer image
- Check if the manuscript is in Sanskrit/Devanagari
- Review the "processing_notes" field for hints

## 📈 Performance

- **Image restoration**: 1-3 seconds (ViT) or <1 second (PIL fallback)
- **Quality analysis**: <1 second
- **Gemini API call**: 3-8 seconds
- **Total pipeline**: ~5-15 seconds per image

## 🤝 Contributing

To improve the pipeline:

1. **Better restoration models**: Train ViT on more data
2. **Quality metrics**: Add more sophisticated quality measures
3. **Language support**: Add more target languages
4. **UI enhancements**: Improve visualization

## 📄 License

See LICENSE file in project root.

## 🙏 Acknowledgments

- **ViT Architecture**: Vision Transformers for image processing
- **Gemini API**: Google's multimodal AI for vision and language
- **Quality Metrics**: scikit-image for SSIM, PSNR
- **UI Framework**: Streamlit for rapid prototyping

## 📞 Support

For issues or questions:
1. Check this README
2. Review error messages and "processing_notes"
3. Test with simpler/clearer images first
4. Check API keys and model paths

## 🎯 Summary

**This pipeline solves the critical problem of restoration degradation.**

Instead of blindly applying restoration (which can make images worse), it:
- ✅ Compares quality metrics
- ✅ Uses the BEST image
- ✅ Provides transparency
- ✅ Guarantees no degradation

**Perfect for production use where quality matters.**

---

**🔰 Quality-Guarded Manuscript Vision Pipeline**  
*Restoration that never degrades. Intelligence that always protects.*

