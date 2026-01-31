# Sanskrit OCR-Translation Agent Setup Guide

## 🎯 New Pipeline Architecture

This project now uses a **production-ready agent system** with the following workflow:

```
Restored Image → Google Lens OCR → Gemini Correction → Translation Model
```

### Stages:
1. **Image Restoration**: ViT-based model enhances degraded manuscripts
2. **Google Lens OCR**: Google Cloud Vision API extracts raw text
3. **Gemini Text Correction**: AI fixes OCR errors, restores proper Sanskrit
4. **Translation**: MarianMT translates to English
5. **Verification**: Quality checks and confidence scoring

---

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
source venv/bin/activate
pip install google-cloud-vision google-generativeai
```

### 2. Google Cloud Vision Setup (for OCR)

**Option A: Use Service Account (Recommended)**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable "Cloud Vision API"
4. Create a service account key:
   - Go to IAM & Admin → Service Accounts
   - Create Service Account
   - Grant "Cloud Vision AI Service Agent" role
   - Create JSON key
   - Download the JSON file

5. Set the credentials:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-credentials.json"
   ```

**Option B: Use in Streamlit UI**
- Upload the JSON credentials path in the sidebar

### 3. Gemini API Setup (for Text Correction)

1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

**Or enter directly in the Streamlit UI sidebar**

### 4. Verify Model Checkpoints

Ensure your trained restoration model is in place:
```bash
ls -lh checkpoints/kaggle/final.pth
ls -lh checkpoints/kaggle/best_psnr.pth
ls -lh checkpoints/kaggle/desti.pth
```

---

## 🚀 Running the Application

### Option 1: Streamlit Web UI (Recommended)

```bash
source venv/bin/activate
streamlit run app_enhanced.py
```

Then open: http://localhost:8501

### Option 2: Command Line

```bash
source venv/bin/activate
python sanskrit_ocr_agent.py \
  path/to/manuscript.jpg \
  --model checkpoints/kaggle/final.pth \
  --google-creds path/to/credentials.json \
  --gemini-key YOUR_API_KEY \
  --output output/results
```

---

## 📂 Output Format

The agent produces a JSON output with:

```json
{
  "restored_image_status": "Image restored: 1024×768px, noise removed...",
  "restored_image_path": "output/restored_output.png",
  "ocr_output_text": "Original text from Google Lens (may have errors)",
  "ocr_confidence": "85.00%",
  "corrected_sanskrit_text": "Gemini-corrected Sanskrit in Devanagari",
  "english_translation": "Accurate English translation",
  "notes": "All checks passed",
  "confidence_score": "92.50%",
  "is_valid": true,
  "processing_time_seconds": "3.45",
  "timestamp": "2024-11-27 14:30:00"
}
```

---

## 🔒 Agent Rules (Built-in)

The agent follows strict rules:
- ✅ Sequential workflow (no shortcuts)
- ✅ No hallucination - only extracts what OCR provides
- ✅ No invented content
- ✅ Accuracy > speed
- ✅ Confidence scoring and validation
- ✅ Complete paragraph extraction (not partial)

---

## 📊 Web UI Features

### Side-by-Side Image Comparison
- Original manuscript vs. Restored version
- Visual quality assessment

### Results Table
| Corrected Sanskrit | Original OCR | English Translation |
|-------------------|--------------|---------------------|
| Gemini-fixed text | Google Lens raw | Complete paragraph |

### Metrics Dashboard
- Processing time
- Word count
- Character count
- Confidence score

### Download Options
- Restored image (PNG)
- Sanskrit text (TXT)
- English translation (TXT)
- Full results (JSON)

---

## 🐛 Troubleshooting

### "Google Vision not available"
```bash
pip install google-cloud-vision
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

### "Gemini not available"
```bash
pip install google-generativeai
export GEMINI_API_KEY="your-key"
```

### "Model checkpoint not found"
Ensure your Kaggle-trained models are in:
```
checkpoints/kaggle/final.pth
checkpoints/kaggle/best_psnr.pth
checkpoints/kaggle/desti.pth
```

### Low OCR accuracy
1. Check image quality (try enhancing contrast)
2. Verify Google Cloud Vision is active
3. Check Gemini correction is working

---

## 🎓 Example Usage

```python
from sanskrit_ocr_agent import SanskritOCRTranslationAgent

# Initialize agent
agent = SanskritOCRTranslationAgent(
    restoration_model_path="checkpoints/kaggle/final.pth",
    google_credentials_path="credentials.json",
    gemini_api_key="YOUR_KEY"
)

# Process manuscript
result = agent.process(
    "path/to/manuscript.jpg",
    output_dir="output/results"
)

print(result["corrected_sanskrit_text"])
print(result["english_translation"])
```

---

## 📝 Notes

- First run may be slow (model loading)
- Google Vision API has quotas (check your project)
- Gemini API has rate limits
- Processing time: ~3-10 seconds per image

---

**Ready to process manuscripts!** 🕉️

