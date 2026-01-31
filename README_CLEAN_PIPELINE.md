# 🕉️ Sanskrit Manuscript Pipeline - Clean Version

**100% Clean Implementation**  
ViT Restoration → Google Cloud Vision → Gemini Correction & Translation

---

## 🎯 What This Pipeline Does

This is a **completely rebuilt** Sanskrit manuscript processing pipeline that:

1. ✨ **Restores** degraded manuscript images using your trained ViT model
2. 📖 **Extracts** Sanskrit text using Google Cloud Vision API
3. 🔧 **Corrects** OCR errors using Gemini AI
4. 🌍 **Translates** to English using Gemini AI

**What's Different:**
- ❌ NO Tesseract
- ❌ NO TrOCR
- ❌ NO old translation models
- ✅ ONLY: ViT + Google Cloud Vision + Gemini
- ✅ Clean, simple, production-ready code

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies

```bash
cd /home/bagesh/EL-project
source venv/bin/activate
pip install -r requirements_clean.txt
```

### 2️⃣ Configure API Keys

Edit the `.env` file with your API keys:

```bash
# Google Cloud Vision API Key
GOOGLE_CLOUD_API_KEY=your_google_vision_api_key

# Gemini API Key
GEMINI_API_KEY=your_gemini_api_key

# Model Path (already set)
RESTORATION_MODEL_PATH=checkpoints/kaggle/final.pth
```

**Your current keys are already in `.env`:**
- Google Cloud Vision: `d48382987f9cddac6b042e3703797067fd46f2b0`
- Gemini: `AIzaSyBIORWk0PZThY5m3yCudftd3sssnZADi_A`

### 3️⃣ Run the Web Interface

```bash
./run_clean.sh
```

Then open: **http://localhost:8504**

### 4️⃣ Or Use Command Line

```bash
python pipeline_clean.py \
    --image data/raw/test/test_0001.jpg \
    --model checkpoints/kaggle/final.pth \
    --google-key "your_google_key" \
    --gemini-key "your_gemini_key" \
    --output outputs/test
```

---

## 📂 File Structure

```
/home/bagesh/EL-project/
├── pipeline_clean.py          # Clean pipeline (ONLY Gemini + Google Vision)
├── app_clean.py               # Clean Streamlit UI
├── run_clean.sh               # Launch script
├── test_clean_pipeline.py     # Test script
├── requirements_clean.txt     # Minimal dependencies
├── .env                       # API keys (configured)
│
├── checkpoints/kaggle/
│   └── final.pth             # Your trained ViT model (330MB)
│
├── data/raw/test/            # Test images (59 samples)
│
└── outputs/                  # Results will be saved here
    ├── restored_output.png
    ├── ocr_raw.txt
    ├── sanskrit_cleaned.txt
    ├── english_translation.txt
    └── pipeline_output.json
```

---

## 🎨 Web Interface

The clean UI provides:

1. **Upload Section**
   - Browse and upload manuscript image
   - Configure API keys in sidebar

2. **Image Comparison**
   - Original vs Restored (side-by-side)
   - Quality metrics

3. **Results Table**
   - Sanskrit Text (Corrected)
   - English Translation
   - Confidence scores

4. **Download Buttons**
   - Restored image
   - Sanskrit text
   - English translation
   - Full JSON output

---

## 🔧 Pipeline Stages

### Stage 1: Image Restoration
- **Model**: Your trained ViT (86.4M parameters)
- **Input**: Degraded manuscript
- **Output**: Enhanced, noise-removed image
- **File**: `restored_output.png`

### Stage 2: OCR Extraction
- **Engine**: Google Cloud Vision API
- **Method**: Document text detection (full paragraph)
- **Fallback**: Gemini Vision if Google fails
- **Output**: Raw Sanskrit text
- **File**: `ocr_raw.txt`

### Stage 3: Text Correction
- **Engine**: Gemini Pro
- **Task**: Fix OCR errors, restore proper Devanagari
- **Rules**: No hallucination, no translation
- **Output**: Corrected Sanskrit
- **File**: `sanskrit_cleaned.txt`

### Stage 4: Translation
- **Engine**: Gemini Pro
- **Task**: Sanskrit → English translation
- **Style**: Natural, faithful, complete paragraph
- **Output**: English translation
- **File**: `english_translation.txt`

---

## 📊 Output Format

### JSON Output (`pipeline_output.json`)

```json
{
  "restored_image_path": "outputs/restored_output.png",
  "ocr_sanskrit_raw": "रामो राजमणिः सदा विजयते...",
  "sanskrit_corrected": "रामो राजमणिः सदा विजयते...",
  "translation_english": "Rama, the jewel of kings, always triumphs...",
  "confidence": {
    "ocr_score": "85.0%",
    "correction_reliability": "high"
  },
  "metadata": {
    "processing_time_seconds": 12.34,
    "word_count": 45,
    "character_count": 234
  }
}
```

---

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_clean_pipeline.py
```

This will:
1. Load your ViT model
2. Process a test image
3. Run through all 4 stages
4. Display results

---

## ⚙️ Configuration

### API Keys

**Google Cloud Vision API:**
- Get key from: https://console.cloud.google.com/apis/credentials
- Enable: Cloud Vision API
- Your key: `d48382987f9cddac6b042e3703797067fd46f2b0`

**Gemini API:**
- Get key from: https://makersuite.google.com/app/apikey
- Your key: `AIzaSyBIORWk0PZThY5m3yCudftd3sssnZADi_A`

### Model Path

Default: `checkpoints/kaggle/final.pth` (330MB)
Alternative: `checkpoints/kaggle/desti.pth` (990MB - full checkpoint)

---

## 🆚 Comparison with Old Pipeline

| Feature | Old Pipeline | Clean Pipeline |
|---------|-------------|----------------|
| OCR Engine | Tesseract + TrOCR | Google Cloud Vision + Gemini |
| Translation | MarianMT/IndicTrans | Gemini |
| Text Correction | Manual rules | Gemini AI |
| Code Complexity | High (multiple models) | Low (2 APIs) |
| Dependencies | 20+ packages | 8 packages |
| File Size | Large | Minimal |
| Accuracy | Variable | High (AI-powered) |

---

## 📝 Notes

1. **API Costs:**
   - Google Cloud Vision: ~$1.50 per 1000 images
   - Gemini: Free tier available, then pay-per-use
   
2. **Fallback:**
   - If Google Vision fails, pipeline uses Gemini Vision for OCR
   
3. **Rate Limits:**
   - Google Vision: 1800 requests/min
   - Gemini: 60 requests/min (free tier)

4. **Model Loading:**
   - First run loads ViT model (~5-10 seconds)
   - Subsequent runs are cached (instant)

---

## 🐛 Troubleshooting

### "Model not found"
```bash
# Check if model exists
ls -lh checkpoints/kaggle/final.pth

# If missing, check desti.pth
ls -lh checkpoints/kaggle/desti.pth
```

### "API key invalid"
- Verify keys in `.env` file
- Check API is enabled in Google Cloud Console
- Ensure no extra spaces in keys

### "No text extracted"
- Check image quality
- Try different test image
- Pipeline will fallback to Gemini Vision automatically

---

## ✅ Success Criteria

Your pipeline is working correctly when:

1. ✅ Web UI loads without errors
2. ✅ Image restoration completes (see side-by-side comparison)
3. ✅ Sanskrit text appears in results table
4. ✅ English translation is generated
5. ✅ All download buttons work
6. ✅ JSON output is valid

---

## 🚀 Next Steps

1. **Test the Pipeline:**
   ```bash
   python test_clean_pipeline.py
   ```

2. **Launch Web UI:**
   ```bash
   ./run_clean.sh
   ```

3. **Process Your Images:**
   - Upload via web UI, or
   - Use command line for batch processing

---

## 📧 Support

If you encounter issues:
1. Check API keys in `.env`
2. Verify model exists: `checkpoints/kaggle/final.pth`
3. Check test images exist: `data/raw/test/`
4. Review error messages carefully

---

**Built with:**
- 🤖 PyTorch (ViT Restoration)
- ☁️ Google Cloud Vision API
- 🧠 Google Gemini AI
- 🎨 Streamlit

**Status:** ✅ Ready to Use

