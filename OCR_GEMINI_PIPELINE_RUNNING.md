# 🔰 OCR GEMINI STREAMLIT PIPELINE - RUNNING

## ✅ STATUS: ACTIVE

The complete Sanskrit manuscript processing pipeline is now running with:
- **ViT Restoration Model** for image quality enhancement
- **Gemini API** for OCR text extraction
- **Multilingual Translation** (Sanskrit → English, Hindi, Kannada)

---

## 🌐 ACCESS INFORMATION

### Local Access
```
http://localhost:8501
```

### Network Access (from other devices on same network)
```
http://172.20.66.141:8501
```

### External Access (if port forwarding is configured)
```
http://106.51.197.176:8501
```

---

## 📋 PIPELINE FEATURES

### ✨ Image Restoration
- **Checkpoint**: `checkpoints/kaggle/final_converted.pth`
- **Model**: Vision Transformer (ViT) base model
- **Processing**: Patch-based restoration with enhanced quality
- **Toggle**: Can be enabled/disabled in UI sidebar
- **Note**: OCR always uses original image; restoration is for visualization only

### 🔍 OCR & Text Extraction
- **Engine**: Gemini 3 Pro Preview
- **Input**: Original manuscript image (not restored)
- **Output**: Raw Sanskrit text in Devanagari script

### ✍️ Text Correction
- **Smart correction** of OCR errors
- Preserves matras, ligatures (संयुक्ताक्षर)
- Handles damaged/unclear characters with [?]

### 🌍 Multilingual Translation
1. **English**: Accurate, literal translation
2. **Hindi**: अर्थ सुरक्षित रखते हुए
3. **Kannada**: ಅರ್ಥವನ್ನು ಕಾಪಾಡುವಂತೆ

### 📥 Export Options
- Download restored image (PNG)
- Download analysis results (TXT)
- Copy Sanskrit text to clipboard

---

## 🎯 HOW TO USE

### 1. Access the Application
Open your browser and navigate to one of the URLs above

### 2. Upload Manuscript Image
- Click "Choose a manuscript image..." in the left column
- Supported formats: PNG, JPG, JPEG, BMP
- Image will be displayed immediately

### 3. Configure Settings (Optional)
In the sidebar:
- ✅ Enable/disable image restoration
- 🎚️ Adjust creativity (temperature) for translation

### 4. Process the Manuscript
- Click the **"🔍 Analyze & Translate"** button
- Wait for processing (may take 10-30 seconds)
- AI will analyze using the original image

### 5. View Results
Results will display:
- Extracted Sanskrit text
- Corrected Sanskrit text
- English translation
- Hindi translation (हिन्दी)
- Kannada translation (ಕನ್ನಡ)

### 6. Download (Optional)
- **Restored Image**: Click "📥 Download Restored Image"
- **Text Results**: Click "📥 Download Results"

---

## 🛠️ TECHNICAL DETAILS

### Process ID
```bash
PID: 9415
```

### Log File
```bash
/home/bagesh/EL-project/streamlit_ocr_gemini.log
```

### Configuration
- **API Key**: Configured via environment variable
- **Model Size**: Base (256x256 patches)
- **Device**: Auto-detected (CUDA if available, else CPU)
- **Port**: 8501

---

## 📊 PIPELINE WORKFLOW

```
┌─────────────────────────────────────────────────┐
│         UPLOAD MANUSCRIPT IMAGE                 │
└────────────────┬────────────────────────────────┘
                 │
                 ├─────────────────────┬──────────────────────┐
                 │                     │                      │
                 ▼                     ▼                      ▼
         ┌──────────────┐      ┌──────────────┐     ┌──────────────┐
         │   ORIGINAL   │      │  RESTORATION │     │   ORIGINAL   │
         │    IMAGE     │      │  (Optional)  │     │    IMAGE     │
         │   Display    │      │   Display    │     │  For OCR     │
         └──────────────┘      └──────────────┘     └──────┬───────┘
                                                            │
                                                            ▼
                                                    ┌──────────────┐
                                                    │  GEMINI API  │
                                                    │  OCR + NLP   │
                                                    └──────┬───────┘
                                                           │
         ┌─────────────────────────────────────────────────┴─────┐
         │                                                         │
         ▼                                                         ▼
┌────────────────────┐                                  ┌────────────────────┐
│  TEXT EXTRACTION   │                                  │   TRANSLATION      │
│  • Raw OCR         │                                  │  • English         │
│  • Correction      │                                  │  • Hindi           │
│                    │                                  │  • Kannada         │
└────────────────────┘                                  └────────────────────┘
         │                                                         │
         └────────────────────┬────────────────────────────────────┘
                              │
                              ▼
                     ┌────────────────┐
                     │    DISPLAY     │
                     │    RESULTS     │
                     └────────────────┘
                              │
                              ▼
                     ┌────────────────┐
                     │    EXPORT      │
                     │  • Images      │
                     │  • Text        │
                     └────────────────┘
```

---

## 🔧 MANAGEMENT COMMANDS

### View Logs (Real-time)
```bash
cd /home/bagesh/EL-project
tail -f streamlit_ocr_gemini.log
```

### Stop the Application
```bash
kill 9415
# or
pkill -f "streamlit run ocr_gemini_streamlit.py"
```

### Restart the Application
```bash
cd /home/bagesh/EL-project
source venv/bin/activate
export GEMINI_API_KEY="AIzaSyAxTN1rfdyJQVE3ecy5d8Zqkl5I431nBh0"
streamlit run ocr_gemini_streamlit.py --server.port 8501 --server.headless true
```

### Check Status
```bash
# Check if running
lsof -i :8501

# View process details
ps aux | grep streamlit
```

---

## ⚠️ IMPORTANT NOTES

### Image Processing Strategy
- **OCR always uses the ORIGINAL image** (not the restored version)
- **Restoration is for DISPLAY ONLY** to help users visualize better quality
- This ensures OCR accuracy isn't affected by restoration artifacts

### Why This Approach?
1. **Consistency**: OCR sees exactly what you uploaded
2. **Reliability**: No risk of restoration introducing errors
3. **Flexibility**: Users can see both versions side-by-side
4. **Quality**: Restoration enhances visualization without affecting analysis

### API Key
- Currently using hardcoded API key in the code
- For production, use environment variables only
- Never commit API keys to version control

---

## 🎉 SUCCESS INDICATORS

You'll know the pipeline is working correctly when:
- ✅ Web interface loads without errors
- ✅ Image uploads successfully
- ✅ Restoration produces visually enhanced image (if enabled)
- ✅ "Analyze & Translate" button processes within 30 seconds
- ✅ Sanskrit text is extracted in Devanagari script
- ✅ All three translations appear (English, Hindi, Kannada)
- ✅ Download buttons work for both image and text

---

## 📞 TROUBLESHOOTING

### If the app doesn't load:
1. Check if the process is running: `ps aux | grep streamlit`
2. Check the log file: `tail -50 streamlit_ocr_gemini.log`
3. Verify port 8501 is available: `lsof -i :8501`

### If restoration fails:
- Check if checkpoint exists: `ls -lh checkpoints/kaggle/final_converted.pth`
- View console for error messages
- Disable restoration and use original image only

### If OCR fails:
- Verify Gemini API key is valid
- Check internet connectivity
- View log file for API errors

---

## 📚 FILES INVOLVED

```
ocr_gemini_streamlit.py          # Main Streamlit application
models/vit_restorer.py            # ViT model architecture
utils/image_restoration_enhanced.py  # Enhanced restoration utilities
checkpoints/kaggle/final_converted.pth  # Trained model weights
.env                              # Environment variables (API key)
streamlit_ocr_gemini.log         # Application logs
```

---

## 🎓 NEXT STEPS

### To enhance the pipeline further:
1. Add batch processing for multiple images
2. Implement quality comparison metrics
3. Add Sanskrit text-to-speech
4. Create API endpoint for programmatic access
5. Add user authentication
6. Implement result caching for faster re-processing

---

**Last Updated**: December 25, 2025
**Status**: ✅ RUNNING
**Port**: 8501
**PID**: 9415

---

🔰 **Quality-Guarded Manuscript Vision Pipeline**
*ViT Restoration + Gemini API for Sanskrit Manuscripts*

