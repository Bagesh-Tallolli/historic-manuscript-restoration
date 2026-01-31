# 🎯 Complete Sanskrit Manuscript OCR Project - Integration Guide

## Project Overview

This project combines **AI-powered image restoration** with **advanced OCR and translation** to digitize and translate Sanskrit manuscripts. The complete pipeline includes:

1. **Image Restoration**: Trained ViT model to enhance degraded manuscripts
2. **OCR & Translation**: Gemini AI for text extraction and multi-language translation
3. **Interactive Interface**: Streamlit web application

---

## 🚀 Quick Start (Complete Workflow)

### Option 1: Run the Enhanced OCR App (Recommended)

```bash
# Navigate to project directory
cd /home/bagesh/EL-project

# Run the enhanced OCR application
./run_enhanced_ocr.sh
```

This will:
- ✅ Check and install dependencies
- ✅ Verify restoration model checkpoint
- ✅ Start the web application on http://localhost:8501

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r ocr_gemini_streamlit_requirements.txt

# Run application
streamlit run ocr_gemini_streamlit.py
```

---

## 📊 Complete Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPLETE OCR PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Input Image    │  ← Upload Sanskrit manuscript
│  (Any format)   │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  STAGE 1: Image Restoration          │
│  ─────────────────────────────────   │
│  • Load ViT Restoration Model        │
│  • Detect image size                 │
│  • Apply patch-based processing      │
│    (for large images)                │
│  • Enhance contrast & sharpness      │
│  • Remove noise & artifacts          │
└────────┬─────────────────────────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────┐   ┌────────────────┐
│  Original   │   │  Restored      │
│  Image      │   │  Image         │
└─────────────┘   └────────┬───────┘
                           │
                           ▼
         ┌──────────────────────────────────────┐
         │  STAGE 2: OCR & Translation          │
         │  ─────────────────────────────────   │
         │  • Send to Gemini AI                 │
         │  • Extract Sanskrit text             │
         │  • Correct & reconstruct verses      │
         │  • Translate to:                     │
         │    - Hindi                           │
         │    - English                         │
         │    - Kannada                         │
         └────────┬─────────────────────────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │  STAGE 3: Output     │
         │  ─────────────────── │
         │  • Display results   │
         │  • Download options  │
         │    - Restored image  │
         │    - Text results    │
         └──────────────────────┘
```

---

## 🔧 Project Components

### 1. Image Restoration Model

**Location**: `models/vit_restorer.py`
**Checkpoint**: `checkpoints/kaggle/final_converted.pth`

**Features**:
- Vision Transformer architecture
- Patch-based processing for high resolution
- Post-processing enhancements
- GPU acceleration support

**Training**:
- Trained on Kaggle using historical manuscript dataset
- See `KAGGLE_TRAINING_GUIDE.md` for training details

### 2. OCR Application

**Location**: `ocr_gemini_streamlit.py`
**Dependencies**: `ocr_gemini_streamlit_requirements.txt`

**Features**:
- Streamlit web interface
- Real-time image restoration
- Multi-language translation
- Download options for results

### 3. Utilities

**Image Restoration**: `utils/image_restoration_enhanced.py`
- Enhanced restoration class
- Patch blending algorithms
- Post-processing filters

---

## 📝 Usage Instructions

### Step 1: Start the Application

```bash
./run_enhanced_ocr.sh
```

The browser will open automatically at `http://localhost:8501`

### Step 2: Upload Manuscript

1. Click "Choose a manuscript image..."
2. Select your Sanskrit manuscript (PNG, JPG, JPEG, BMP)
3. Image appears in the left column

### Step 3: Configure Settings (Sidebar)

**Image Restoration**:
- ✅ **Enable** for: Degraded, faded, damaged, or noisy manuscripts
- ❌ **Disable** for: Already clear, high-quality scans

**Temperature** (0.0 - 1.0):
- **Low (0.0-0.3)**: Literal, conservative translations
- **Medium (0.3-0.6)**: Balanced approach
- **High (0.6-1.0)**: Creative, interpretive translations

### Step 4: Process Image

1. Click "🔍 Analyze & Translate"
2. Wait for processing (2-10 seconds depending on image size)
3. View results:
   - Restored image (if enabled) in right column
   - Sanskrit text extraction
   - Hindi translation
   - English translation
   - Kannada translation

### Step 5: Download Results

- **Restored Image**: Click "📥 Download Restored Image" (PNG format)
- **Text Results**: Click "📥 Download Results" (TXT format)

---

## ⚙️ Configuration Options

### Model Checkpoints

Edit in `ocr_gemini_streamlit.py`:

```python
RESTORATION_CHECKPOINT_PATHS = [
    "checkpoints/kaggle/final_converted.pth",  # Primary
    "checkpoints/kaggle/final.pth",            # Fallback 1
    "models/trained_models/final.pth",         # Fallback 2
]
```

### Model Size

```python
RESTORATION_MODEL_SIZE = "base"  # Options: tiny, small, base, large
RESTORATION_IMG_SIZE = 256       # Patch size for processing
```

### API Configuration

```python
GEMINI_API_KEY = "your-api-key-here"
DEFAULT_MODEL = "gemini-3-pro-preview"
```

---

## 🎓 Training Your Own Model

If you want to train a custom restoration model:

### 1. Prepare Dataset

```bash
# Download or prepare your manuscript dataset
# Structure:
# dataset/
#   ├── train/
#   │   ├── degraded/
#   │   └── clean/
#   └── val/
#       ├── degraded/
#       └── clean/
```

### 2. Train on Kaggle

1. Upload `kaggle_training_notebook.py` to Kaggle
2. Enable GPU accelerator
3. Run training (see `KAGGLE_TRAINING_GUIDE.md`)
4. Download checkpoint

### 3. Convert Checkpoint (if needed)

```bash
python convert_kaggle_checkpoint.py \
  --input checkpoints/kaggle/final.pth \
  --output checkpoints/kaggle/final_converted.pth
```

### 4. Use in Application

Place converted checkpoint in `checkpoints/kaggle/` directory

---

## 🔍 Troubleshooting

### Issue: Restoration Model Not Found

**Symptom**: Warning message in application

**Solution**:
1. Check if checkpoint exists in configured paths
2. Train model using Kaggle workflow
3. Download and place checkpoint in correct location

### Issue: CUDA Out of Memory

**Symptom**: Error during restoration

**Solutions**:
- Reduce image size before upload
- Disable restoration for very large images
- Use CPU mode (automatic fallback)

### Issue: Poor OCR Results

**Solutions**:
1. Enable image restoration
2. Adjust temperature setting
3. Crop image to focus on text
4. Ensure good lighting in original

### Issue: Slow Processing

**Normal Processing Times**:
- Small images (<512px): 1-2 seconds
- Medium images (512-1024px): 3-5 seconds  
- Large images (>1024px): 5-15 seconds

**Optimizations**:
- Use GPU if available
- Disable restoration for clear images
- Resize very large images

---

## 📚 File Structure

```
EL-project/
├── ocr_gemini_streamlit.py              # Main application
├── run_enhanced_ocr.sh                  # Startup script
├── ocr_gemini_streamlit_requirements.txt # Dependencies
├── ENHANCED_OCR_README.md               # Detailed docs
├── COMPLETE_PROJECT_GUIDE.md            # This file
│
├── models/
│   ├── vit_restorer.py                  # Model architecture
│   └── trained_models/
│       └── final.pth                    # Checkpoint (fallback)
│
├── utils/
│   └── image_restoration_enhanced.py    # Restoration utilities
│
├── checkpoints/
│   └── kaggle/
│       ├── final_converted.pth          # Primary checkpoint
│       └── final.pth                    # Original checkpoint
│
└── data/                                # Sample images (optional)
```

---

## 🎯 Common Workflows

### Workflow 1: Quick OCR (Clear Images)

1. Start application: `./run_enhanced_ocr.sh`
2. Upload image
3. **Disable** restoration checkbox
4. Click "Analyze & Translate"
5. Download results

**Time**: ~2-3 seconds

### Workflow 2: Complete Restoration + OCR (Degraded Images)

1. Start application: `./run_enhanced_ocr.sh`
2. Upload image
3. **Enable** restoration checkbox
4. Click "Analyze & Translate"
5. Compare original vs restored
6. Download both restored image and results

**Time**: ~5-15 seconds (depending on image size)

### Workflow 3: Batch Processing (Multiple Images)

Currently manual process:
1. Process first image
2. Download results
3. Refresh page or upload next image
4. Repeat

*Future enhancement: Batch processing feature*

---

## 🚀 Performance Tips

### For Best Results

1. **Image Quality**:
   - Upload highest resolution available
   - Ensure good lighting
   - Avoid shadows and glare

2. **Restoration**:
   - Enable for old/degraded manuscripts
   - Disable for modern clear scans
   - Check restored preview before OCR

3. **Translation**:
   - Adjust temperature based on content type
   - Poetry: Higher temperature (0.4-0.6)
   - Technical: Lower temperature (0.1-0.3)

### Hardware Recommendations

**Minimum**:
- CPU: Dual-core processor
- RAM: 4GB
- Storage: 2GB free space

**Recommended**:
- CPU: Quad-core processor
- GPU: NVIDIA GPU with 4GB+ VRAM
- RAM: 8GB+
- Storage: 5GB free space

---

## 📖 Additional Resources

- **Training Guide**: `KAGGLE_TRAINING_GUIDE.md`
- **Model Details**: `KAGGLE_MODEL_INTEGRATION.md`
- **API Setup**: `GOOGLE_API_SETUP_COMPLETE.md`
- **Enhanced OCR**: `ENHANCED_OCR_README.md`

---

## ✅ Project Completion Checklist

- [x] Image restoration model integrated
- [x] OCR pipeline connected
- [x] Web interface implemented
- [x] Download functionality added
- [x] Documentation created
- [x] Startup script configured
- [x] Error handling implemented
- [x] Performance optimized

---

## 🎉 Success!

Your enhanced Sanskrit OCR system is now complete with:

✅ AI-powered image restoration
✅ High-accuracy OCR
✅ Multi-language translation  
✅ User-friendly interface
✅ Production-ready deployment

**Start using it now**:
```bash
./run_enhanced_ocr.sh
```

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review error messages in application
3. Verify checkpoint paths and dependencies
4. Check logs in terminal output

---

**Last Updated**: November 30, 2025
**Project Status**: ✅ Complete and Ready to Use

