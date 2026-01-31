# 🚀 QUICK START - BOTH APPS RUNNING

## ✅ TWO STREAMLIT APPS ARE ACTIVE

---

## 📱 APP 1: OCR GEMINI (Port 8501)
**Purpose**: ViT Restoration + OCR + Translation

### Access:
```
http://localhost:8501
```

### Features:
- ✨ **ViT Image Restoration** (trained AI model)
- 🔍 **Sanskrit OCR** (Gemini API)
- 🌍 **Multi-language Translation** (English, Hindi, Kannada)
- 📊 **Quality Comparison** (original vs restored)
- 📥 **Export Results**

### Process:
```
PID: 9415
Status: ✅ RUNNING
```

---

## 📱 APP 2: IMAGE POLISH (Port 8502) ⭐ NEW
**Purpose**: Direct Gemini API Image Polishing

### Access:
```
http://localhost:8502
```

### Features:
- 📤 **Simple Upload** (any image format)
- 🎨 **7 Enhancement Types**
  - General Polish
  - Restore Old Photos
  - Enhance Documents
  - Colorize B&W
  - Remove Noise
  - Sharpen & Clarify
  - Professional Edit
- ⚡ **Direct API** (no local model)
- 📥 **Download Enhanced Image**

### Process:
```
PID: 14854
Status: ✅ RUNNING
```

---

## 🎯 WHICH APP TO USE?

### Use Image Polish (8502) for:
- Quick experiments
- Testing Gemini API
- Simple interface

### Use OCR Gemini (8501) for: ⭐ RECOMMENDED
- **Reliable image restoration**
- Sanskrit manuscripts
- OCR + translation
- Production use

---

## 🛠️ MANAGEMENT

### Stop Both Apps:
```bash
pkill -f streamlit
```

### Restart App 1 (OCR Gemini):
```bash
cd /home/bagesh/EL-project
./start_ocr_gemini_pipeline.sh
```

### Restart App 2 (Image Polish):
```bash
cd /home/bagesh/EL-project
./start_image_polish.sh
```

### Check Status:
```bash
lsof -i :8501    # OCR Gemini
lsof -i :8502    # Image Polish
```

---

## 📚 DOCUMENTATION

- **OCR Gemini Guide**: `OCR_GEMINI_PIPELINE_RUNNING.md`
- **Image Polish Guide**: `IMAGE_POLISH_GEMINI_GUIDE.md`

---

## ⚡ QUICK COMMANDS

```bash
# View OCR Gemini logs
tail -f streamlit_ocr_gemini.log

# View Image Polish logs
tail -f streamlit_image_polish.log

# Check all Streamlit processes
ps aux | grep streamlit | grep -v grep
```

---

**Status**: ✅ BOTH APPS RUNNING
**Date**: December 25, 2025

