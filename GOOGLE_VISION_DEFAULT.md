# Google Cloud Vision - Now Default OCR Engine

## ✅ What Changed

**Google Cloud Vision (Google Lens) is now the DEFAULT OCR engine** in the Sanskrit OCR application.

### Before:
- Tesseract OCR was the default
- Google Cloud Vision was optional

### After (Now):
- ⭐ **Google Cloud Vision is the default** (when available)
- Tesseract OCR is the fallback option
- User can switch between them

## Why This Change?

### Superior Accuracy
Google Cloud Vision provides:
- ✅ **Better accuracy** for Sanskrit/Devanagari script
- ✅ **Better handling** of degraded or low-quality images
- ✅ **Automatic language detection** - no configuration needed
- ✅ **Works with handwritten text** better than Tesseract
- ✅ **State-of-the-art ML models** from Google

### User Experience
- 🎯 Most users get best results immediately
- 🚀 No need to configure PSM/OEM settings
- ⚡ Faster setup for new users
- 🔄 Can still switch to Tesseract if needed

## How to Use

### Option 1: Use Default (Google Cloud Vision) ⭐

**If you have Google Cloud credentials set up:**

1. Start the app:
   ```bash
   cd /home/bagesh/EL-project
   source venv/bin/activate
   streamlit run app_sanskrit_ocr.py
   ```

2. Upload your Sanskrit image

3. The OCR Engine will be **"Google Cloud Vision (Google Lens)"** by default

4. Click **"Extract Text (Google Lens)"**

5. Done! ✅

### Option 2: Switch to Tesseract OCR

**If you prefer offline processing or don't have Google credentials:**

1. Start the app (same as above)

2. Upload your image

3. In sidebar, **change OCR Engine** to **"Tesseract OCR"**

4. Configure Tesseract settings if needed

5. Click **"Extract Text (Tesseract)"**

## Setting Up Google Cloud Vision

### Quick Setup:

```bash
# Set your credentials (one-time setup)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"

# Or add to your .bashrc for permanent setup
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/home/bagesh/EL-project/google-credentials.json"' >> ~/.bashrc
source ~/.bashrc
```

### Detailed Setup:

See the complete guide: `GOOGLE_VISION_SETUP.md`

## When to Use Each Engine

### Use Google Cloud Vision (Default) when:
- ✅ You want best accuracy
- ✅ Working with degraded/old manuscripts
- ✅ Have handwritten text
- ✅ Mixed languages in image
- ✅ Don't want to configure settings

### Use Tesseract OCR when:
- ✅ Need offline processing
- ✅ Privacy-sensitive documents
- ✅ Want full control over settings
- ✅ Batch processing many images
- ✅ Don't have Google credentials

## Cost Consideration

### Google Cloud Vision:
- **FREE**: First 1,000 images per month
- **Paid**: $1.50 per 1,000 images after that

For most users, the free tier is sufficient!

### Tesseract OCR:
- **FREE**: Always, unlimited

## Application Behavior

### If Google Cloud Vision is Available:
```
Default OCR Engine: Google Cloud Vision ⭐
Fallback: Tesseract OCR
```

### If Google Cloud Vision is NOT Available:
```
Default OCR Engine: Tesseract OCR
Only Option: Tesseract OCR
```

The app automatically detects availability and sets the best default!

## Checking Your Setup

### Test if Google Cloud Vision is working:

```bash
cd /home/bagesh/EL-project
source venv/bin/activate
python3 -c "from google.cloud import vision; client = vision.ImageAnnotatorClient(); print('✅ Google Cloud Vision is ready!')"
```

### If you see an error:

1. **"google.cloud not found"**
   ```bash
   pip install google-cloud-vision
   ```

2. **"Could not determine credentials"**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   ```

3. **Need credentials file?**
   - See `GOOGLE_VISION_SETUP.md` for complete setup

## UI Changes

### In the Sidebar:

**When Google Vision is default:**
```
✅ Using Google Cloud Vision (Recommended)
Advanced Google Lens technology for superior accuracy
```

**When Tesseract is selected:**
```
💡 Google Cloud Vision is available - switch to it for better accuracy!
```

**When Google Vision is not available:**
```
⚠️ Google Cloud Vision not available
[How to enable] (expandable section)
```

### Extract Button:

- Google Vision: **"🔍 Extract Text (Google Lens)"**
- Tesseract: **"🔍 Extract Text (Tesseract)"** + **"🔄 Auto-Retry"**

## Summary

🎉 **Google Cloud Vision is now the default OCR engine!**

**This means:**
- ✅ Better accuracy out of the box
- ✅ Less configuration needed
- ✅ Best results for most users
- ✅ Can still use Tesseract when needed

**To use:**
1. Set up Google credentials (one-time)
2. Run the app
3. Upload and extract - it just works!

---

**Updated**: November 29, 2025  
**Application**: app_sanskrit_ocr.py  
**Status**: ✅ Google Cloud Vision is now default

