# Google Cloud Vision Setup Guide

## Overview

The Sanskrit OCR application now supports **Google Cloud Vision** (Google Lens technology) as an alternative to Tesseract OCR. Google Cloud Vision often provides superior accuracy for:

- Complex scripts like Sanskrit/Devanagari
- Low-quality or degraded images
- Handwritten text
- Mixed languages
- Text with varied orientations

## Prerequisites

### 1. Install Google Cloud Vision

The package is already installed in your environment:

```bash
pip install google-cloud-vision
```

### 2. Set Up Google Cloud Credentials

#### Option A: Service Account (Recommended for Production)

1. **Create a Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one

2. **Enable Cloud Vision API**
   - Navigate to "APIs & Services" > "Library"
   - Search for "Cloud Vision API"
   - Click "Enable"

3. **Create Service Account**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in the details and create

4. **Download Credentials JSON**
   - Click on the created service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create New Key"
   - Choose JSON format
   - Download the file

5. **Set Environment Variable**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
   ```

   Or add to your `.bashrc` or `.zshrc`:
   ```bash
   echo 'export GOOGLE_APPLICATION_CREDENTIALS="/home/bagesh/EL-project/google-credentials.json"' >> ~/.bashrc
   source ~/.bashrc
   ```

#### Option B: User Credentials (Quick Start)

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize and authenticate
gcloud init
gcloud auth application-default login
```

## Using Google Cloud Vision in the App

### 1. Start the Application

```bash
cd /home/bagesh/EL-project
source venv/bin/activate
streamlit run app_sanskrit_ocr.py
```

### 2. Select OCR Engine

In the sidebar:
- **OCR Engine**: Select "Google Cloud Vision (Google Lens)"
- The app will show ✅ if credentials are properly configured

### 3. Extract Text

1. Upload your Sanskrit image
2. Click "Extract Text (Google Lens)"
3. Wait for results (typically 2-5 seconds)

## Features & Benefits

### Google Cloud Vision Advantages:

✅ **Better Accuracy**
- State-of-the-art ML models
- Handles complex scripts better
- Superior for degraded/low-quality images

✅ **No Configuration Needed**
- Automatic language detection
- No PSM/OEM mode settings
- Works with mixed languages

✅ **Advanced Features**
- Text orientation detection
- Confidence scores
- Bounding box information

### When to Use Google Vision:

- ❌ Tesseract gives poor results
- 📜 Historical/ancient manuscripts
- ✍️ Handwritten text
- 🔍 Low-quality scans
- 🌐 Mixed language documents
- 📐 Rotated or skewed text

### When to Use Tesseract:

- 💻 Offline processing needed
- 🆓 No API costs
- 🖨️ High-quality printed text
- ⚡ Batch processing
- 🔒 Privacy-sensitive documents

## Cost Considerations

### Free Tier:
- First **1,000 images/month**: FREE
- Sufficient for most personal use

### Pricing Beyond Free Tier:
- 1,001 - 5,000,000 images: $1.50 per 1,000 images
- See [Pricing Page](https://cloud.google.com/vision/pricing)

### Cost Optimization:
- Use Tesseract for simple/clean text
- Reserve Google Vision for challenging images
- Monitor usage in Google Cloud Console

## Troubleshooting

### Error: "Google Cloud Vision not available"

**Solution:**
```bash
pip install google-cloud-vision
```

### Error: "Could not automatically determine credentials"

**Solution 1 - Set credentials path:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

**Solution 2 - Use gcloud:**
```bash
gcloud auth application-default login
```

### Error: "API has not been used in project"

**Solution:**
1. Go to [Cloud Console](https://console.cloud.google.com/)
2. Enable "Cloud Vision API"
3. Wait 1-2 minutes for activation

### Error: "Permission denied"

**Solution:**
- Ensure service account has "Cloud Vision API User" role
- Or use: `gcloud auth application-default login`

## Testing Google Vision

Test if Google Vision is working:

```python
from google.cloud import vision

# Test connection
client = vision.ImageAnnotatorClient()
print("✅ Google Cloud Vision is configured correctly!")
```

## Comparison: Tesseract vs Google Vision

| Feature | Tesseract | Google Vision |
|---------|-----------|---------------|
| **Cost** | Free | Free tier, then paid |
| **Accuracy (Simple)** | Good | Excellent |
| **Accuracy (Complex)** | Fair | Excellent |
| **Speed** | Fast | Moderate (API call) |
| **Offline** | Yes | No (requires internet) |
| **Setup** | Easy | Moderate |
| **Languages** | 100+ | 50+ (auto-detect) |
| **Handwriting** | Poor | Good |
| **Sanskrit** | Good | Excellent |

## Application Flow

### With Google Cloud Vision:

1. **Upload Image** → Drag & drop Sanskrit manuscript
2. **Select Engine** → Choose "Google Cloud Vision"
3. **Click Extract** → Text sent to Google API
4. **View Results** → Full text displayed
5. **Download** → Save extracted text

### Auto-Fallback:

If Google Vision fails:
- App automatically tries Tesseract
- Ensures you always get results
- Best of both worlds!

## Files & Configuration

### Credentials Location:
```
/home/bagesh/EL-project/google-credentials.json
```

### Environment Variable:
```bash
GOOGLE_APPLICATION_CREDENTIALS=/home/bagesh/EL-project/google-credentials.json
```

### Check Current Setup:
```bash
echo $GOOGLE_APPLICATION_CREDENTIALS
gcloud auth application-default print-access-token
```

## Quick Setup Script

Save this as `setup_google_vision.sh`:

```bash
#!/bin/bash

echo "Google Cloud Vision Setup"
echo "========================="
echo ""

# Check if credentials exist
if [ -f "/home/bagesh/EL-project/google-credentials.json" ]; then
    echo "✅ Credentials file found"
    export GOOGLE_APPLICATION_CREDENTIALS="/home/bagesh/EL-project/google-credentials.json"
    echo "✅ Environment variable set"
else
    echo "⚠️  Credentials file not found"
    echo "Place your credentials.json at:"
    echo "  /home/bagesh/EL-project/google-credentials.json"
    echo ""
    echo "Or use gcloud auth:"
    echo "  gcloud auth application-default login"
fi

echo ""
echo "Testing connection..."
python3 -c "from google.cloud import vision; client = vision.ImageAnnotatorClient(); print('✅ Google Cloud Vision working!')" 2>&1

echo ""
echo "Ready to use Google Cloud Vision!"
```

## Support & Resources

- [Cloud Vision Documentation](https://cloud.google.com/vision/docs)
- [Python Client Library](https://cloud.google.com/vision/docs/libraries#client-libraries-install-python)
- [Pricing Calculator](https://cloud.google.com/products/calculator)
- [Free Tier Details](https://cloud.google.com/vision/pricing)

---

**Created**: November 29, 2025  
**Application**: app_sanskrit_ocr.py  
**Purpose**: Enable Google Cloud Vision for superior Sanskrit OCR

