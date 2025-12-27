# 📦 Installation Guide - Fixed Dependencies

## ✅ Problem Solved

The repository previously had a dependency conflict between `googletrans==4.0.0rc1` and `roboflow>=1.1.0` that prevented successful installation of requirements.

### Issue Details

- `googletrans==4.0.0rc1` depends on `httpx==0.13.3`
- `httpx==0.13.3` requires old versions: `idna==2.*`, `chardet==3.*`
- `roboflow>=1.1.0` requires newer versions: `idna==3.7`, `chardet==4.0.0`
- This created an unresolvable conflict

### Solution Implemented

✅ **Replaced `googletrans==4.0.0rc1` with `deep-translator>=1.11.0`**

`deep-translator` is a modern, actively maintained translation library that:
- Has minimal dependencies (only `beautifulsoup4` and `requests`)
- Is fully compatible with `roboflow` and other project dependencies
- Provides the same Google Translate functionality
- Supports multiple translation backends (Google, DeepL, Microsoft, etc.)

---

## 🚀 Installation Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/Bagesh-Tallolli/historic-manuscript-restoration.git
cd historic-manuscript-restoration
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will now install successfully without any dependency conflicts!

### Step 5: Install System Dependencies

For OCR functionality, install Tesseract:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-san tesseract-ocr-hin
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### Step 6: Verify Installation

```bash
python test_setup.py
```

You should see all checks pass ✓

---

## 💡 What Changed?

### 1. Updated `requirements.txt`

**Before:**
```
googletrans==4.0.0rc1
```

**After:**
```
deep-translator>=1.11.0
```

### 2. Updated `nlp/translation.py`

The translation module now uses `deep-translator` instead of `googletrans`:

**Key Changes:**
- Import: `from deep_translator import GoogleTranslator`
- API: `GoogleTranslator(source='sa', target='en').translate(text)`
- No functional changes to the pipeline - same translation quality

---

## 🧪 Testing the Fix

### Test 1: Import Test

```python
from deep_translator import GoogleTranslator
from roboflow import Roboflow

print("✓ Both packages work together!")
```

### Test 2: Translation Test

```python
from nlp.translation import SanskritTranslator

translator = SanskritTranslator(method='google')
result = translator.translate("रामः वनं गच्छति")
print(f"Translation: {result}")
```

### Test 3: Full Pipeline Test

```bash
python main.py --image_path data/datasets/samples/test_sample.png
```

---

## 📚 Additional Notes

### For Minimal Installation

If you only need specific functionality, install minimal dependencies:

**For Translation Only:**
```bash
pip install deep-translator transformers torch
```

**For OCR Only:**
```bash
pip install pytesseract opencv-python Pillow numpy
```

**For Image Restoration Only:**
```bash
pip install torch torchvision timm opencv-python numpy
```

### For CPU-Only Installation

If you don't have a GPU, install CPU-only PyTorch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'deep_translator'"

**Solution:** The installation didn't complete. Try:
```bash
pip install deep-translator
```

### Issue: Still getting dependency conflicts

**Solution:** Clean your pip cache and try again:
```bash
pip cache purge
pip install -r requirements.txt
```

### Issue: Out of disk space during installation

**Solution:** PyTorch and other deep learning libraries are large. Free up space:
```bash
# Clean pip cache
pip cache purge

# Clean apt cache (Linux)
sudo apt-get clean

# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

---

## ✨ Benefits of This Fix

✅ **No more dependency conflicts**  
✅ **Faster installation**  
✅ **More reliable and maintained library**  
✅ **Same translation quality**  
✅ **Compatible with all project dependencies**  
✅ **Supports multiple translation backends**  

---

## 🔗 Resources

- **deep-translator documentation**: https://deep-translator.readthedocs.io/
- **Project README**: README.md
- **Getting Started**: GETTING_STARTED.md

---

**Installation issue fixed! The model can now execute without errors. 🎉**
