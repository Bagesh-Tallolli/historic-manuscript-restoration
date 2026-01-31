# 🔐 API KEYS AND LIBRARIES REFERENCE

## Complete Guide to All Google APIs and Libraries Used in the Project

**Last Updated:** November 27, 2025

---

## 📚 LIBRARIES USED IN THE PROJECT

### 1. **Google Cloud Vision API** (for OCR - Google Lens)

**File:** `sanskrit_ocr_agent.py`

**Library:**
```python
from google.cloud import vision
```

**Installation:**
```bash
pip install google-cloud-vision
```

**API Key Type:** **JSON Credentials File** (NOT a simple API key)

**Environment Variable:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

**Usage in Code:**
```python
# In sanskrit_ocr_agent.py, line 149
if credentials_path and Path(credentials_path).exists():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

# Initialize client
self.vision_client = vision.ImageAnnotatorClient()
```

**How to Get Credentials:**
1. Go to: https://console.cloud.google.com/
2. Create a project (or select existing)
3. Enable "Cloud Vision API"
4. Go to "Credentials" → "Create Credentials" → "Service Account"
5. Create service account with "Cloud Vision API User" role
6. Click on the service account → "Keys" → "Add Key" → "JSON"
7. Download the JSON file
8. Save it securely (e.g., `~/.gcloud/vision-credentials.json`)
9. Set environment variable pointing to this file

**Cost:**
- First 1,000 units/month: FREE
- After that: ~$1.50 per 1,000 units
- See: https://cloud.google.com/vision/pricing

---

### 2. **Gemini API** (for Text Correction)

**File:** `sanskrit_ocr_agent.py`

**Library:**
```python
import google.generativeai as genai
```

**Installation:**
```bash
pip install google-generativeai
```

**API Key Type:** **Simple API Key String**

**Environment Variable:**
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

**Usage in Code:**
```python
# In sanskrit_ocr_agent.py, line 169
key = api_key or os.getenv('GEMINI_API_KEY')

if not key:
    print("[STAGE 3] ⚠ No Gemini API key - correction disabled")
    self.gemini_model = None
    return

try:
    genai.configure(api_key=key)
    self.gemini_model = genai.GenerativeModel('gemini-pro')
    print("[STAGE 3] ✓ Gemini correction ready")
```

**How to Get API Key:**
1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key (starts with "AIza...")
4. Save it securely
5. Set environment variable

**Cost:**
- Free tier: 60 requests per minute
- Paid tier available
- See: https://ai.google.dev/pricing

---

### 3. **Google Translate Library** (NOT USED IN STRICT MODE)

**File:** `nlp/translation.py` (OLD implementation, not used by strict agent)

**Library:**
```python
from googletrans import Translator as GoogleTranslator
```

**Installation:**
```bash
pip install googletrans==4.0.0-rc1
```

**API Key:** **NONE REQUIRED** (uses unofficial free API)

**Note:** This is from the OLD translation.py file and is **NOT used** in the strict pipeline. The strict agent uses **mBART** instead.

---

### 4. **mBART Translation Model** (USED IN STRICT MODE)

**File:** `sanskrit_ocr_agent.py`

**Library:**
```python
from transformers import MarianMTModel, MarianTokenizer
# Actually loads: facebook/mbart-large-50-many-to-many-mmt
```

**Installation:**
```bash
pip install transformers torch
```

**API Key:** **NONE REQUIRED** (local model from HuggingFace)

**Usage in Code:**
```python
# In sanskrit_ocr_agent.py, line 183
self.tokenizer = MarianTokenizer.from_pretrained(model_name)
self.translation_model = MarianMTModel.from_pretrained(model_name)
self.translation_model.to(self.device)
self.translation_model.eval()
```

**Model will auto-download from HuggingFace on first use (~1.2GB)**

---

## 🔑 WHERE API KEYS ARE USED

### **Strict Sanskrit Agent** (`sanskrit_ocr_agent.py`)

This is the **main agent** following strict pipeline requirements.

**Two API Keys:**

1. **Google Cloud Vision Credentials** (JSON file)
   - **Purpose:** OCR text extraction (Google Lens)
   - **Environment Variable:** `GOOGLE_APPLICATION_CREDENTIALS`
   - **Command Line:** `--google-creds /path/to/credentials.json`
   - **Required:** Optional (uses Tesseract fallback if not set)
   - **Line in code:** 149

2. **Gemini API Key** (string)
   - **Purpose:** Sanskrit text correction
   - **Environment Variable:** `GEMINI_API_KEY`
   - **Command Line:** `--gemini-key YOUR_KEY`
   - **Required:** Optional (skips correction if not set)
   - **Line in code:** 169

### **Old Translation Module** (`nlp/translation.py`)

This is the **OLD implementation** and is **NOT used** by the strict agent.

**Libraries:**
- `googletrans` - Unofficial Google Translate (no API key needed)
- `transformers` - For IndicTrans2 or mBART models (no API key needed)

**Note:** The strict agent does NOT use this file for translation.

---

## 📝 HOW TO SET API KEYS

### Method 1: Environment Variables (Recommended)

```bash
# In your .bashrc or .zshrc
export GOOGLE_APPLICATION_CREDENTIALS="/home/user/.gcloud/vision-credentials.json"
export GEMINI_API_KEY="AIzaSy...your_key_here"

# Reload shell
source ~/.bashrc
```

### Method 2: Command Line Arguments

```bash
python3 sanskrit_ocr_agent.py manuscript.jpg \
    --google-creds /path/to/credentials.json \
    --gemini-key "AIzaSy...your_key_here" \
    --output results/
```

### Method 3: In Streamlit Web UI

1. Open http://localhost:8501
2. Look at the sidebar under "API Configuration"
3. Enter paths/keys in the text boxes
4. Click "Process Manuscript"

**Sidebar inputs:**
```python
# In app_professional.py
google_creds = st.sidebar.text_input(
    "Google Cloud Credentials Path",
    value=os.getenv('GOOGLE_APPLICATION_CREDENTIALS', ''),
    placeholder="/path/to/credentials.json"
)

gemini_key = st.sidebar.text_input(
    "Gemini API Key",
    value=os.getenv('GEMINI_API_KEY', ''),
    type="password",
    placeholder="Enter your Gemini API key"
)
```

---

## 🚫 WHAT IS NOT USED (IMPORTANT!)

### Libraries NOT Used in Strict Mode:

1. ❌ **`googletrans`** - Old Google Translate library
   - File: `nlp/translation.py`
   - Status: **NOT USED** by strict agent
   - Reason: Strict agent uses mBART instead

2. ❌ **Google Translate API** (paid service)
   - NOT used anywhere
   - Would require: `from google.cloud import translate_v2`
   - Strict agent uses **mBART** for translation

3. ❌ **Tesseract OCR** (as primary)
   - Only used as **fallback** if Google Vision fails
   - File: `sanskrit_ocr_agent.py`, line 268

4. ❌ **TrOCR** or other OCR engines
   - NOT used in strict pipeline

---

## 📊 SUMMARY TABLE

| Component | Library | API Key Type | Environment Variable | Required? |
|-----------|---------|--------------|---------------------|-----------|
| **OCR** | `google.cloud.vision` | JSON credentials file | `GOOGLE_APPLICATION_CREDENTIALS` | Optional* |
| **Correction** | `google.generativeai` | String key | `GEMINI_API_KEY` | Optional* |
| **Translation** | `transformers` (mBART) | None (local model) | None | Required |
| **Restoration** | `torch` (ViT model) | None (local model) | None | Required |

\* Optional but highly recommended for best accuracy. Falls back to Tesseract/skips if not set.

---

## 🔒 SECURITY BEST PRACTICES

### 1. **Never Commit API Keys to Git**

Add to `.gitignore`:
```
# API credentials
*.json
.env
credentials/
```

### 2. **Store Credentials Securely**

```bash
# Good location
~/.gcloud/vision-credentials.json

# Set restrictive permissions
chmod 600 ~/.gcloud/vision-credentials.json
```

### 3. **Use Environment Variables**

```bash
# In ~/.bashrc
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.gcloud/vision-credentials.json"
export GEMINI_API_KEY="$(cat $HOME/.secrets/gemini_key.txt)"
```

### 4. **Rotate Keys Regularly**

- Gemini keys: Can be regenerated at https://makersuite.google.com/app/apikey
- Google Cloud: Create new service account and delete old one

---

## 🧪 TEST YOUR API KEYS

### Test Google Cloud Vision:

```bash
cd /home/bagesh/EL-project
source venv/bin/activate

python3 -c "
from google.cloud import vision
import os
print('Credentials:', os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
client = vision.ImageAnnotatorClient()
print('✅ Google Cloud Vision API working!')
"
```

### Test Gemini API:

```bash
python3 -c "
import google.generativeai as genai
import os
key = os.getenv('GEMINI_API_KEY')
print('Key set:', 'YES' if key else 'NO')
genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-pro')
print('✅ Gemini API working!')
"
```

### Test Complete Agent:

```bash
python3 test_strict_agent.py
```

---

## 📖 HARDCODED API KEYS IN PROJECT (ROBOFLOW)

**⚠️ WARNING:** There are some hardcoded Roboflow API keys in old files:

**Files with hardcoded keys:**
1. `kaggle_complete_training.py` - Line 44: `API_KEY = "EBJvHlgSWLyW1Ir6ctkH"`
2. `build_notebook.py` - Line 135: `ROBOFLOW_API_KEY = "EBJvHlgSWLyW1Ir6ctkH"`
3. `create_final_notebook.py` - Line 76: `API_KEY = "EBJvHlgSWLyW1Ir6ctkH"`

**Purpose:** These are for dataset download from Roboflow (training data)

**Not used by:** The strict agent (only uses trained models)

**Recommendation:** Move these to environment variables if still needed

---

## ✅ QUICK START (NO API KEYS)

The application works **NOW** without any API keys using fallback methods:

1. **OCR:** Falls back to Tesseract (local, free, lower accuracy)
2. **Correction:** Skipped (uses raw OCR output)
3. **Translation:** Uses local mBART model (no API needed)
4. **Restoration:** Uses local ViT model (no API needed)

**To run without API keys:**
```bash
cd /home/bagesh/EL-project
source venv/bin/activate
streamlit run app_professional.py
```

**To get BEST accuracy, set both API keys:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export GEMINI_API_KEY="your_key"
```

---

## 📞 GET HELP

**Google Cloud Vision:**
- Docs: https://cloud.google.com/vision/docs
- Pricing: https://cloud.google.com/vision/pricing
- Support: https://cloud.google.com/support

**Gemini API:**
- Docs: https://ai.google.dev/docs
- API Keys: https://makersuite.google.com/app/apikey
- Pricing: https://ai.google.dev/pricing

---

**Last Updated:** November 27, 2025  
**Project:** Sanskrit Manuscript Processing Pipeline  
**Mode:** Strict Compliance (Google Lens + Gemini + mBART)

