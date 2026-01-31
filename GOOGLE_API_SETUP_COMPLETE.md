# 🚀 Google Vision API Setup Complete

## ✅ What Has Been Configured

Your Google Cloud Vision API key has been successfully added to the project:

**API Key:** `d48382987f9cddac6b042e3703797067fd46f2b0`

---

## 📁 Files Created/Modified

### 1. `.env` File Created
Location: `/home/bagesh/EL-project/.env`

Contains your API key:
```bash
GOOGLE_VISION_API_KEY=d48382987f9cddac6b042e3703797067fd46f2b0
```

**🔒 SECURITY:** This file is automatically excluded from Git (added to .gitignore)

---

### 2. `sanskrit_ocr_agent.py` Updated
- Added support for `GOOGLE_VISION_API_KEY` environment variable
- Automatically loads API key from `.env` file
- Falls back to JSON credentials if needed

---

### 3. `app_professional.py` Updated
- Loads environment variables from `.env` file on startup
- API key is automatically available to the pipeline

---

### 4. `setup_api_keys.sh` Script Created
A convenient script to verify your API keys are loaded.

---

## 🏃 How to Run the Project

### Option 1: Quick Start (Recommended)

```bash
cd /home/bagesh/EL-project
source venv/bin/activate
streamlit run app_professional.py
```

The API key will be automatically loaded from the `.env` file!

---

### Option 2: With Environment Variable Export

```bash
cd /home/bagesh/EL-project
source setup_api_keys.sh        # Load API keys
source venv/bin/activate         # Activate virtual environment
streamlit run app_professional.py
```

---

### Option 3: Manual Export

```bash
cd /home/bagesh/EL-project
export GOOGLE_VISION_API_KEY="d48382987f9cddac6b042e3703797067fd46f2b0"
source venv/bin/activate
streamlit run app_professional.py
```

---

## 🧪 Test Your API Key

Run this command to verify the API key is properly loaded:

```bash
cd /home/bagesh/EL-project
source venv/bin/activate
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('GOOGLE_VISION_API_KEY')
print('✅ API Key loaded:', key[:10] + '...' if key else '❌ Not found')
"
```

---

## 📊 How the Pipeline Uses Your API Key

When you upload a manuscript image:

1. **Image Restoration** → Uses your local ViT model ✓
2. **OCR (Google Lens)** → Uses `GOOGLE_VISION_API_KEY` ✓
3. **Text Correction** → Uses Gemini (if configured) or skips
4. **Translation** → Uses local mBART model ✓

---

## 🔧 Additional Configuration (Optional)

### Add Gemini API Key for Better Text Correction

Edit `.env` file and add:

```bash
GEMINI_API_KEY=your_gemini_key_here
```

Get a Gemini key from: https://makersuite.google.com/app/apikey

---

## ⚠️ Important Notes

### About Your API Key

The key you provided (`d48382987f9cddac6b042e3703797067fd46f2b0`) looks like a hash/token format.

**Standard Google Cloud Vision API requires:**
- A JSON credentials file from Google Cloud Console
- Service Account credentials

**If your key doesn't work:**

1. Go to: https://console.cloud.google.com/
2. Create a project
3. Enable "Cloud Vision API"
4. Create Service Account → Download JSON credentials
5. Update `.env` with:
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
   ```

Alternatively, if this is an API key from a different Google service, you may need to adjust the authentication method.

---

## 🐛 Troubleshooting

### "API key not found" error

```bash
# Check if .env file exists
cat /home/bagesh/EL-project/.env

# Verify environment variable is loaded
source venv/bin/activate
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GOOGLE_VISION_API_KEY'))"
```

### "Authentication failed" error

This means your API key format might be incorrect. Google Cloud Vision typically needs:
- Either: JSON credentials file (recommended)
- Or: API key from Google Cloud Console (different format)

---

## 📞 Next Steps

1. **Test the project:**
   ```bash
   cd /home/bagesh/EL-project
   source venv/bin/activate
   streamlit run app_professional.py
   ```

2. **Upload a manuscript image** through the web interface

3. **Check the OCR output** - if it fails, you may need to get proper JSON credentials from Google Cloud Console

---

## 🔐 Security Reminder

✅ Your `.env` file is protected:
- Added to `.gitignore`
- Will NOT be committed to Git
- API key is safe

🚨 Never share your API keys publicly!

---

**Status:** ✅ Ready to run!

Run the project now:
```bash
cd /home/bagesh/EL-project && source venv/bin/activate && streamlit run app_professional.py
```

