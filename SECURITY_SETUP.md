# Security Setup Guide

## API Key Management

This project uses environment variables to manage sensitive API keys. **Never commit API keys directly in code.**

## Setup Instructions

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Add Your API Keys

Edit `.env` and add your actual API keys:

```bash
# Gemini API Key (Google AI Studio)
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Groq API Key (optional)
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Get API Keys

#### Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key to your `.env` file

#### Groq API Key (Optional)
1. Go to [Groq Console](https://console.groq.com)
2. Sign up or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

### 4. Verify Setup

The `.env` file should be automatically ignored by git (configured in `.gitignore`). Verify:

```bash
git status
```

You should **NOT** see `.env` in the list of files to be committed.

## Important Security Notes

⚠️ **NEVER** commit the `.env` file to git
⚠️ **NEVER** share your API keys publicly
⚠️ **NEVER** hardcode API keys in your code

✅ Always use environment variables
✅ Keep `.env` file local only
✅ Share `.env.example` as a template

## Checking for Exposed Keys

If you accidentally commit API keys, GitHub will block the push. If this happens:

1. Remove the keys from the code
2. Use environment variables instead
3. Amend the commit: `git commit --amend`
4. Force push: `git push --force`

## For Collaborators

When cloning this repository:

1. Copy `.env.example` to `.env`
2. Get your own API keys
3. Add them to your local `.env` file
4. Never commit your `.env` file

## Running the Application

All scripts are configured to automatically load environment variables from `.env`:

```bash
# Streamlit apps will automatically load .env
streamlit run streamlit_app.py

# Shell scripts check for .env before running
./start_ocr_gemini_pipeline.sh
```

## Troubleshooting

### "API key not found" error

Make sure:
1. `.env` file exists in the project root
2. Keys are properly formatted (no quotes, spaces, or extra characters)
3. `python-dotenv` is installed: `pip install python-dotenv`

### Keys not loading

Check that your Python files include:

```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
```

