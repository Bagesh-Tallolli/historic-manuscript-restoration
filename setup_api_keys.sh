#!/bin/bash
# Quick setup script to load API keys from .env file
# Usage: source setup_api_keys.sh

echo "🔐 Loading API Keys for Sanskrit Manuscript Pipeline..."

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Loaded API keys from .env file"

    # Verify keys are set
    if [ -n "$GOOGLE_VISION_API_KEY" ]; then
        echo "✓ Google Vision API Key: ${GOOGLE_VISION_API_KEY:0:10}..."
    else
        echo "⚠ Google Vision API Key not found"
    fi

    if [ -n "$GEMINI_API_KEY" ]; then
        echo "✓ Gemini API Key: ${GEMINI_API_KEY:0:10}..."
    else
        echo "⚠ Gemini API Key not set (optional)"
    fi
else
    echo "❌ .env file not found!"
    echo "Please create a .env file with your API keys:"
    echo "  GOOGLE_VISION_API_KEY=your_key_here"
    echo "  GEMINI_API_KEY=your_key_here"
fi

echo ""
echo "To activate virtual environment and run the project:"
echo "  source venv/bin/activate"
echo "  streamlit run app_professional.py"

