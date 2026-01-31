#!/bin/bash
# Quick start script for Sanskrit OCR-Translation Agent

echo "============================================"
echo "  Sanskrit Manuscript Pipeline - Quick Start"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check for model checkpoints
echo "📋 Checking model checkpoints..."
if [ ! -f "checkpoints/kaggle/final.pth" ]; then
    echo "⚠️  WARNING: Model checkpoint not found at checkpoints/kaggle/final.pth"
    echo "   Make sure you've downloaded your trained models from Kaggle"
fi

# Kill existing Streamlit processes
echo "🔄 Stopping any existing Streamlit processes..."
pkill -f "streamlit run" 2>/dev/null
sleep 2

# Start Streamlit
echo ""
echo "🚀 Starting Streamlit application..."
echo ""
echo "Pipeline Stages:"
echo "  1️⃣  Image Restoration (ViT model)"
echo "  2️⃣  Google Lens OCR (Cloud Vision API)"
echo "  3️⃣  Gemini Text Correction (AI-powered)"
echo "  4️⃣  Sanskrit → English Translation"
echo "  5️⃣  Quality Verification"
echo ""
echo "📝 Optional Configuration:"
echo "   - Google Cloud credentials: Set GOOGLE_APPLICATION_CREDENTIALS env var"
echo "   - Gemini API key: Set GEMINI_API_KEY env var"
echo "   - Or configure in the Streamlit sidebar"
echo ""
echo "============================================"
echo ""

# Run Streamlit
streamlit run app_enhanced.py --server.headless true

# Deactivate on exit
deactivate

