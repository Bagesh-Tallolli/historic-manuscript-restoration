#!/bin/bash
# Quality-Guarded Manuscript Vision Pipeline - Startup Script
echo "=============================================="
echo "🔰 Quality-Guarded Manuscript Vision Pipeline"
echo "=============================================="
echo ""
# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi
# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate
# Check if dependencies are installed
echo "📦 Checking dependencies..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing required dependencies..."
    pip install -q streamlit google-genai scikit-image torch torchvision opencv-python pillow numpy einops
fi
# Check for Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo ""
    echo "⚠️  GEMINI_API_KEY environment variable not set."
    echo "   The application will use the default key or you can enter it in the UI."
    echo ""
fi
# Check for ViT model checkpoint
VIT_CHECKPOINT="models/trained_models/final.pth"
if [ ! -f "$VIT_CHECKPOINT" ]; then
    echo ""
    echo "⚠️  ViT model checkpoint not found at: $VIT_CHECKPOINT"
    echo "   The application will use PIL-based fallback restoration."
    echo "   This is less powerful but still functional."
    echo ""
fi
echo ""
echo "=============================================="
echo "🚀 Starting Quality-Guarded Pipeline UI..."
echo "=============================================="
echo ""
echo "📌 The application will open in your browser automatically"
echo "📌 If not, navigate to: http://localhost:8501"
echo ""
echo "🔰 Pipeline Features:"
echo "   ✅ ViT Restoration Model (with quality gate)"
echo "   ✅ Automatic quality comparison"
echo "   ✅ Fallback to original if restoration degrades quality"
echo "   ✅ Gemini API for OCR & translation"
echo "   ✅ Multilingual output (English, Hindi, Kannada)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="
echo ""
# Run Streamlit app
python -m streamlit run app_quality_guarded.py
