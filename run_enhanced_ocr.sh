#!/bin/bash

# Enhanced Sanskrit OCR Application Startup Script
# This script runs the enhanced OCR application with image restoration

echo "=============================================="
echo "  Enhanced Sanskrit OCR with AI Restoration  "
echo "  ✅ Restoration Fix Applied (Nov 30, 2025)  "
echo "=============================================="
echo ""

# Check if we're in the project directory
if [ ! -f "ocr_gemini_streamlit.py" ]; then
    echo "❌ Error: ocr_gemini_streamlit.py not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check for virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo ""
    echo "❌ Streamlit not found! Installing dependencies..."
    pip install -r ocr_gemini_streamlit_requirements.txt
fi

# Check for model checkpoint
echo ""
echo "🔍 Checking for restoration model checkpoint..."
CHECKPOINT_FOUND=false

for checkpoint in "checkpoints/kaggle/final_converted.pth" "checkpoints/kaggle/final.pth" "models/trained_models/final.pth"; do
    if [ -f "$checkpoint" ]; then
        echo "✅ Found checkpoint: $checkpoint"
        CHECKPOINT_FOUND=true
        break
    fi
done

if [ "$CHECKPOINT_FOUND" = false ]; then
    echo "⚠️  WARNING: No restoration model checkpoint found!"
    echo "   Restoration feature will be disabled."
    echo "   To enable restoration:"
    echo "   1. Train the model using Kaggle workflow"
    echo "   2. Place checkpoint in checkpoints/kaggle/ directory"
fi

# Check for CUDA
echo ""
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "🚀 GPU detected! Will use CUDA acceleration."
else
    echo "💻 No GPU detected. Will use CPU (slower)."
fi

echo ""
echo "=============================================="
echo "  Starting Streamlit Application...          "
echo "=============================================="
echo ""
echo "The application will open in your browser."
echo "If it doesn't, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Run streamlit
streamlit run ocr_gemini_streamlit.py

