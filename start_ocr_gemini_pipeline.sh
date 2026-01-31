#!/bin/bash
# Quick Start Script for OCR Gemini Streamlit Pipeline
# Complete pipeline with ViT Restoration + Gemini API

echo "=============================================="
echo "🔰 OCR GEMINI STREAMLIT PIPELINE"
echo "=============================================="
echo ""

cd /home/bagesh/EL-project

# Check if already running
if lsof -i :8501 > /dev/null 2>&1; then
    echo "⚠️  Pipeline is already running on port 8501"
    echo ""
    echo "📊 Current Process:"
    ps aux | grep "streamlit run ocr_gemini_streamlit.py" | grep -v grep
    echo ""
    echo "🌐 Access URLs:"
    echo "   • Local:    http://localhost:8501"
    echo "   • Network:  http://172.20.66.141:8501"
    echo ""
    echo "To stop: pkill -f 'streamlit run ocr_gemini_streamlit.py'"
    echo "To restart: kill the process and run this script again"
    echo ""
    exit 0
fi

echo "🔄 Starting pipeline..."
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded"
else
    echo "⚠️  Warning: .env file not found. Please create one from .env.example"
    exit 1
fi

echo ""
echo "🚀 Launching Streamlit app..."
echo ""

# Start Streamlit in background
nohup streamlit run ocr_gemini_streamlit.py \
    --server.port 8501 \
    --server.headless true \
    > streamlit_ocr_gemini.log 2>&1 &

# Get PID
STREAMLIT_PID=$!
echo "📌 Process ID: $STREAMLIT_PID"

# Wait for startup
echo "⏳ Waiting for app to start..."
sleep 5

# Check if actually running
if lsof -i :8501 > /dev/null 2>&1; then
    echo ""
    echo "=============================================="
    echo "✅ PIPELINE IS RUNNING!"
    echo "=============================================="
    echo ""
    echo "🌐 Access the application at:"
    echo ""
    echo "   📍 Local:    http://localhost:8501"
    echo "   📍 Network:  http://172.20.66.141:8501"
    echo ""
    echo "=============================================="
    echo "📋 Features:"
    echo "   ✨ ViT Image Restoration"
    echo "   🔍 Gemini OCR (Sanskrit)"
    echo "   🌍 Multi-language Translation"
    echo "   📥 Export Results"
    echo "=============================================="
    echo ""
    echo "📝 View logs:"
    echo "   tail -f streamlit_ocr_gemini.log"
    echo ""
    echo "🛑 Stop pipeline:"
    echo "   kill $STREAMLIT_PID"
    echo "   # or: pkill -f 'streamlit run ocr_gemini_streamlit.py'"
    echo ""
else
    echo ""
    echo "❌ Failed to start pipeline"
    echo "📝 Check logs: cat streamlit_ocr_gemini.log"
    echo ""
    exit 1
fi

