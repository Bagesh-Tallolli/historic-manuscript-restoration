#!/bin/bash
# Quick Start Script for Image Polish with Gemini API

echo "=============================================="
echo "✨ IMAGE POLISH WITH GEMINI API"
echo "=============================================="
echo ""

cd /home/bagesh/EL-project

# Check if already running on port 8502
if lsof -i :8502 > /dev/null 2>&1; then
    echo "⚠️  Image Polish app is already running on port 8502"
    echo ""
    echo "🌐 Access at: http://localhost:8502"
    echo ""
    echo "To stop: pkill -f 'streamlit run image_polish_gemini.py'"
    exit 0
fi

echo "🔄 Starting Image Polish app..."
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
echo "🚀 Launching Streamlit app on port 8502..."
echo ""

# Start Streamlit
streamlit run image_polish_gemini.py \
    --server.port 8502 \
    --server.headless true \
    --server.address localhost

echo ""
echo "=============================================="
echo "✅ IMAGE POLISH APP STARTED!"
echo "=============================================="
echo ""
echo "🌐 Access at:"
echo "   http://localhost:8502"
echo ""

