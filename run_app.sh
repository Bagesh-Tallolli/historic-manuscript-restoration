#!/bin/bash
# Launcher script for Sanskrit Manuscript Pipeline Streamlit App

echo "🕉️  Sanskrit Manuscript Pipeline - Streamlit Web App"
echo "======================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit>=1.28.0
fi

# Create output directory for streamlit
mkdir -p output/streamlit

echo ""
echo "🚀 Starting Streamlit app..."
echo "📱 The app will open in your default browser"
echo "🌐 URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run streamlit app
streamlit run app.py

