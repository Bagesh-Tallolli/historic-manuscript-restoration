#!/bin/bash

# Quick start script for Sanskrit OCR Streamlit Application

echo "=========================================="
echo "Sanskrit OCR & Translation App"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r ocr_requirements.txt

echo ""
echo "=========================================="
echo "Starting Streamlit Application..."
echo "=========================================="
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the streamlit app
streamlit run ocr_gemini_streamlit.py

