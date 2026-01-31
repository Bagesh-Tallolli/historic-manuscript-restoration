#!/bin/bash

# Run the Sanskrit OCR Streamlit Application

echo "=========================================="
echo "Sanskrit OCR Text Extraction Application"
echo "=========================================="
echo ""
echo "Starting Streamlit application..."
echo ""
echo "The application will open in your browser at:"
echo "http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run streamlit
streamlit run app_sanskrit_ocr.py --server.port 8501 --server.address localhost

