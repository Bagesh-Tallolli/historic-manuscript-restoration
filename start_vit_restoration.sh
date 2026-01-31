#!/bin/bash
# Sanskrit Manuscript Image Restoration - ViT Model
# Blur/Unclear → Clear/Polished

echo "=============================================="
echo "✨ ViT Image Restoration for Sanskrit Manuscripts"
echo "=============================================="
echo ""

cd /home/bagesh/EL-project

# Check if already running on port 8503
if lsof -i :8503 > /dev/null 2>&1; then
    echo "⚠️  ViT Restoration app is already running on port 8503"
    echo ""
    echo "🌐 Access at: http://localhost:8503"
    echo ""
    echo "To stop: pkill -f 'streamlit run vit_image_restoration.py'"
    exit 0
fi

echo "🔄 Starting ViT Image Restoration app..."
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found, using system Python"
fi

echo ""
echo "🚀 Launching Streamlit app on port 8503..."
echo ""

# Start Streamlit
streamlit run vit_image_restoration.py \
    --server.port 8503 \
    --server.headless true \
    --server.address localhost

echo ""
echo "=============================================="
echo "✅ ViT RESTORATION APP STARTED!"
echo "=============================================="
echo ""
echo "🌐 Access at:"
echo "   http://localhost:8503"
echo ""
echo "🎯 What it does:"
echo "   Upload blur/unclear image → Get clear/polished image"
echo ""

