#!/bin/bash
# Quick setup script for Roboflow Sanskrit OCR Dataset

echo "=================================="
echo "🚀 Roboflow Dataset Quick Setup"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "Run: bash setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "Step 1: Installing Roboflow package..."
pip install -q roboflow

echo "✅ Roboflow installed"
echo ""

echo "Step 2: Get your API key"
echo "----------------------------------------"
echo "1. Visit: https://app.roboflow.com/"
echo "2. Sign in or create account"
echo "3. Go to: Settings → Roboflow API"
echo "4. Copy your API key"
echo ""

read -p "Enter your Roboflow API key: " API_KEY

if [ -z "$API_KEY" ]; then
    echo "❌ No API key provided"
    echo ""
    echo "You can also download manually:"
    echo "  python3 download_roboflow_dataset.py --api-key YOUR_KEY"
    exit 1
fi

echo ""
echo "Step 3: Downloading dataset..."
python3 download_roboflow_dataset.py --api-key "$API_KEY"

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next: Start training"
echo "  python3 train.py --train_dir data/raw/train --val_dir data/raw/val"
echo ""

