#!/bin/bash
# Quick verification script for Roboflow dataset setup

echo "🔍 Checking Roboflow Dataset Setup Status"
echo "=========================================="
echo ""

# Check 1: Roboflow package
echo "1. Checking Roboflow package..."
if python3 -c "import roboflow" 2>/dev/null; then
    echo "   ✅ Roboflow package installed"
else
    echo "   ❌ Roboflow package NOT installed"
    echo "      Install with: pip install roboflow"
fi
echo ""

# Check 2: Download script
echo "2. Checking download script..."
if [ -f "download_roboflow_dataset.py" ]; then
    echo "   ✅ Download script exists"
else
    echo "   ❌ Download script missing"
fi
echo ""

# Check 3: Data directories
echo "3. Checking data directories..."
if [ -d "data/raw" ]; then
    echo "   ✅ data/raw/ exists"
else
    echo "   ❌ data/raw/ missing"
    mkdir -p data/raw
    echo "      Created data/raw/"
fi
echo ""

# Check 4: Training images
echo "4. Checking training images..."
TRAIN_COUNT=$(find data/raw/train -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
VAL_COUNT=$(find data/raw/val -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
TEST_COUNT=$(find data/raw/test -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)

if [ $TRAIN_COUNT -gt 0 ]; then
    echo "   ✅ Training images: $TRAIN_COUNT"
else
    echo "   ⚠️  No training images found"
    echo "      Download with: bash setup_roboflow.sh"
fi

if [ $VAL_COUNT -gt 0 ]; then
    echo "   ✅ Validation images: $VAL_COUNT"
else
    echo "   ⚠️  No validation images found"
fi

if [ $TEST_COUNT -gt 0 ]; then
    echo "   ✅ Test images: $TEST_COUNT"
else
    echo "   ⚠️  No test images found"
fi
echo ""

# Summary
echo "=========================================="
TOTAL=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))
if [ $TOTAL -gt 0 ]; then
    echo "✅ Dataset Status: READY ($TOTAL total images)"
    echo ""
    echo "Next steps:"
    echo "  python3 train.py --train_dir data/raw/train --val_dir data/raw/val"
else
    echo "⚠️  Dataset Status: NOT READY"
    echo ""
    echo "Next steps:"
    echo "  1. Get API key from https://app.roboflow.com/"
    echo "  2. Run: bash setup_roboflow.sh"
fi
echo "=========================================="

