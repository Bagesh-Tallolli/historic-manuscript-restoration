#!/bin/bash

# Quick setup after downloading trained model from Kaggle

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║              🚀 POST-TRAINING SETUP AUTOMATION 🚀                            ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running from project root
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from the project root directory${NC}"
    exit 1
fi

echo "📋 Step 1: Checking for trained model..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if model exists
MODEL_PATH="checkpoints/best_psnr.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠️  Model not found at: $MODEL_PATH${NC}"
    echo ""
    echo "Please download your trained model from Kaggle:"
    echo "  1. Go to your Kaggle notebook"
    echo "  2. Right sidebar → Output"
    echo "  3. Download: /kaggle/working/checkpoints/best_psnr.pth"
    echo ""
    read -p "Enter the path to your downloaded model file: " DOWNLOADED_MODEL

    if [ -f "$DOWNLOADED_MODEL" ]; then
        echo ""
        echo "📥 Copying model to project..."
        mkdir -p checkpoints
        cp "$DOWNLOADED_MODEL" "$MODEL_PATH"
        echo -e "${GREEN}✓ Model copied successfully!${NC}"
    else
        echo -e "${RED}❌ File not found: $DOWNLOADED_MODEL${NC}"
        echo "Please download the model first and run this script again."
        exit 1
    fi
else
    FILE_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo -e "${GREEN}✓ Model found: $MODEL_PATH ($FILE_SIZE)${NC}"
fi

echo ""
echo "🧪 Step 2: Testing model..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Test model
python3 test_trained_model.py "$MODEL_PATH"
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Model test failed. Please check the errors above.${NC}"
    exit 1
fi

echo ""
echo "📊 Step 3: Checking dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if key packages are installed
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  PyTorch not installed${NC}"
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo -e "${GREEN}✓ PyTorch installed${NC}"
fi

python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Streamlit not installed${NC}"
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo -e "${GREEN}✓ Streamlit installed${NC}"
fi

echo ""
echo "🎨 Step 4: Testing inference..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if sample data exists
SAMPLE_DIR="data/samples"
if [ -d "$SAMPLE_DIR" ] && [ "$(ls -A $SAMPLE_DIR/*.jpg 2>/dev/null)" ]; then
    echo "Found sample images. Running test inference..."
    SAMPLE_IMG=$(ls $SAMPLE_DIR/*.jpg | head -1)

    python3 inference.py \
        --model "$MODEL_PATH" \
        --input "$SAMPLE_IMG" \
        --output "output/test_restoration.jpg" \
        --device cpu

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Inference test successful!${NC}"
        echo "  Output saved to: output/test_restoration.jpg"
    else
        echo -e "${YELLOW}⚠️  Inference test had issues (check above)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No sample images found in $SAMPLE_DIR${NC}"
    echo "  Skipping inference test"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                        ✅ SETUP COMPLETE! ✅                                  ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎉 Your model is ready to use!"
echo ""
echo "📋 Quick Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  Start Streamlit Web App:"
echo "   streamlit run app.py"
echo ""
echo "2️⃣  Process a single image:"
echo "   python inference.py --model checkpoints/best_psnr.pth \\"
echo "                       --input your_image.jpg \\"
echo "                       --output restored.jpg"
echo ""
echo "3️⃣  Batch process folder:"
echo "   python inference.py --model checkpoints/best_psnr.pth \\"
echo "                       --input data/samples/ \\"
echo "                       --output output/"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📚 For more information, see:"
echo "   • NEXT_STEPS_AFTER_TRAINING.txt"
echo "   • DEPLOYMENT_GUIDE.md"
echo "   • QUICKSTART.md"
echo ""
echo "🚀 Happy manuscript restoration!"
echo ""

