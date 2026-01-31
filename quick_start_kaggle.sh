#!/bin/bash

# ============================================================================
# Quick Start Script for Kaggle-Trained Models
# ============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║    KAGGLE MODEL QUICK START - Sanskrit Manuscript Restoration    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Paths
PROJECT_ROOT="/home/bagesh/EL-project"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/kaggle"
CONVERTED_MODEL="$CHECKPOINT_DIR/final_converted.pth"
SAMPLE_DIR="$PROJECT_ROOT/data/datasets/samples"
OUTPUT_DIR="$PROJECT_ROOT/output/quick_test"

# Change to project directory
cd "$PROJECT_ROOT"

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source activate_venv.sh > /dev/null 2>&1
echo ""

# Check if converted model exists
if [ ! -f "$CONVERTED_MODEL" ]; then
    echo -e "${YELLOW}⚠️  Converted model not found. Converting now...${NC}"
    if [ -f "$CHECKPOINT_DIR/final.pth" ]; then
        python convert_kaggle_checkpoint.py "$CHECKPOINT_DIR/final.pth"
        echo ""
    else
        echo -e "${RED}❌ No model found to convert!${NC}"
        echo "Please place your Kaggle model files in: $CHECKPOINT_DIR"
        exit 1
    fi
fi

# Main menu
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║               What would you like to do?                     ║${NC}"
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  1. Test model on sample images"
echo "  2. Run complete pipeline (Restoration + OCR + Translation)"
echo "  3. Start Streamlit web app"
echo "  4. Start API server"
echo "  5. Convert another checkpoint"
echo "  6. View model info"
echo "  0. Exit"
echo ""
read -p "Enter your choice [0-6]: " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}🖼️  Testing model on sample images...${NC}"
        echo "─────────────────────────────────────────────────────────────"

        # Check if samples exist
        if [ ! -d "$SAMPLE_DIR" ] || [ -z "$(ls -A $SAMPLE_DIR 2>/dev/null)" ]; then
            echo -e "${YELLOW}⚠️  No sample images found in $SAMPLE_DIR${NC}"
            read -p "Enter path to your image or directory: " INPUT_PATH
        else
            INPUT_PATH="$SAMPLE_DIR"
        fi

        echo ""
        python inference.py \
            --input "$INPUT_PATH" \
            --output "$OUTPUT_DIR" \
            --checkpoint "$CONVERTED_MODEL" \
            --model_size base

        echo ""
        echo -e "${GREEN}✓ Results saved to: $OUTPUT_DIR${NC}"
        ;;

    2)
        echo ""
        echo -e "${GREEN}🔄 Running complete pipeline...${NC}"
        echo "─────────────────────────────────────────────────────────────"
        read -p "Enter path to manuscript image: " IMG_PATH

        if [ ! -f "$IMG_PATH" ]; then
            echo -e "${RED}❌ File not found: $IMG_PATH${NC}"
            exit 1
        fi

        echo ""
        python main.py \
            --image_path "$IMG_PATH" \
            --restoration_model "$CONVERTED_MODEL" \
            --ocr_engine tesseract \
            --translation_method google \
            --output_dir output/pipeline_results

        echo ""
        echo -e "${GREEN}✓ Pipeline complete! Check output/pipeline_results/${NC}"
        ;;

    3)
        echo ""
        echo -e "${GREEN}🌐 Starting Streamlit web app...${NC}"
        echo "─────────────────────────────────────────────────────────────"
        echo "The app will open in your browser automatically."
        echo "Press Ctrl+C to stop the server."
        echo ""
        streamlit run app.py
        ;;

    4)
        echo ""
        echo -e "${GREEN}🚀 Starting API server...${NC}"
        echo "─────────────────────────────────────────────────────────────"
        read -p "Enter port number (default: 8000): " PORT
        PORT=${PORT:-8000}

        echo ""
        echo "API will be available at: http://localhost:$PORT"
        echo "Press Ctrl+C to stop the server."
        echo ""
        python api_server.py --checkpoint "$CONVERTED_MODEL" --port "$PORT"
        ;;

    5)
        echo ""
        echo -e "${GREEN}🔄 Convert checkpoint...${NC}"
        echo "─────────────────────────────────────────────────────────────"
        read -p "Enter path to checkpoint file: " CKPT_PATH

        if [ ! -f "$CKPT_PATH" ]; then
            echo -e "${RED}❌ File not found: $CKPT_PATH${NC}"
            exit 1
        fi

        echo ""
        python convert_kaggle_checkpoint.py "$CKPT_PATH"
        ;;

    6)
        echo ""
        echo -e "${GREEN}📊 Model Information${NC}"
        echo "─────────────────────────────────────────────────────────────"
        python -c "
import torch
from models.vit_restorer import create_vit_restorer

print('\n🔍 Model Details:')
print('─' * 60)
model = create_vit_restorer('base', img_size=256)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Architecture: ViT-Base')
print(f'Total Parameters: {total_params:,}')
print(f'Trainable Parameters: {trainable_params:,}')
print(f'Input Size: 256x256 pixels')
print(f'Transformer Blocks: 12')
print(f'Attention Heads: 12')
print(f'Embedding Dimension: 768')

print('\n📁 Available Checkpoints:')
print('─' * 60)
import os
ckpt_dir = 'checkpoints/kaggle'
if os.path.exists(ckpt_dir):
    for file in os.listdir(ckpt_dir):
        if file.endswith('.pth'):
            size_mb = os.path.getsize(os.path.join(ckpt_dir, file)) / (1024*1024)
            print(f'  • {file} ({size_mb:.1f} MB)')
else:
    print('  (No checkpoints found)')
print()
"
        ;;

    0)
        echo ""
        echo -e "${GREEN}👋 Goodbye!${NC}"
        echo ""
        exit 0
        ;;

    *)
        echo ""
        echo -e "${RED}❌ Invalid choice!${NC}"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Done!${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

