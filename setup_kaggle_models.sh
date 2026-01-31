#!/bin/bash

# ============================================================================
# Kaggle Model Setup Script
# ============================================================================
# This script helps you set up trained models downloaded from Kaggle
# Usage: bash setup_kaggle_models.sh [source_directory]
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     KAGGLE MODEL SETUP - Historic Manuscript Restoration       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/home/bagesh/EL-project"
KAGGLE_CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/kaggle"
TRAINED_MODELS_DIR="$PROJECT_ROOT/models/trained_models"

# Create directories
echo -e "${BLUE}📁 Setting up directories...${NC}"
mkdir -p "$KAGGLE_CHECKPOINT_DIR"
mkdir -p "$TRAINED_MODELS_DIR"
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Function to check if file exists and get size
check_model_file() {
    local filepath=$1
    local filename=$(basename "$filepath")

    if [ -f "$filepath" ]; then
        local size=$(du -h "$filepath" | cut -f1)
        echo -e "${GREEN}✓ Found: $filename ($size)${NC}"
        return 0
    else
        echo -e "${YELLOW}✗ Not found: $filename${NC}"
        return 1
    fi
}

# Check for source directory argument
if [ $# -eq 1 ]; then
    SOURCE_DIR="$1"
    echo -e "${BLUE}📥 Copying models from: $SOURCE_DIR${NC}"
    echo ""

    if [ -d "$SOURCE_DIR" ]; then
        # Copy .pth files
        if ls "$SOURCE_DIR"/*.pth 1> /dev/null 2>&1; then
            cp "$SOURCE_DIR"/*.pth "$KAGGLE_CHECKPOINT_DIR/"
            echo -e "${GREEN}✓ Model files copied to $KAGGLE_CHECKPOINT_DIR${NC}"
        else
            echo -e "${RED}✗ No .pth files found in $SOURCE_DIR${NC}"
        fi
    else
        echo -e "${RED}✗ Directory not found: $SOURCE_DIR${NC}"
        exit 1
    fi
    echo ""
fi

# Check what model files are present
echo -e "${BLUE}🔍 Checking for model files in $KAGGLE_CHECKPOINT_DIR...${NC}"
echo ""

DESTI_FOUND=false
FINAL_FOUND=false

if check_model_file "$KAGGLE_CHECKPOINT_DIR/desti.pth"; then
    DESTI_FOUND=true
fi

if check_model_file "$KAGGLE_CHECKPOINT_DIR/final.pth"; then
    FINAL_FOUND=true
fi

# List all .pth files
echo ""
echo -e "${BLUE}📋 All model files in kaggle directory:${NC}"
if ls "$KAGGLE_CHECKPOINT_DIR"/*.pth 1> /dev/null 2>&1; then
    ls -lh "$KAGGLE_CHECKPOINT_DIR"/*.pth | awk '{printf "   %s  %s\n", $5, $9}'
else
    echo -e "${YELLOW}   (No .pth files found)${NC}"
fi

echo ""
echo "─────────────────────────────────────────────────────────────────"
echo ""

# Provide guidance based on what was found
if [ "$DESTI_FOUND" = true ] || [ "$FINAL_FOUND" = true ]; then
    echo -e "${GREEN}✓ Model files are ready to use!${NC}"
    echo ""
    echo -e "${BLUE}📖 Next steps:${NC}"
    echo ""

    if [ "$FINAL_FOUND" = true ]; then
        echo "1. Test your model with inference:"
        echo -e "   ${YELLOW}python inference.py --input data/datasets/samples/ \\${NC}"
        echo -e "   ${YELLOW}      --output output/kaggle_test \\${NC}"
        echo -e "   ${YELLOW}      --checkpoint checkpoints/kaggle/final.pth \\${NC}"
        echo -e "   ${YELLOW}      --model_size base${NC}"
        echo ""
    fi

    echo "2. Run the Streamlit app:"
    echo -e "   ${YELLOW}streamlit run app.py${NC}"
    echo ""

    echo "3. Start the API server:"
    echo -e "   ${YELLOW}python api_server.py --checkpoint checkpoints/kaggle/final.pth${NC}"
    echo ""

else
    echo -e "${YELLOW}⚠ No model files found yet.${NC}"
    echo ""
    echo -e "${BLUE}📥 To add your models from Kaggle:${NC}"
    echo ""
    echo "Option 1 - Manual placement:"
    echo -e "   ${YELLOW}cp ~/Downloads/desti.pth $KAGGLE_CHECKPOINT_DIR/${NC}"
    echo -e "   ${YELLOW}cp ~/Downloads/final.pth $KAGGLE_CHECKPOINT_DIR/${NC}"
    echo ""
    echo "Option 2 - Specify source directory:"
    echo -e "   ${YELLOW}bash setup_kaggle_models.sh ~/Downloads${NC}"
    echo ""
    echo "Option 3 - Use Kaggle API:"
    echo -e "   ${YELLOW}kaggle kernels output <username>/<notebook> -p $KAGGLE_CHECKPOINT_DIR${NC}"
    echo ""
fi

# Create a symlink for easy access
if [ "$FINAL_FOUND" = true ]; then
    ln -sf "$KAGGLE_CHECKPOINT_DIR/final.pth" "$PROJECT_ROOT/checkpoints/latest_model.pth" 2>/dev/null || true
    echo -e "${GREEN}✓ Created symlink: checkpoints/latest_model.pth → kaggle/final.pth${NC}"
elif [ "$DESTI_FOUND" = true ]; then
    ln -sf "$KAGGLE_CHECKPOINT_DIR/desti.pth" "$PROJECT_ROOT/checkpoints/latest_model.pth" 2>/dev/null || true
    echo -e "${GREEN}✓ Created symlink: checkpoints/latest_model.pth → kaggle/desti.pth${NC}"
fi

echo ""
echo -e "${BLUE}📚 For more information, see: KAGGLE_MODEL_INTEGRATION.md${NC}"
echo ""
echo "════════════════════════════════════════════════════════════════"

