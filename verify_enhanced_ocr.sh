#!/bin/bash

# Verification script for Enhanced OCR setup
# Run this to verify everything is configured correctly

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Enhanced Sanskrit OCR - System Verification                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Check 1: Main application file
echo "📝 Checking application file..."
if [ -f "ocr_gemini_streamlit.py" ]; then
    echo -e "${GREEN}✅ ocr_gemini_streamlit.py found${NC}"
else
    echo -e "${RED}❌ ocr_gemini_streamlit.py not found${NC}"
    ERRORS=$((ERRORS+1))
fi

# Check 2: Startup script
echo "📝 Checking startup script..."
if [ -f "run_enhanced_ocr.sh" ] && [ -x "run_enhanced_ocr.sh" ]; then
    echo -e "${GREEN}✅ run_enhanced_ocr.sh found and executable${NC}"
else
    echo -e "${RED}❌ run_enhanced_ocr.sh not found or not executable${NC}"
    ERRORS=$((ERRORS+1))
fi

# Check 3: Model architecture
echo "📝 Checking model architecture..."
if [ -f "models/vit_restorer.py" ]; then
    echo -e "${GREEN}✅ models/vit_restorer.py found${NC}"
else
    echo -e "${RED}❌ models/vit_restorer.py not found${NC}"
    ERRORS=$((ERRORS+1))
fi

# Check 4: Restoration utilities
echo "📝 Checking restoration utilities..."
if [ -f "utils/image_restoration_enhanced.py" ]; then
    echo -e "${GREEN}✅ utils/image_restoration_enhanced.py found${NC}"
else
    echo -e "${RED}❌ utils/image_restoration_enhanced.py not found${NC}"
    ERRORS=$((ERRORS+1))
fi

# Check 5: Model checkpoint
echo "📝 Checking model checkpoint..."
CHECKPOINT_FOUND=false
if [ -f "checkpoints/kaggle/final_converted.pth" ]; then
    SIZE=$(du -h checkpoints/kaggle/final_converted.pth | cut -f1)
    echo -e "${GREEN}✅ Primary checkpoint found: final_converted.pth ($SIZE)${NC}"
    CHECKPOINT_FOUND=true
elif [ -f "checkpoints/kaggle/final.pth" ]; then
    SIZE=$(du -h checkpoints/kaggle/final.pth | cut -f1)
    echo -e "${YELLOW}⚠️  Fallback checkpoint found: final.pth ($SIZE)${NC}"
    WARNINGS=$((WARNINGS+1))
    CHECKPOINT_FOUND=true
elif [ -f "models/trained_models/final.pth" ]; then
    SIZE=$(du -h models/trained_models/final.pth | cut -f1)
    echo -e "${YELLOW}⚠️  Fallback checkpoint found: models/trained_models/final.pth ($SIZE)${NC}"
    WARNINGS=$((WARNINGS+1))
    CHECKPOINT_FOUND=true
else
    echo -e "${RED}❌ No model checkpoint found${NC}"
    echo "   Restoration feature will be disabled"
    ERRORS=$((ERRORS+1))
fi

# Check 6: Python dependencies
echo "📝 Checking Python dependencies..."

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✅ Virtual environment active: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}⚠️  No virtual environment active${NC}"
    WARNINGS=$((WARNINGS+1))
fi

# Check key packages
MISSING_PACKAGES=()

for package in streamlit torch numpy cv2 PIL google.genai; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✅ $package installed${NC}"
    else
        echo -e "${RED}❌ $package not installed${NC}"
        MISSING_PACKAGES+=($package)
        ERRORS=$((ERRORS+1))
    fi
done

# Check 7: CUDA availability
echo "📝 Checking GPU support..."
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✅ CUDA available: $GPU_NAME${NC}"
else
    echo -e "${YELLOW}⚠️  No GPU detected - will use CPU (slower)${NC}"
    WARNINGS=$((WARNINGS+1))
fi

# Check 8: Documentation
echo "📝 Checking documentation..."
DOCS_FOUND=0
for doc in "ENHANCED_OCR_README.md" "COMPLETE_PROJECT_GUIDE.md" "PROJECT_INTEGRATION_COMPLETE.md"; do
    if [ -f "$doc" ]; then
        DOCS_FOUND=$((DOCS_FOUND+1))
    fi
done

if [ $DOCS_FOUND -eq 3 ]; then
    echo -e "${GREEN}✅ All documentation files found ($DOCS_FOUND/3)${NC}"
else
    echo -e "${YELLOW}⚠️  Some documentation missing ($DOCS_FOUND/3)${NC}"
    WARNINGS=$((WARNINGS+1))
fi

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Verification Summary                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
    echo ""
    echo "Your enhanced OCR system is ready to use."
    echo "Run: ./run_enhanced_ocr.sh"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}✓ System is functional with $WARNINGS warning(s)${NC}"
    echo ""
    echo "You can run the application, but consider addressing warnings."
    echo "Run: ./run_enhanced_ocr.sh"
else
    echo -e "${RED}✗ Found $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors before running the application."

    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo ""
        echo "To install missing packages:"
        echo "  pip install -r ocr_gemini_streamlit_requirements.txt"
    fi
fi

echo ""
echo "For more information, see:"
echo "  - ENHANCED_OCR_QUICK_REFERENCE.txt"
echo "  - COMPLETE_PROJECT_GUIDE.md"
echo ""

exit $ERRORS

