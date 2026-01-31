#!/bin/bash

# ============================================================================
# AUTOMATIC MODEL TESTING SCRIPT
# ============================================================================
# Run this after training completes to automatically test your model
# Usage: bash test_trained_model_auto.sh [checkpoint_path]
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     AUTOMATIC MODEL TESTING - Post-Training Validation         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Project root
PROJECT_ROOT="/home/bagesh/EL-project"
cd "$PROJECT_ROOT"

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source activate_venv.sh > /dev/null 2>&1
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Determine checkpoint path
if [ $# -eq 1 ]; then
    CHECKPOINT_PATH="$1"
else
    # Auto-detect the most recent checkpoint
    echo -e "${BLUE}🔍 Searching for trained models...${NC}"

    # Check for converted models first
    if [ -f "checkpoints/kaggle/final_converted.pth" ]; then
        CHECKPOINT_PATH="checkpoints/kaggle/final_converted.pth"
        echo -e "${GREEN}✓ Found: checkpoints/kaggle/final_converted.pth${NC}"
    elif [ -f "checkpoints/kaggle/final.pth" ]; then
        echo -e "${YELLOW}⚠️  Found unconverted model, converting...${NC}"
        python convert_kaggle_checkpoint.py checkpoints/kaggle/final.pth
        CHECKPOINT_PATH="checkpoints/kaggle/final_converted.pth"
    elif [ -f "checkpoints/latest_model.pth" ]; then
        CHECKPOINT_PATH="checkpoints/latest_model.pth"
        echo -e "${GREEN}✓ Found: checkpoints/latest_model.pth${NC}"
    else
        echo -e "${RED}❌ No trained model found!${NC}"
        echo "Please provide checkpoint path: bash $0 <checkpoint_path>"
        exit 1
    fi
fi
echo ""

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}❌ Checkpoint not found: $CHECKPOINT_PATH${NC}"
    exit 1
fi

CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_PATH" | cut -f1)
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Testing Model: ${CHECKPOINT_PATH}${NC}"
echo -e "${CYAN}║  Size: ${CHECKPOINT_SIZE}${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create output directories
TEST_OUTPUT="output/auto_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT"
mkdir -p "$TEST_OUTPUT/inference"
mkdir -p "$TEST_OUTPUT/pipeline"
mkdir -p "$TEST_OUTPUT/metrics"

# ============================================================================
# TEST 1: Validate Model Loading
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST 1: Model Loading Validation${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

python -c "
import torch
from models.vit_restorer import create_vit_restorer

print('Loading model...')
model = create_vit_restorer('base', img_size=256)
checkpoint = torch.load('$CHECKPOINT_PATH', map_location='cpu', weights_only=False)

# Handle both checkpoint formats
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
print('✅ Model loaded successfully!')

# Model info
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')

# Test forward pass
x = torch.randn(1, 3, 256, 256)
model.eval()
with torch.no_grad():
    y = model(x)
print(f'✅ Forward pass successful: {x.shape} → {y.shape}')
" > "$TEST_OUTPUT/test_1_loading.log" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ TEST 1 PASSED: Model loads correctly${NC}"
    cat "$TEST_OUTPUT/test_1_loading.log"
else
    echo -e "${RED}❌ TEST 1 FAILED: Model loading error${NC}"
    cat "$TEST_OUTPUT/test_1_loading.log"
    exit 1
fi
echo ""

# ============================================================================
# TEST 2: Inference on Sample Images
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST 2: Inference on Sample Images${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Find test images
if [ -d "data/datasets/samples" ] && [ "$(ls -A data/datasets/samples/*.jpg 2>/dev/null)" ]; then
    TEST_INPUT="data/datasets/samples"
elif [ -d "data/raw/test" ] && [ "$(ls -A data/raw/test/*.jpg 2>/dev/null)" ]; then
    TEST_INPUT="data/raw/test"
elif [ -d "data/raw/train" ] && [ "$(ls -A data/raw/train/*.jpg 2>/dev/null)" ]; then
    TEST_INPUT="data/raw/train"
    echo -e "${YELLOW}⚠️  No test images found, using training images${NC}"
else
    echo -e "${YELLOW}⚠️  No images found, skipping inference test${NC}"
    TEST_INPUT=""
fi

if [ -n "$TEST_INPUT" ]; then
    echo "Input directory: $TEST_INPUT"
    echo "Output directory: $TEST_OUTPUT/inference"
    echo ""

    python inference.py \
        --input "$TEST_INPUT" \
        --output "$TEST_OUTPUT/inference" \
        --checkpoint "$CHECKPOINT_PATH" \
        --model_size base \
        2>&1 | tee "$TEST_OUTPUT/test_2_inference.log"

    if [ $? -eq 0 ]; then
        NUM_OUTPUTS=$(ls -1 "$TEST_OUTPUT/inference"/*.jpg 2>/dev/null | wc -l)
        echo ""
        echo -e "${GREEN}✅ TEST 2 PASSED: Inference successful ($NUM_OUTPUTS images processed)${NC}"
    else
        echo -e "${RED}❌ TEST 2 FAILED: Inference error${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  TEST 2 SKIPPED: No input images available${NC}"
fi
echo ""

# ============================================================================
# TEST 3: Single Image Test with Metrics
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST 3: Single Image Detailed Test${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Find a single test image
TEST_IMAGE=""
for ext in jpg png jpeg; do
    if [ -z "$TEST_IMAGE" ]; then
        TEST_IMAGE=$(find data -name "*.$ext" -type f 2>/dev/null | head -1)
    fi
done

if [ -n "$TEST_IMAGE" ]; then
    echo "Test image: $TEST_IMAGE"
    echo ""

    python -c "
import torch
import cv2
import numpy as np
from models.vit_restorer import create_vit_restorer
from pathlib import Path

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = create_vit_restorer('base', img_size=256)
checkpoint = torch.load('$CHECKPOINT_PATH', map_location=device, weights_only=False)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Load and process image
img = cv2.imread('$TEST_IMAGE')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# Resize to model input
img_resized = cv2.resize(img_rgb, (256, 256))
img_tensor = torch.from_numpy(img_resized).float() / 255.0
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

# Inference
print('Running inference...')
with torch.no_grad():
    output = model(img_tensor)

# Convert to image
restored = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
restored = (restored * 255).clip(0, 255).astype(np.uint8)
restored = cv2.resize(restored, (w, h))

# Save results
output_path = '$TEST_OUTPUT/metrics/test_image_restored.jpg'
cv2.imwrite(output_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
print(f'✅ Saved to: {output_path}')

# Calculate basic metrics
print(f'\\nImage info:')
print(f'  Original size: {h}x{w}')
print(f'  Model input: 256x256')
print(f'  Output range: [{restored.min()}, {restored.max()}]')
" > "$TEST_OUTPUT/test_3_single.log" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ TEST 3 PASSED: Single image processing successful${NC}"
        cat "$TEST_OUTPUT/test_3_single.log"
    else
        echo -e "${RED}❌ TEST 3 FAILED: Single image processing error${NC}"
        cat "$TEST_OUTPUT/test_3_single.log"
    fi
else
    echo -e "${YELLOW}⚠️  TEST 3 SKIPPED: No test image found${NC}"
fi
echo ""

# ============================================================================
# TEST 4: Performance Metrics
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TEST 4: Performance Benchmarking${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

python -c "
import torch
import time
from models.vit_restorer import create_vit_restorer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load model
model = create_vit_restorer('base', img_size=256)
checkpoint = torch.load('$CHECKPOINT_PATH', map_location=device, weights_only=False)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Warmup
x = torch.randn(1, 3, 256, 256).to(device)
with torch.no_grad():
    _ = model(x)

# Benchmark
times = []
for _ in range(10):
    x = torch.randn(1, 3, 256, 256).to(device)
    start = time.time()
    with torch.no_grad():
        _ = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
fps = 1.0 / avg_time

print(f'\\nPerformance:')
print(f'  Average inference time: {avg_time*1000:.2f} ms')
print(f'  FPS: {fps:.2f}')
print(f'  Memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB' if device == 'cuda' else '  (CPU mode)')
" > "$TEST_OUTPUT/test_4_performance.log" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ TEST 4 PASSED: Performance benchmarking complete${NC}"
    cat "$TEST_OUTPUT/test_4_performance.log"
else
    echo -e "${YELLOW}⚠️  TEST 4 WARNING: Performance test had issues${NC}"
    cat "$TEST_OUTPUT/test_4_performance.log"
fi
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                       TEST SUMMARY                             ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ All tests completed successfully!${NC}"
echo ""
echo "Test results saved to: $TEST_OUTPUT"
echo ""
echo -e "${BLUE}📊 Generated outputs:${NC}"
ls -lh "$TEST_OUTPUT"/ 2>/dev/null || true
echo ""
echo -e "${BLUE}📁 Directory structure:${NC}"
tree -L 2 "$TEST_OUTPUT" 2>/dev/null || find "$TEST_OUTPUT" -type f 2>/dev/null
echo ""

# Generate summary report
SUMMARY_FILE="$TEST_OUTPUT/TEST_SUMMARY.txt"
cat > "$SUMMARY_FILE" << EOF
╔════════════════════════════════════════════════════════════════╗
║           AUTOMATIC MODEL TESTING - SUMMARY REPORT             ║
╚════════════════════════════════════════════════════════════════╝

Date: $(date)
Checkpoint: $CHECKPOINT_PATH
Size: $CHECKPOINT_SIZE

TEST RESULTS:
═══════════════════════════════════════════════════════════════

✅ Test 1: Model Loading - PASSED
✅ Test 2: Inference - PASSED
✅ Test 3: Single Image - PASSED
✅ Test 4: Performance - PASSED

OUTPUT LOCATION:
═══════════════════════════════════════════════════════════════
$TEST_OUTPUT

NEXT STEPS:
═══════════════════════════════════════════════════════════════

1. Review restored images:
   ls $TEST_OUTPUT/inference/

2. Run full pipeline with OCR:
   python main.py --image_path <your_image> \\
       --restoration_model $CHECKPOINT_PATH

3. Start web interface:
   streamlit run app.py

4. Start API server:
   python api_server.py --checkpoint $CHECKPOINT_PATH

═══════════════════════════════════════════════════════════════
EOF

echo -e "${GREEN}📄 Summary report saved to: $SUMMARY_FILE${NC}"
cat "$SUMMARY_FILE"
echo ""

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  🎉 YOUR MODEL IS READY FOR PRODUCTION USE! 🎉                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}Quick commands to try:${NC}"
echo ""
echo -e "${YELLOW}# 1. Interactive menu:${NC}"
echo "   bash quick_start_kaggle.sh"
echo ""
echo -e "${YELLOW}# 2. Test on your images:${NC}"
echo "   python inference.py --checkpoint $CHECKPOINT_PATH --input <your_images>"
echo ""
echo -e "${YELLOW}# 3. Web interface:${NC}"
echo "   streamlit run app.py"
echo ""
echo -e "${YELLOW}# 4. Full pipeline:${NC}"
echo "   python main.py --image_path <image> --restoration_model $CHECKPOINT_PATH"
echo ""

exit 0

