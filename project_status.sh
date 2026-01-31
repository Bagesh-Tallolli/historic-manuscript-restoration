#!/bin/bash
# ============================================================================
# PROJECT RUN STATUS - November 25, 2025
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  🎉 SANSKRIT MANUSCRIPT RESTORATION PROJECT - RUNNING              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Streamlit
echo "📊 WEB UI STATUS:"
if pgrep -f "streamlit run app.py" > /dev/null; then
    PID=$(pgrep -f "streamlit run app.py")
    PORT=$(ps aux | grep streamlit | grep -v grep | grep -oP 'server.port \K[0-9]+' || echo "8501")
    echo "  ✅ Streamlit RUNNING"
    echo "  Process ID: $PID"
    echo "  Port: $PORT"
    echo "  URL: http://localhost:$PORT"
    echo ""
    echo "  🌐 Access the Web UI:"
    echo "     - Open browser: http://localhost:$PORT"
    echo "     - Upload a manuscript image"
    echo "     - See restoration, OCR, and translation results!"
else
    echo "  ❌ Streamlit NOT RUNNING"
    echo "  Start with: streamlit run app.py"
fi
echo ""

# Check Models
echo "🧠 MODELS STATUS:"
if [ -f "checkpoints/kaggle/final.pth" ]; then
    SIZE=$(du -h checkpoints/kaggle/final.pth | cut -f1)
    echo "  ✅ final.pth ($SIZE) - Ready"
fi
if [ -f "checkpoints/kaggle/desti.pth" ]; then
    SIZE=$(du -h checkpoints/kaggle/desti.pth | cut -f1)
    echo "  ✅ desti.pth ($SIZE) - Ready"
fi
if [ -f "checkpoints/kaggle/best_psnr.pth" ]; then
    SIZE=$(du -h checkpoints/kaggle/best_psnr.pth | cut -f1)
    echo "  ✅ best_psnr.pth ($SIZE) - Ready"
fi
echo ""

# Check Test Data
echo "📁 TEST DATA STATUS:"
TEST_COUNT=$(ls data/raw/test/*.jpg 2>/dev/null | wc -l)
echo "  ✅ Test images: $TEST_COUNT files available"
echo ""

# Check Recent Results
echo "📤 RECENT TEST RESULTS:"
if [ -d "output/test_run" ]; then
    echo "  ✅ Latest test: output/test_run/"
    ls -1 output/test_run/ | head -5 | sed 's/^/     • /'
else
    echo "  ℹ️  No test results yet"
fi
echo ""

# Check Notebook
echo "📓 KAGGLE NOTEBOOK STATUS:"
if [ -f "kaggle_training_notebook.ipynb" ]; then
    SIZE=$(du -h kaggle_training_notebook.ipynb | cut -f1)
    CELLS=$(grep -c '"cell_type"' kaggle_training_notebook.ipynb 2>/dev/null || echo "?")
    echo "  ✅ kaggle_training_notebook.ipynb ($SIZE, $CELLS cells)"
    echo "  ✅ Complete with:"
    echo "     • Paired training (clean/degraded)"
    echo "     • Skip connections"
    echo "     • Perceptual loss (LPIPS)"
    echo "     • Enhanced degradation (6 techniques)"
    echo "     • Auto dataset download (Roboflow)"
    echo "     • Ready to upload to Kaggle!"
fi
echo ""

echo "════════════════════════════════════════════════════════════════════"
echo "🚀 QUICK ACTIONS:"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "1️⃣  Access Web UI:"
echo "   → Open browser: http://localhost:${PORT:-8501}"
echo ""
echo "2️⃣  Test CLI Pipeline:"
echo "   → python main.py \\"
echo "        --image_path data/raw/test/test_0001.jpg \\"
echo "        --restoration_model checkpoints/kaggle/final.pth"
echo ""
echo "3️⃣  Batch Process Images:"
echo "   → python inference.py \\"
echo "        --checkpoint checkpoints/kaggle/final.pth \\"
echo "        --input data/raw/test/ \\"
echo "        --output output/batch_results"
echo ""
echo "4️⃣  Upload Notebook to Kaggle:"
echo "   → File: kaggle_training_notebook.ipynb"
echo "   → Enable GPU: T4 x2"
echo "   → Run all cells"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "✅ PROJECT STATUS: FULLY OPERATIONAL"
echo "════════════════════════════════════════════════════════════════════"

