#!/bin/bash
# ============================================================================
# FAST PROJECT RUNNER - Sanskrit Manuscript Processing
# ============================================================================

cd /home/bagesh/EL-project
source activate_venv.sh

clear
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        🏛️  SANSKRIT MANUSCRIPT RESTORATION PROJECT                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Streamlit
if pgrep -f "streamlit run app.py" > /dev/null; then
    PORT=$(ps aux | grep streamlit | grep -v grep | grep -oP '8[0-9]{3}' | head -1)
    echo "✅ WEB UI RUNNING: http://localhost:${PORT:-8501}"
    echo ""
else
    echo "⚠️  Web UI not running. Starting..."
    streamlit run app.py --server.port 8501 --server.headless true &
    sleep 5
    echo "✅ WEB UI STARTED: http://localhost:8501"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════════"
echo "🚀 CHOOSE AN OPTION:"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "1️⃣  Open Web UI (Recommended)"
echo "   → http://localhost:8501"
echo "   → Upload images via browser"
echo ""
echo "2️⃣  Run AI Pipeline Agent (HuggingFace OCR + Translation)"
echo "   → Uses TrOCR + Helsinki/Google Translate"
echo "   → Command ready below"
echo ""
echo "3️⃣  Run Standard Pipeline (Tesseract + Google)"
echo "   → Uses Tesseract OCR + Google Translate"
echo "   → Command ready below"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Prompt user
read -p "Enter choice (1/2/3) or press Enter to see commands: " choice

case $choice in
    1)
        echo ""
        echo "🌐 Opening Web UI..."
        echo "   Visit: http://localhost:8501"
        echo ""
        echo "   If browser doesn't open automatically, copy the URL above"
        ;;
    2)
        echo ""
        echo "🤖 Running AI Pipeline Agent..."
        python pipeline_agent.py \
            --image_path data/raw/test/test_0010.jpg \
            --restoration_model checkpoints/kaggle/final.pth \
            --output_dir output/ai_agent
        echo ""
        echo "✅ Check results: output/ai_agent/"
        ;;
    3)
        echo ""
        echo "🔧 Running Standard Pipeline..."
        python main.py \
            --image_path data/raw/test/test_0010.jpg \
            --restoration_model checkpoints/kaggle/final.pth \
            --output_dir output/standard
        echo ""
        echo "✅ Check results: output/standard/"
        ;;
    *)
        echo ""
        echo "📋 MANUAL COMMANDS:"
        echo ""
        echo "# AI Agent Pipeline (HuggingFace models):"
        echo "python pipeline_agent.py \\"
        echo "    --image_path data/raw/test/test_0001.jpg \\"
        echo "    --restoration_model checkpoints/kaggle/final.pth \\"
        echo "    --output_dir output/ai_agent"
        echo ""
        echo "# Standard Pipeline (Tesseract):"
        echo "python main.py \\"
        echo "    --image_path data/raw/test/test_0001.jpg \\"
        echo "    --restoration_model checkpoints/kaggle/final.pth \\"
        echo "    --output_dir output/standard"
        echo ""
        echo "# Web UI:"
        echo "streamlit run app.py"
        echo ""
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "✅ PROJECT READY!"
echo ""
echo "📁 Models: checkpoints/kaggle/ (final.pth, desti.pth)"
echo "📓 Notebook: kaggle_training_notebook.ipynb (Ready for Kaggle)"
echo "🌐 Web UI: http://localhost:8501"
echo "════════════════════════════════════════════════════════════════════"

