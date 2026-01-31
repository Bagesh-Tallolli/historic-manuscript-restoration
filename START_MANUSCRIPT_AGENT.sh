#!/bin/bash
# Startup script for ManuscriptVision-Agent

clear
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║           ManuscriptVision-Agent — Complete Pipeline           ║
║                                                                ║
║  📜 Sanskrit Manuscript Processing                             ║
║  🔧 API-based Restoration (No Custom Models)                   ║
║  📖 OCR Extraction & Correction                                ║
║  🌍 Multilingual Translation (EN, HI, KN)                      ║
║  ✓  Automatic Verification                                     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "🚀 Starting ManuscriptVision-Agent..."
echo ""

# Check if we're in the project directory
if [ ! -f "manuscript_vision_agent.py" ]; then
    echo "❌ Error: manuscript_vision_agent.py not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check for virtual environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo ""
    echo "📥 Installing dependencies..."
    pip install -r requirements_manuscript_agent.txt
fi

# Check if google-genai is installed
if ! python -c "import google.genai" 2>/dev/null; then
    echo ""
    echo "📥 Installing Google GenAI..."
    pip install google-genai
fi

echo ""
echo "✅ All dependencies ready!"
echo ""
echo "🌐 Starting web interface..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Access the application at: http://localhost:8501"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start Streamlit
streamlit run app_manuscript_agent.py --server.port 8501 --server.headless true

