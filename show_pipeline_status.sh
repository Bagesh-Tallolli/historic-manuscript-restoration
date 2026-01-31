#!/bin/bash

# Display the current status of the Quality-Guarded Pipeline

clear

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🔰 QUALITY-GUARDED MANUSCRIPT PIPELINE - ACTIVE STATUS      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

EOF

echo "📅 Date: $(date '+%B %d, %Y %H:%M:%S')"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "🌐 APPLICATION STATUS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Check if quality-guarded pipeline is running
if ps aux | grep -q "[s]treamlit run app_quality_guarded.py"; then
    PORT=$(ps aux | grep "[s]treamlit run app_quality_guarded.py" | grep -oP 'port \K[0-9]+' || echo "8501")
    PID=$(ps aux | grep "[s]treamlit run app_quality_guarded.py" | awk '{print $2}' | head -1)

    echo "✅ Quality-Guarded Pipeline: RUNNING"
    echo "   • Status: ✅ Active"
    echo "   • Port: $PORT"
    echo "   • Process ID: $PID"
    echo "   • URL: http://localhost:$PORT"
    echo ""

    # Check health
    if curl -s http://localhost:$PORT/_stcore/health > /dev/null 2>&1; then
        echo "   • Health Check: ✅ Responding"
    else
        echo "   • Health Check: ⚠️  Not responding"
    fi
else
    echo "❌ Quality-Guarded Pipeline: NOT RUNNING"
    echo ""
    echo "To start it, run:"
    echo "   ./run_quality_guarded_pipeline.sh"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "🔰 PIPELINE FEATURES"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "✅ ViT Restoration Model"
echo "   • AI-based image enhancement"
echo "   • PIL fallback if model unavailable"
echo ""
echo "✅ Quality Gate System"
echo "   • Automatic quality comparison"
echo "   • Uses restored ONLY if better"
echo "   • Falls back to original if worse"
echo "   • Guaranteed no quality degradation"
echo ""
echo "✅ Gemini Vision API"
echo "   • OCR text extraction"
echo "   • Text correction"
echo "   • Multilingual translation"
echo ""
echo "✅ Quality Metrics"
echo "   • Sharpness analysis"
echo "   • Contrast measurement"
echo "   • Text clarity evaluation"
echo "   • SSIM & PSNR comparison"
echo ""
echo "✅ Multilingual Output"
echo "   • English translation"
echo "   • Hindi translation (हिन्दी)"
echo "   • Kannada translation (ಕನ್ನಡ)"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "🚀 QUICK ACCESS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Open your browser and navigate to:"
echo ""
echo "   🌐 http://localhost:8501"
echo ""
echo "Or if accessing remotely:"
echo ""
echo "   🌐 http://YOUR_SERVER_IP:8501"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "📋 HOW TO USE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "1. Open http://localhost:8501 in your browser"
echo "2. Upload a Sanskrit manuscript image"
echo "3. Click 'Run Quality-Guarded Pipeline'"
echo "4. Review quality comparison and results"
echo "5. Export JSON results and images"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "🛠️  MANAGEMENT COMMANDS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Check Status:"
echo "   ./show_pipeline_status.sh"
echo ""
echo "Test Pipeline:"
echo "   source venv/bin/activate"
echo "   python test_pipeline.py your_image.jpg"
echo ""
echo "Stop Pipeline:"
echo "   pkill -f 'streamlit run app_quality_guarded.py'"
echo ""
echo "Restart Pipeline:"
echo "   ./run_quality_guarded_pipeline.sh"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "📚 DOCUMENTATION"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "• QUICK_START_GUIDE.md - Getting started"
echo "• QUALITY_GUARDED_PIPELINE_README.md - Technical details"
echo "• IMPLEMENTATION_COMPLETE.md - Implementation info"
echo "• PIPELINE_RUNNING_STATUS.txt - Test results"
echo ""

cat << 'EOF'
═══════════════════════════════════════════════════════════════════

╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  🔰 Quality-Guarded Manuscript Vision Pipeline                  ║
║                                                                  ║
║  Restoration that never degrades. Intelligence that protects.   ║
║                                                                  ║
║  🌐 Access Now: http://localhost:8501                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

EOF

