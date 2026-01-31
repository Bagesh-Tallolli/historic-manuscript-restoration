#!/bin/bash
# Quick launcher for the restored OCR app

clear
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Sanskrit Manuscript OCR + AI Restoration                 ║
║                                                              ║
║     ✅ Restoration Feature: WORKING                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "Starting application..."
echo ""

cd /home/bagesh/EL-project
exec ./run_enhanced_ocr.sh

