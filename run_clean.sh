#!/bin/bash

# Run Clean Sanskrit Manuscript Pipeline - Streamlit UI

echo "🕉️ Starting Clean Sanskrit Manuscript Pipeline..."
echo ""
echo "Pipeline: ViT Restoration → Google Cloud Vision → Gemini"
echo ""

# Activate virtual environment
source venv/bin/activate

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Run Streamlit app
streamlit run app_clean.py \
    --server.port 8504 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false

