"""
Clean Streamlit Web Application for Sanskrit Manuscript Processing
Using ONLY: ViT Restoration + Google Cloud Vision + Gemini

UI Layout:
1. Upload Image
2. Side-by-side: Original | Restored
3. Results Table: Sanskrit | English
4. Download buttons
"""

import streamlit as st
from PIL import Image
import tempfile
from pathlib import Path
import json
import os

# Import clean pipeline
from pipeline_clean import SanskritManuscriptPipeline

# Page configuration
st.set_page_config(
    page_title="Sanskrit Manuscript Pipeline",
    page_icon="🕉️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .image-box {
        text-align: center;
        padding: 10px;
        border: 2px solid #ddd;
        border-radius: 10px;
        background: #f9f9f9;
    }
    .result-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
        font-size: 1.1rem;
    }
    .result-table th {
        background-color: #667eea;
        color: white;
        padding: 15px;
        text-align: left;
        font-weight: bold;
        border: 1px solid #ddd;
    }
    .result-table td {
        padding: 15px;
        border: 1px solid #ddd;
        vertical-align: top;
        line-height: 1.8;
    }
    .devanagari-text {
        font-family: 'Noto Sans Devanagari', 'Mangal', sans-serif;
        font-size: 1.3rem;
        line-height: 2;
    }
    .english-text {
        font-family: 'Open Sans', sans-serif;
        font-size: 1.1rem;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🕉️ Sanskrit Manuscript Pipeline</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">'
           'ViT Restoration → Google Cloud Vision → Gemini Correction & Translation</p>',
           unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'restored_image' not in st.session_state:
    st.session_state.restored_image = None

# API Configuration (sidebar)
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model path
    model_path = st.text_input(
        "Restoration Model Path",
        value="checkpoints/kaggle/final.pth",
        help="Path to ViT restoration model checkpoint"
    )

    st.markdown("---")
    st.subheader("🔑 API Keys")

    # Google Cloud Vision API key
    google_api_key = st.text_input(
        "Google Cloud Vision API Key",
        value=os.getenv('GOOGLE_CLOUD_API_KEY', ''),
        type="password",
        help="Your Google Cloud Vision API key"
    )

    # Gemini API key
    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.getenv('GEMINI_API_KEY', ''),
        type="password",
        help="Your Gemini API key"
    )

    st.markdown("---")

    # Status
    if google_api_key and gemini_key and Path(model_path).exists():
        st.success("✅ All configurations set!")
    else:
        st.warning("⚠️ Please configure API keys")

# Main content
st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "📤 Upload Sanskrit Manuscript Image",
    type=['jpg', 'jpeg', 'png', 'tiff'],
    help="Upload a degraded Sanskrit manuscript image"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.session_state.original_image = image

    # Process button
    if st.button("🚀 PROCESS MANUSCRIPT", type="primary", use_container_width=True):

        # Validate configuration
        if not google_api_key or not gemini_key:
            st.error("❌ Please configure Google Cloud Vision API key and Gemini API key in the sidebar!")
            st.stop()

        if not Path(model_path).exists():
            st.error(f"❌ Model not found: {model_path}")
            st.stop()

        try:
            # Initialize pipeline
            with st.spinner("🔄 Initializing pipeline..."):
                pipeline = SanskritManuscriptPipeline(
                    restoration_model_path=model_path,
                    google_api_key=google_api_key,
                    gemini_api_key=gemini_key,
                    device='auto'
                )

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                temp_path = tmp_file.name

            # Process with progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🎨 Stage 1/4: Restoring image...")
            progress_bar.progress(25)

            # Create output directory
            output_dir = Path('outputs/streamlit')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process
            status_text.text("📖 Stage 2/4: Extracting text with Google Vision...")
            progress_bar.progress(50)

            status_text.text("🔧 Stage 3/4: Correcting text with Gemini...")
            progress_bar.progress(75)

            status_text.text("🌍 Stage 4/4: Translating to English...")

            result = pipeline.process(temp_path, str(output_dir))

            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except:
                pass

            # Load restored image
            restored_img = Image.open(result['restored_image_path'])

            # Store results
            st.session_state.results = result
            st.session_state.restored_image = restored_img

            st.success(f"✅ Processing complete in {result['metadata']['processing_time_seconds']}s!")

        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.exception(e)
            st.stop()

# Display results if available
if st.session_state.results is not None:
    st.markdown("---")
    st.markdown("## 🖼️ Image Comparison: Original vs Restored")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="image-box">', unsafe_allow_html=True)
        st.markdown("#### 📷 Original Manuscript")
        st.image(st.session_state.original_image, use_container_width=True)
        st.caption(f"Size: {st.session_state.original_image.size[0]} × {st.session_state.original_image.size[1]} px")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="image-box">', unsafe_allow_html=True)
        st.markdown("#### ✨ Restored & Enhanced")
        st.image(st.session_state.restored_image, use_container_width=True)
        st.caption("Noise removed, clarity improved")
        st.markdown('</div>', unsafe_allow_html=True)

    # Results table
    st.markdown("---")
    st.markdown("## 📊 Text Extraction & Translation Results")

    results = st.session_state.results

    # Metadata
    st.markdown(
        f'<div style="margin-bottom: 20px;">'
        f'<span style="background-color: #10b981; color: white; padding: 5px 15px; '
        f'border-radius: 20px; font-weight: bold; font-size: 0.9rem;">'
        f'✅ OCR Confidence: {results["confidence"]["ocr_score"]}</span> '
        f'<span style="color: #666; font-size: 1.1rem; font-weight: 500; margin-left: 15px;">'
        f'📝 Words: {results["metadata"]["word_count"]} | '
        f'📄 Characters: {results["metadata"]["character_count"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Create results table
    sanskrit_text = results['sanskrit_corrected']
    english_text = results['translation_english']

    table_html = f"""
    <table class="result-table">
        <thead>
            <tr>
                <th style="width: 50%;">📜 Sanskrit Text (Corrected)<br><small>(Complete Devanagari Paragraph)</small></th>
                <th style="width: 50%;">🌍 English Translation<br><small>(Full Paragraph Translation)</small></th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="devanagari-text">{sanskrit_text}</td>
                <td class="english-text">{english_text}</td>
            </tr>
        </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)

    # Download buttons
    st.markdown("---")
    st.markdown("### 💾 Download Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Restored image
        with open(results['restored_image_path'], 'rb') as f:
            st.download_button(
                "⬇️ Restored Image",
                f,
                file_name="restored_manuscript.png",
                mime="image/png",
                use_container_width=True
            )

    with col2:
        # Sanskrit text
        st.download_button(
            "⬇️ Sanskrit Text",
            sanskrit_text.encode('utf-8'),
            file_name="sanskrit_corrected.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col3:
        # English translation
        st.download_button(
            "⬇️ English Translation",
            english_text.encode('utf-8'),
            file_name="english_translation.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col4:
        # Full JSON
        json_str = json.dumps(results, ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ Full JSON",
            json_str.encode('utf-8'),
            file_name="pipeline_output.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #999; font-size: 0.9rem;">'
    '🕉️ Sanskrit Manuscript Pipeline | '
    'ViT Restoration + Google Cloud Vision + Gemini<br>'
    'Clean Implementation - No Tesseract, No TrOCR, No Old Models'
    '</p>',
    unsafe_allow_html=True
)

