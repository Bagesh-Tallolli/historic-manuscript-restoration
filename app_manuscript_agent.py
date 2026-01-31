"""
ManuscriptVision-Agent — Streamlit UI
Complete pipeline for Sanskrit manuscript processing
"""
import io
import json
import streamlit as st
from PIL import Image
from manuscript_vision_agent import ManuscriptVisionAgent

# Page config
st.set_page_config(
    page_title="ManuscriptVision Agent",
    page_icon="📜",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        padding: 1rem;
        font-weight: bold;
        border-bottom: 3px solid #E74C3C;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #E74C3C;
        border-left: 4px solid #E74C3C;
        padding-left: 1rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .pipeline-step {
        background-color: #ECF0F1;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #3498DB;
    }
    .result-box {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #DEE2E6;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #27AE60;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-medium {
        color: #F39C12;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-low {
        color: #E74C3C;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📜 ManuscriptVision-Agent</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #7F8C8D; margin-bottom: 2rem;">
    Complete Pipeline: Restoration → OCR → Correction → Translation → Verification
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Agent Settings")

    st.markdown("#### 🔑 API Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value="",
        help="Leave empty to use default key"
    )

    st.markdown("---")
    st.markdown("### 📋 Pipeline Steps")

    st.markdown("""
    <div class="pipeline-step">
        <b>Step 1:</b> API-based Image Restoration<br>
        <small>Brightness, contrast, sharpness, denoising</small>
    </div>
    <div class="pipeline-step">
        <b>Step 2:</b> OCR Extraction<br>
        <small>Extract Sanskrit text in Devanagari</small>
    </div>
    <div class="pipeline-step">
        <b>Step 3:</b> Text Correction<br>
        <small>Fix OCR errors, normalize Unicode</small>
    </div>
    <div class="pipeline-step">
        <b>Step 4:</b> Multilingual Translation<br>
        <small>English, Hindi, Kannada</small>
    </div>
    <div class="pipeline-step">
        <b>Step 5:</b> Verification<br>
        <small>Validate and score confidence</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **ManuscriptVision-Agent** uses API-based vision and language models only.
    
    ✓ No custom trained models  
    ✓ Natural readability-first restoration  
    ✓ Accurate OCR and correction  
    ✓ Multilingual translation  
    ✓ Automatic verification
    """)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">📤 Upload Manuscript</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a Sanskrit manuscript image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Upload a clear image of a Sanskrit manuscript"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Manuscript", use_container_width=True)

        # Display image info
        st.caption(f"Size: {image.size[0]} × {image.size[1]} pixels | Mode: {image.mode}")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Process button
if uploaded_file:
    if st.button("🚀 Run Complete Pipeline", type="primary", use_container_width=True):
        st.session_state.processing = True
        st.session_state.results = None

        with st.spinner("🔄 Processing manuscript through complete pipeline..."):
            try:
                # Initialize agent
                agent = ManuscriptVisionAgent(api_key=api_key if api_key else None)

                # Process manuscript
                results = agent.process_manuscript(image)

                st.session_state.results = results
                st.session_state.processing = False
                st.success("✅ Pipeline completed successfully!")

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.session_state.processing = False

# Display results
if st.session_state.results:
    results = st.session_state.results

    # Restored image
    with col2:
        st.markdown('<div class="section-header">🔧 Restored Image</div>', unsafe_allow_html=True)

        if 'restored_image' in results:
            st.image(results['restored_image'], caption="API-Restored Manuscript", use_container_width=True)

            # Download button
            buf = io.BytesIO()
            results['restored_image'].save(buf, format='PNG')
            st.download_button(
                label="📥 Download Restored Image",
                data=buf.getvalue(),
                file_name="restored_manuscript.png",
                mime="image/png",
                use_container_width=True
            )

            # Quality assessment
            if 'restored_image_quality' in results:
                with st.expander("📊 Image Quality Analysis"):
                    st.write(results['restored_image_quality'])

    # Separator
    st.markdown("---")

    # OCR and Translations
    st.markdown('<div class="section-header">📖 OCR & Translation Results</div>', unsafe_allow_html=True)

    # Confidence score at the top
    if 'confidence_score' in results:
        try:
            conf = float(results['confidence_score'])
            if conf >= 0.8:
                conf_class = "confidence-high"
                conf_label = "High Confidence"
            elif conf >= 0.5:
                conf_class = "confidence-medium"
                conf_label = "Medium Confidence"
            else:
                conf_class = "confidence-low"
                conf_label = "Low Confidence"

            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background-color: #F8F9FA; border-radius: 10px; margin-bottom: 1rem;">
                <span class="{conf_class}">Confidence Score: {conf:.2%}</span> ({conf_label})
            </div>
            """, unsafe_allow_html=True)
        except:
            pass

    # Create tabs for different outputs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Raw OCR",
        "✏️ Corrected Sanskrit",
        "🇬🇧 English",
        "🇮🇳 हिन्दी",
        "🇮🇳 ಕನ್ನಡ",
        "💾 JSON"
    ])

    with tab1:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("**Raw OCR Extracted Text (Devanagari)**")
        st.code(results.get('ocr_extracted_text', 'N/A'), language='text')
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("**Corrected Sanskrit Text (Devanagari)**")
        st.code(results.get('corrected_sanskrit_text', 'N/A'), language='text')
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("**English Translation**")
        st.write(results.get('english_translation', 'N/A'))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("**Hindi Translation (हिन्दी अनुवाद)**")
        st.write(results.get('hindi_translation', 'N/A'))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("**Kannada Translation (ಕನ್ನಡ ಅನುವಾದ)**")
        st.write(results.get('kannada_translation', 'N/A'))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        # Prepare JSON output (exclude images)
        json_output = {k: v for k, v in results.items()
                      if k not in ['restored_image', 'original_image']}

        st.json(json_output)

        # Download JSON button
        json_str = json.dumps(json_output, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download JSON Results",
            data=json_str,
            file_name="manuscript_analysis.json",
            mime="application/json",
            use_container_width=True
        )

    # Error display if any
    if 'error' in results:
        st.error(f"⚠️ Warning: {results['error']}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95A5A6; padding: 1rem;">
    <small>
        ManuscriptVision-Agent v1.0 | API-based processing only | 
        No custom trained models | Readability-first restoration
    </small>
</div>
""", unsafe_allow_html=True)

