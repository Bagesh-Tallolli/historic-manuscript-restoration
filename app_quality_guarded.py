"""
Quality-Guarded Manuscript Vision Pipeline - Streamlit UI
Complete pipeline with ViT restoration model and Gemini API
"""

import streamlit as st
import os
from PIL import Image
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_quality_guarded_pipeline import ManuscriptQualityGuardedPipeline

# Page configuration
st.set_page_config(
    page_title="Quality-Guarded Manuscript Vision",
    page_icon="🔰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .quality-good {
        color: #28a745;
        font-weight: bold;
    }
    .quality-bad {
        color: #dc3545;
        font-weight: bold;
    }
    .devanagari-text {
        font-size: 1.5rem;
        line-height: 2;
        direction: ltr;
    }
    .kannada-text {
        font-size: 1.3rem;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.markdown('<h1 class="main-header">🔰 Quality-Guarded Manuscript Vision Pipeline</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ViT Restoration + Gemini API OCR & Translation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key input
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Enter your Google Gemini API key"
    )

    # VIT Checkpoint selection
    st.subheader("🤖 ViT Model")
    checkpoint_options = {
        "models/trained_models/final.pth": "Final Model",
        "checkpoints/kaggle/final.pth": "Kaggle Final",
        "checkpoints/kaggle/final_converted.pth": "Kaggle Converted",
    }

    # Filter to only show existing checkpoints
    available_checkpoints = {k: v for k, v in checkpoint_options.items() if Path(k).exists()}

    if available_checkpoints:
        selected_checkpoint = st.selectbox(
            "Select Checkpoint",
            options=list(available_checkpoints.keys()),
            format_func=lambda x: available_checkpoints[x]
        )
        st.success(f"✓ Model found: {available_checkpoints[selected_checkpoint]}")
    else:
        st.warning("⚠️ No ViT checkpoints found. Will use PIL-based fallback.")
        selected_checkpoint = "models/trained_models/final.pth"

    st.divider()

    st.subheader("ℹ️ Pipeline Features")
    st.markdown("""
    ✅ **Quality Gate**: Restoration is applied only if it improves readability
    
    ✅ **Fallback Safety**: Uses original image if restoration degrades quality
    
    ✅ **Multi-metric Analysis**: SSIM, PSNR, sharpness, contrast
    
    ✅ **Multilingual**: Sanskrit, English, Hindi, Kannada
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Quality Analysis", "📖 Results"])

with tab1:
    st.header("Upload Sanskrit Manuscript")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload a Sanskrit manuscript image for processing"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")

        # Process button
        if st.button("🚀 Process Manuscript", type="primary", use_container_width=True):
            with st.spinner("Processing manuscript... This may take a moment..."):
                try:
                    # Initialize pipeline
                    if st.session_state.pipeline is None or st.session_state.pipeline.api_key != api_key:
                        st.session_state.pipeline = ManuscriptQualityGuardedPipeline(
                            api_key=api_key,
                            vit_checkpoint=selected_checkpoint
                        )

                    # Process the manuscript
                    results = st.session_state.pipeline.process_manuscript(image)
                    st.session_state.results = results

                    st.success("✅ Processing complete!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)

with tab2:
    if st.session_state.results is not None:
        results = st.session_state.results

        st.header("🔍 Quality Analysis")

        # Image comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(results['original_image'], use_container_width=True)

            if 'original_metrics' in results:
                metrics = results['original_metrics']
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**Quality Metrics:**")
                st.write(f"• Sharpness: {metrics['sharpness']:.3f}")
                st.write(f"• Contrast: {metrics['contrast']:.3f}")
                st.write(f"• Text Clarity: {metrics['text_clarity']:.3f}")
                st.write(f"• Overall: {metrics['overall']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.subheader("✨ Selected Image for OCR")
            st.image(results['selected_image'], use_container_width=True)

            # Decision badge
            if results['image_used'] == 'restored':
                st.success("✅ Restoration Applied (Quality Improved)")
            else:
                st.warning("⚠️ Original Image Used (Restoration Did Not Improve Quality)")

            if results.get('restored_metrics'):
                metrics = results['restored_metrics']
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**Quality Metrics:**")
                st.write(f"• Sharpness: {metrics['sharpness']:.3f}")
                st.write(f"• Contrast: {metrics['contrast']:.3f}")
                st.write(f"• Text Clarity: {metrics['text_clarity']:.3f}")
                st.write(f"• Overall: {metrics['overall']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)

        # Comparison metrics
        st.divider()
        st.subheader("📊 Comparison Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            improvement = results.get('improvement', 0)
            delta_color = "normal" if improvement >= 0 else "inverse"
            st.metric(
                "Quality Improvement",
                f"{improvement:+.3f}",
                delta=f"{improvement*100:+.1f}%",
                delta_color=delta_color
            )

        with col2:
            ssim = results.get('ssim', 0)
            st.metric(
                "SSIM (Similarity)",
                f"{ssim:.3f}",
                help="Structural Similarity Index (higher is better)"
            )

        with col3:
            psnr = results.get('psnr', 0)
            st.metric(
                "PSNR (dB)",
                f"{psnr:.2f}",
                help="Peak Signal-to-Noise Ratio (higher is better)"
            )

        # Decision reason
        st.info(f"**Decision:** {results.get('decision_reason', 'N/A')}")

    else:
        st.info("👆 Upload and process an image in the 'Upload & Process' tab to see quality analysis")

with tab3:
    if st.session_state.results is not None:
        results = st.session_state.results

        st.header("📖 Extracted Text & Translations")

        # Confidence score
        if 'confidence_score' in results:
            confidence = results['confidence_score']
            if confidence > 0.8:
                st.success(f"✅ High Confidence: {confidence:.2%}")
            elif confidence > 0.6:
                st.warning(f"⚠️ Medium Confidence: {confidence:.2%}")
            else:
                st.error(f"❌ Low Confidence: {confidence:.2%}")

        st.divider()

        # OCR Extracted Text
        if 'ocr_extracted_text' in results and results['ocr_extracted_text']:
            st.subheader("🔤 OCR Extracted Text (Raw)")
            st.markdown(f'<div class="devanagari-text">{results["ocr_extracted_text"]}</div>', unsafe_allow_html=True)
            st.divider()

        # Corrected Sanskrit Text
        if 'corrected_sanskrit_text' in results and results['corrected_sanskrit_text']:
            st.subheader("✍️ Corrected Sanskrit Text")
            st.markdown(f'<div class="devanagari-text">{results["corrected_sanskrit_text"]}</div>', unsafe_allow_html=True)

            # Copy button
            if st.button("📋 Copy Sanskrit Text"):
                st.code(results['corrected_sanskrit_text'], language=None)

            st.divider()

        # Translations
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'english_translation' in results and results['english_translation']:
                st.subheader("🇬🇧 English")
                st.write(results['english_translation'])

        with col2:
            if 'hindi_translation' in results and results['hindi_translation']:
                st.subheader("🇮🇳 हिन्दी")
                st.markdown(f'<div class="devanagari-text">{results["hindi_translation"]}</div>', unsafe_allow_html=True)

        with col3:
            if 'kannada_translation' in results and results['kannada_translation']:
                st.subheader("🇮🇳 ಕನ್ನಡ")
                st.markdown(f'<div class="kannada-text">{results["kannada_translation"]}</div>', unsafe_allow_html=True)

        st.divider()

        # Processing notes
        if 'processing_notes' in results and results['processing_notes']:
            with st.expander("📝 Processing Notes"):
                st.write(results['processing_notes'])

        # Image quality assessment
        if 'image_quality_assessment' in results:
            with st.expander("🔍 Image Quality Assessment"):
                assessment = results['image_quality_assessment']
                st.write(f"**Score:** {assessment.get('score', 'N/A')}")
                st.write(f"**Description:** {assessment.get('description', 'N/A')}")

        # Export results
        st.divider()
        st.subheader("💾 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # Prepare JSON export (exclude images)
            export_data = {k: v for k, v in results.items() if k not in ['original_image', 'selected_image']}
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="manuscript_results.json",
                mime="application/json"
            )

        with col2:
            # Prepare text export
            text_export = f"""SANSKRIT MANUSCRIPT ANALYSIS RESULTS
=======================================

Image Used: {results.get('image_used', 'N/A')}
Restoration Applied: {results.get('restoration_applied', 'N/A')}
Confidence Score: {results.get('confidence_score', 'N/A')}

CORRECTED SANSKRIT TEXT:
{results.get('corrected_sanskrit_text', 'N/A')}

ENGLISH TRANSLATION:
{results.get('english_translation', 'N/A')}

HINDI TRANSLATION:
{results.get('hindi_translation', 'N/A')}

KANNADA TRANSLATION:
{results.get('kannada_translation', 'N/A')}

PROCESSING NOTES:
{results.get('processing_notes', 'N/A')}
"""
            st.download_button(
                label="📥 Download TXT",
                data=text_export,
                file_name="manuscript_results.txt",
                mime="text/plain"
            )

    else:
        st.info("👆 Upload and process an image in the 'Upload & Process' tab to see results")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔰 <strong>Quality-Guarded Manuscript Vision Pipeline</strong></p>
    <p>ViT Restoration Model + Gemini API | Ensuring Quality Never Degrades</p>
</div>
""", unsafe_allow_html=True)

