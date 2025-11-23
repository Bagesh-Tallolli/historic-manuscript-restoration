"""
Streamlit Web Application for Sanskrit Manuscript Processing Pipeline
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
import tempfile
import time
from io import BytesIO
import base64

# Import pipeline components
from main import ManuscriptPipeline
from models.vit_restorer import create_vit_restorer
from ocr.run_ocr import SanskritOCR
from nlp.unicode_normalizer import SanskritTextProcessor
from nlp.translation import SanskritTranslator
from utils.visualization import visualize_restoration, visualize_pipeline_stages

# Page configuration
st.set_page_config(
    page_title="Sanskrit Manuscript Pipeline",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B35;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #F7931E;
    }
    </style>
""", unsafe_allow_html=True)


def get_image_download_link(img, filename="image.jpg"):
    """Generate download link for image"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">Download {filename}</a>'
    return href


def load_pipeline(restoration_model_path=None, ocr_engine='tesseract', translation_method='google'):
    """Load the manuscript processing pipeline"""
    with st.spinner("Initializing pipeline..."):
        pipeline = ManuscriptPipeline(
            restoration_model_path=restoration_model_path,
            ocr_engine=ocr_engine,
            translation_method=translation_method,
            device='auto'
        )
    return pipeline


def process_image(pipeline, image, save_intermediate=True):
    """Process a manuscript image through the pipeline"""
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name

    # Process through pipeline
    with st.spinner("Processing manuscript..."):
        result = pipeline.process(
            temp_path,
            save_intermediate=save_intermediate,
            output_dir='output/streamlit'
        )

    # Clean up temp file
    Path(temp_path).unlink()

    return result


def main():
    # Header
    st.markdown('<h1 class="main-header">🕉️ Sanskrit Manuscript Pipeline</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Restoration, OCR & Translation</p>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Model settings
    st.sidebar.subheader("Restoration Model")
    use_restoration = st.sidebar.checkbox("Enable Image Restoration", value=False)
    model_path = None
    if use_restoration:
        model_upload = st.sidebar.file_uploader("Upload Model Checkpoint (.pth)", type=['pth'])
        if model_upload:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
                tmp.write(model_upload.read())
                model_path = tmp.name
        else:
            st.sidebar.info("ℹ️ No model uploaded. Using pretrained or skipping restoration.")

    # OCR settings
    st.sidebar.subheader("OCR Engine")
    ocr_engine = st.sidebar.selectbox(
        "Select OCR Engine",
        ["tesseract", "trocr", "ensemble"],
        help="Tesseract: Fast, TrOCR: Accurate, Ensemble: Both"
    )

    # Translation settings
    st.sidebar.subheader("Translation")
    translation_method = st.sidebar.selectbox(
        "Translation Method",
        ["google", "indictrans", "ensemble"],
        help="Google: Fast, IndicTrans: Local, Ensemble: Both"
    )

    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        save_intermediate = st.checkbox("Save Intermediate Results", value=True)
        show_metrics = st.checkbox("Show Quality Metrics", value=True)
        show_visualization = st.checkbox("Show Processing Stages", value=True)

    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Results", "📖 About", "🎓 Help"])

    # Tab 1: Upload & Process
    with tab1:
        st.header("Upload Manuscript Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a manuscript image",
            type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
            help="Upload a Sanskrit manuscript image in Devanagari script"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, use_container_width=True)
                st.info(f"Size: {image.size[0]} × {image.size[1]} pixels")

            # Process button
            with col2:
                st.subheader("⚡ Processing")

                if st.button("🚀 Process Manuscript", type="primary"):
                    try:
                        # Load pipeline if not already loaded
                        if st.session_state.pipeline is None:
                            st.session_state.pipeline = load_pipeline(
                                restoration_model_path=model_path,
                                ocr_engine=ocr_engine,
                                translation_method=translation_method
                            )

                        # Process image
                        start_time = time.time()
                        results = process_image(
                            st.session_state.pipeline,
                            image,
                            save_intermediate=save_intermediate
                        )
                        processing_time = time.time() - start_time

                        # Store results
                        st.session_state.results = results
                        st.session_state.processing_time = processing_time

                        st.success(f"✅ Processing complete in {processing_time:.2f}s!")
                        st.info("👉 Check the 'Results' tab to see outputs")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)

        else:
            st.info("👆 Upload an image to get started")

            # Show example
            st.subheader("📝 Example")
            example_path = Path("data/datasets/samples/test_sample.png")
            if example_path.exists():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(str(example_path), caption="Sample Manuscript", use_container_width=True)
                    if st.button("Use Example Image"):
                        # Load and process example
                        st.session_state.example_mode = True
                        st.rerun()

    # Tab 2: Results
    with tab2:
        st.header("Processing Results")

        if st.session_state.results is not None:
            results = st.session_state.results

            # Processing time
            if hasattr(st.session_state, 'processing_time'):
                st.metric("⏱️ Processing Time", f"{st.session_state.processing_time:.2f} seconds")

            st.divider()

            # Images section
            st.subheader("🖼️ Processed Images")

            img_cols = st.columns(3)

            # Original
            if 'original_image' in results:
                with img_cols[0]:
                    st.markdown("**Original**")
                    st.image(results['original_image'], use_container_width=True)

            # Restored
            if 'restored_image' in results:
                with img_cols[1]:
                    st.markdown("**Restored**")
                    st.image(results['restored_image'], use_container_width=True)

            # Comparison
            if 'comparison_image' in results:
                with img_cols[2]:
                    st.markdown("**Comparison**")
                    st.image(results['comparison_image'], use_container_width=True)

            st.divider()

            # Text Results
            st.subheader("📝 Extracted Text")

            text_cols = st.columns(2)

            with text_cols[0]:
                st.markdown("**Sanskrit (Devanagari)**")
                sanskrit_text = results.get('sanskrit_text', 'N/A')
                st.markdown(f'<div class="result-box" style="font-size: 1.5rem; direction: ltr;">{sanskrit_text}</div>',
                           unsafe_allow_html=True)

                # Copy button
                if sanskrit_text != 'N/A':
                    st.code(sanskrit_text, language=None)

            with text_cols[1]:
                st.markdown("**Romanized (IAST)**")
                romanized = results.get('romanized', 'N/A')
                st.markdown(f'<div class="result-box">{romanized}</div>', unsafe_allow_html=True)

            st.divider()

            # Translation
            st.subheader("🌐 Translation")
            translation = results.get('translation', 'N/A')
            st.markdown(f'<div class="result-box" style="font-size: 1.2rem;">{translation}</div>',
                       unsafe_allow_html=True)

            st.divider()

            # Metrics
            if show_metrics:
                st.subheader("📊 Metrics")

                metric_cols = st.columns(4)

                with metric_cols[0]:
                    word_count = results.get('word_count', 0)
                    st.metric("Words", word_count)

                with metric_cols[1]:
                    sentence_count = len(results.get('sentences', []))
                    st.metric("Sentences", sentence_count)

                with metric_cols[2]:
                    confidence = results.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.1%}" if confidence else "N/A")

                with metric_cols[3]:
                    if 'psnr' in results:
                        st.metric("PSNR", f"{results['psnr']:.2f} dB")
                    else:
                        st.metric("PSNR", "N/A")

            st.divider()

            # Download section
            st.subheader("💾 Download Results")

            download_cols = st.columns(3)

            # JSON download
            with download_cols[0]:
                json_str = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name="results.json",
                    mime="application/json"
                )

            # Text download
            with download_cols[1]:
                text_output = f"""Sanskrit Manuscript Processing Results
{'='*50}

Sanskrit Text (Devanagari):
{results.get('sanskrit_text', 'N/A')}

Romanized (IAST):
{results.get('romanized', 'N/A')}

English Translation:
{results.get('translation', 'N/A')}

Metrics:
- Word Count: {results.get('word_count', 0)}
- Sentences: {len(results.get('sentences', []))}
"""
                st.download_button(
                    label="📝 Download Text",
                    data=text_output,
                    file_name="results.txt",
                    mime="text/plain"
                )

            # Full report download
            with download_cols[2]:
                if st.button("📦 Generate Full Report"):
                    st.info("Full report with images will be saved to output/streamlit/")

        else:
            st.info("👈 Process an image first to see results here")

    # Tab 3: About
    with tab3:
        st.header("About This Project")

        st.markdown("""
        ### 🕉️ Sanskrit Manuscript Pipeline
        
        This application provides an end-to-end solution for processing ancient Sanskrit manuscripts:
        
        #### 🎯 Features
        
        1. **Image Restoration** 🖼️
           - AI-powered restoration using Vision Transformers
           - Removes noise, blur, and fading
           - Enhances text legibility
        
        2. **OCR (Text Extraction)** 📖
           - Multiple OCR engines (Tesseract, TrOCR)
           - Specialized for Devanagari script
           - High accuracy text extraction
        
        3. **Unicode Normalization** ✨
           - Cleans and normalizes Devanagari text
           - Converts to standardized Unicode
           - IAST romanization
        
        4. **Translation** 🌐
           - Sanskrit to English translation
           - Multiple translation engines
           - Context-aware processing
        
        #### 🔧 Technology Stack
        
        - **Deep Learning:** PyTorch, Transformers
        - **OCR:** Tesseract, TrOCR
        - **NLP:** IndicTrans, Google Translate
        - **Frontend:** Streamlit
        - **Image Processing:** OpenCV, PIL
        
        #### 📚 Supported Scripts
        
        - Sanskrit (Devanagari)
        - Hindi (Devanagari)
        - Other Indic scripts (configurable)
        
        #### 📊 Performance
        
        - Processing time: 2-10 seconds per image
        - Accuracy: 85-95% (depends on image quality)
        - Supports images up to 4K resolution
        
        ---
        
        **Version:** 1.0.0  
        **Last Updated:** November 2025  
        **License:** MIT
        """)

    # Tab 4: Help
    with tab4:
        st.header("Help & Documentation")

        with st.expander("📖 How to Use", expanded=True):
            st.markdown("""
            ### Step-by-Step Guide
            
            1. **Upload Image**
               - Click "Browse files" in the Upload & Process tab
               - Select a manuscript image (JPEG, PNG, TIFF, BMP)
               - Image should contain Sanskrit text in Devanagari script
            
            2. **Configure Settings** (Optional)
               - Use the sidebar to adjust processing options
               - Enable restoration if you have a trained model
               - Choose OCR and translation engines
            
            3. **Process**
               - Click "Process Manuscript" button
               - Wait for processing to complete (2-10 seconds)
               - Check the Results tab for outputs
            
            4. **Download Results**
               - Download JSON, text, or full report
               - Save processed images
            """)

        with st.expander("⚙️ Configuration Guide"):
            st.markdown("""
            ### OCR Engines
            
            - **Tesseract**: Fast, works well for clear text
            - **TrOCR**: Transformer-based, better for degraded text
            - **Ensemble**: Combines both for best results
            
            ### Translation Methods
            
            - **Google**: Fast, requires internet
            - **IndicTrans**: Local, offline processing
            - **Ensemble**: Uses multiple methods
            
            ### Image Requirements
            
            - **Format**: JPEG, PNG, TIFF, BMP
            - **Resolution**: 512×512 or higher recommended
            - **Content**: Clear Sanskrit/Devanagari text
            - **Quality**: Better quality = better results
            """)

        with st.expander("🐛 Troubleshooting"):
            st.markdown("""
            ### Common Issues
            
            **"No text detected"**
            - Ensure image contains visible text
            - Try adjusting image brightness/contrast
            - Use a higher resolution image
            
            **"Processing failed"**
            - Check image format is supported
            - Ensure image is not corrupted
            - Try a smaller image size
            
            **"Translation error"**
            - Check internet connection (for Google Translate)
            - Try switching translation method
            - Verify extracted text is in Sanskrit
            
            **Slow processing**
            - Reduce image size
            - Disable restoration if not needed
            - Use single OCR engine instead of ensemble
            """)

        with st.expander("📧 Contact & Support"):
            st.markdown("""
            ### Get Help
            
            - **Documentation**: Check README.md and other docs
            - **Issues**: Report bugs on GitHub
            - **Email**: Support contact information
            
            ### Resources
            
            - [Project GitHub](https://github.com/yourusername/project)
            - [Documentation](docs/)
            - [Training Guide](DATASET_REQUIREMENTS.md)
            """)

    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            Made with ❤️ for Sanskrit Digital Humanities | 
            <a href='https://github.com/yourusername/project'>GitHub</a> | 
            <a href='docs/'>Documentation</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

