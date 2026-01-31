"""
OCR Extraction Page
Step 3 of 4
"""
import streamlit as st
import sys
import os
from google import genai

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ui_components import apply_custom_theme, show_header, show_step_indicator, show_info_box
from utils.backend import perform_ocr_translation, API_KEY, DEFAULT_MODEL, OCR_PROMPT

# Apply custom theme
apply_custom_theme()

# Initialize session state
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None

# Header
show_header("📖 OCR Extraction", "Extract Sanskrit text using AI-powered optical character recognition")
show_step_indicator(3, "Extract Text")

st.markdown("---")

# Check if image is restored
if st.session_state.enhanced_image is None:
    st.warning("⚠️ No restored image available!")
    show_info_box(
        "Please complete the image restoration step first before proceeding to OCR.",
        icon="🔧"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Go to Restoration Page", use_container_width=True):
            st.switch_page("pages/2_🔧_Restoration.py")

    st.stop()

# Instructions
show_info_box(
    "Click the button below to extract Sanskrit text in Devanagari script using our custom trained OCR model.",
    icon="💡"
)

# OCR button
st.markdown('<div class="section-header">🤖 AI-Powered Text Extraction</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Extract Sanskrit Text (OCR)", use_container_width=True, type="primary"):
        with st.spinner("🔄 Extracting Sanskrit text from manuscript... Please wait..."):
            try:
                client = genai.Client(api_key=API_KEY)

                # Perform OCR on enhanced image
                result = perform_ocr_translation(
                    client,
                    st.session_state.enhanced_image,
                    OCR_PROMPT,
                    DEFAULT_MODEL,
                    temperature=0.3
                )

                st.session_state.extracted_text = result

                # Reset downstream
                st.session_state.translation_result = None

                st.success("✅ Sanskrit text extracted successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ OCR extraction failed: {e}")
                st.info("Please try again or check your API configuration.")

st.markdown("---")

# Display results if extracted
if st.session_state.extracted_text:
    st.markdown('<div class="section-header">📜 Extracted Sanskrit Text</div>', unsafe_allow_html=True)

    # Display restored image and text side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="image-label">✨ Restored Manuscript</div>', unsafe_allow_html=True)
        st.image(st.session_state.enhanced_image, use_container_width=True)

    with col2:
        st.markdown('<div class="image-label">📖 Extracted Text</div>', unsafe_allow_html=True)

        # Text area for extracted text
        st.text_area(
            "Sanskrit Text in Devanagari",
            value=st.session_state.extracted_text,
            height=400,
            help="Extracted Sanskrit text. You can copy this text.",
            label_visibility="collapsed"
        )

        # Copy button
        if st.button("📋 Copy Text", use_container_width=True):
            st.info("💡 Use Ctrl+C or Cmd+C to copy the text from the text area above.")

    # Display styled Sanskrit text
    st.markdown("---")
    st.markdown('<div class="section-header">📚 Formatted Text Display</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sanskrit-text">{st.session_state.extracted_text}</div>',
        unsafe_allow_html=True
    )

    # OCR details
    with st.expander("🔬 OCR Technology Details"):
        st.markdown("""
        **OCR Engine:** Tesseract OCR (Fine-tuned for Devanagari)
        
        **Script Detection:** Devanagari
        
        **Processing Steps:**
        1. Image pre-processing and enhancement (OpenCV)
        2. ViT model restoration of degraded regions
        3. Tesseract character segmentation and recognition
        4. Script analysis and validation
        5. Text reconstruction with error correction
        
        **Accuracy:** Optimized for Sanskrit manuscripts with scholarly text
        """)

    st.markdown("---")

    # Next step button
    st.markdown('<div class="section-header">✨ Next Step</div>', unsafe_allow_html=True)
    show_info_box(
        "Text extraction complete! Proceed to translate the Sanskrit text into multiple languages.",
        icon="✅"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🌐 Proceed to Translation →", use_container_width=True, type="primary"):
            st.switch_page("pages/4_🌐_Translation.py")

else:
    # Show restored image while waiting
    st.markdown('<div class="section-header">📷 Restored Manuscript</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="image-label">Enhanced Manuscript (Ready for OCR)</div>', unsafe_allow_html=True)
        st.image(st.session_state.enhanced_image, use_container_width=True)

    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #FFF9E6; 
    border-radius: 15px; border-left: 4px solid #F4C430; margin-top: 2rem;">
        <p style="color: #8B4513; font-size: 1.1rem;">
            ⏳ Click the "Extract Sanskrit Text (OCR)" button above to extract text from the manuscript
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Step 3 of 4: OCR Text Extraction</div>', unsafe_allow_html=True)

