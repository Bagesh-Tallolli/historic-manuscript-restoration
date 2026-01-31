"""
Image Restoration Page
Step 2 of 4
"""
import streamlit as st
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ui_components import apply_custom_theme, show_header, show_step_indicator, show_info_box
from utils.backend import enhance_manuscript_simple

# Apply custom theme
apply_custom_theme()

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None

# Header
show_header("🔧 Image Restoration", "Enhance manuscript quality using advanced image processing")
show_step_indicator(2, "Restore Image")

st.markdown("---")

# Check if image is uploaded
if st.session_state.uploaded_image is None:
    st.warning("⚠️ No image uploaded yet!")
    show_info_box(
        "Please upload a manuscript image first before proceeding to restoration.",
        icon="📤"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Go to Upload Page", use_container_width=True):
            st.switch_page("pages/1_📤_Upload.py")

    st.stop()

# Instructions
show_info_box(
    "Click the button below to enhance your manuscript using CLAHE (Contrast Limited Adaptive Histogram Equalization) and Unsharp Mask techniques.",
    icon="💡"
)

# Restoration button
st.markdown('<div class="section-header">🎨 Enhancement Process</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🧹 Restore Manuscript Image", use_container_width=True, type="primary"):
        with st.spinner("🔄 Restoring manuscript... Please wait..."):
            enhanced = enhance_manuscript_simple(st.session_state.uploaded_image)
            st.session_state.enhanced_image = enhanced

            # Reset downstream states
            st.session_state.extracted_text = None
            st.session_state.translation_result = None

            st.success("✅ Image restoration completed successfully!")
            st.balloons()

st.markdown("---")

# Display comparison if restored
if st.session_state.enhanced_image:
    st.markdown('<div class="section-header">📊 Before & After Comparison</div>', unsafe_allow_html=True)

    # Side-by-side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="image-label">📷 Original Manuscript</div>', unsafe_allow_html=True)
        st.image(st.session_state.uploaded_image, use_container_width=True)

    with col2:
        st.markdown('<div class="image-label">✨ Restored Manuscript</div>', unsafe_allow_html=True)
        st.image(st.session_state.enhanced_image, use_container_width=True)

    # Enhancement details
    with st.expander("🔬 Enhancement Techniques Applied"):
        st.markdown("""
        **1. CLAHE (Contrast Limited Adaptive Histogram Equalization)**
        - Improves local contrast
        - Enhances text visibility
        - Adaptive to different image regions
        
        **2. Unsharp Mask**
        - Sharpens edges and text
        - Improves character definition
        - Reduces blur
        
        **Result:** Enhanced clarity for optimal OCR accuracy
        """)

    st.markdown("---")

    # Next step button
    st.markdown('<div class="section-header">✨ Next Step</div>', unsafe_allow_html=True)
    show_info_box(
        "Image restoration complete! Proceed to extract Sanskrit text using AI-powered OCR.",
        icon="✅"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📖 Proceed to OCR Extraction →", use_container_width=True, type="primary"):
            st.switch_page("pages/3_📖_OCR.py")

else:
    # Show original image while waiting
    st.markdown('<div class="section-header">📷 Current Manuscript</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="image-label">Original Manuscript (Not Yet Restored)</div>', unsafe_allow_html=True)
        st.image(st.session_state.uploaded_image, use_container_width=True)

    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #FFF9E6; 
    border-radius: 15px; border-left: 4px solid #F4C430; margin-top: 2rem;">
        <p style="color: #8B4513; font-size: 1.1rem;">
            ⏳ Click the "Restore Manuscript Image" button above to enhance the image quality
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Step 2 of 4: Image Restoration</div>', unsafe_allow_html=True)

