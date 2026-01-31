"""
Upload Manuscript Page
Step 1 of 4
"""
import streamlit as st
import sys
import os
from PIL import Image
import io

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ui_components import apply_custom_theme, show_header, show_step_indicator, show_info_box

# Apply custom theme
apply_custom_theme()

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'current_image_name' not in st.session_state:
    st.session_state.current_image_name = None

# Header
show_header("📤 Upload Manuscript", "Upload your Sanskrit manuscript image to begin digitization")
show_step_indicator(1, "Upload Image")

st.markdown("---")

# Instructions
show_info_box(
    "Select a clear, high-resolution image of your Sanskrit manuscript. Supported formats: PNG, JPG, JPEG",
    icon="💡"
)

# File uploader
st.markdown('<div class="section-header">📷 Select Manuscript Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a manuscript image file",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image of your Sanskrit manuscript"
)

if uploaded:
    try:
        # Check if this is a new image
        if uploaded.name != st.session_state.current_image_name:
            img_bytes = uploaded.read()
            image_original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            st.session_state.uploaded_image = image_original
            st.session_state.current_image_name = uploaded.name

            # Reset downstream states
            st.session_state.enhanced_image = None
            st.session_state.extracted_text = None
            st.session_state.translation_result = None

            st.success(f"✅ Successfully uploaded: {uploaded.name}")

        # Display uploaded image
        st.markdown('<div class="section-header">📋 Image Preview</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<div class="image-label">Original Manuscript</div>', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, use_container_width=True)

        # Image information
        width, height = st.session_state.uploaded_image.size

        with st.expander("📊 Image Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Width", f"{width} px")
                st.metric("Height", f"{height} px")
            with col2:
                st.metric("Format", st.session_state.uploaded_image.format or "PIL")
                st.metric("Mode", st.session_state.uploaded_image.mode)

        st.markdown("---")

        # Next step button
        st.markdown('<div class="section-header">✨ Next Step</div>', unsafe_allow_html=True)
        show_info_box(
            "Your image is ready! Proceed to the restoration page to enhance the manuscript quality.",
            icon="✅"
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔧 Proceed to Image Restoration →", use_container_width=True, type="primary"):
                st.switch_page("pages/2_🔧_Restoration.py")

    except Exception as e:
        st.error(f"❌ Failed to load image: {e}")
        st.info("Please try uploading a different image in PNG, JPG, or JPEG format.")

else:
    # Show upload placeholder
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background-color: #FAF0E6; 
    border-radius: 15px; border: 2px dashed #DEB887;">
        <h2 style="color: #8B7355;">📁 No Image Uploaded Yet</h2>
        <p style="color: #5A3E1B; font-size: 1.1rem;">
            Use the file uploader above to select your manuscript image
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Step 1 of 4: Upload Manuscript Image</div>', unsafe_allow_html=True)

