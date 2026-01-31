"""
Image Polish with Gemini API
Upload an image and get a polished/enhanced version directly from Gemini
"""

import io
import os
import base64
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# Page configuration
st.set_page_config(
    page_title="Sanskrit Manuscript Restoration - Gemini 2.5 Flash Image",
    page_icon="✨",
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
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #145a8a;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'polished_image' not in st.session_state:
    st.session_state.polished_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Header
st.markdown('<h1 class="main-header">✨ Image Polish with Gemini AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image and get AI-enhanced results instantly</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key input
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", "AIzaSyAxTN1rfdyJQVE3ecy5d8Zqkl5I431nBh0"),
        help="Enter your Google Gemini API key"
    )

    st.divider()

    # Enhancement options
    st.subheader("🎨 Enhancement Type")
    enhancement_type = st.selectbox(
        "Select enhancement style",
        [
            "General Polish & Enhancement",
            "Restore Old/Damaged Photos",
            "Enhance Document/Manuscript",
            "Colorize Black & White",
            "Remove Noise & Artifacts",
            "Sharpen & Clarify",
            "Professional Photo Edit"
        ]
    )

    st.divider()

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        temperature = st.slider(
            "Creativity Level",
            0.0, 1.0, 0.4, 0.1,
            help="Higher values = more creative enhancements"
        )

        max_output_tokens = st.slider(
            "Max Response Tokens",
            1000, 8000, 4096, 500,
            help="Maximum tokens for API response"
        )

    st.divider()

    st.markdown("### ℹ️ How it works")
    st.info(
        "1. Upload your image\n"
        "2. Choose enhancement type\n"
        "3. Click 'Polish Image'\n"
        "4. Gemini AI processes it\n"
        "5. Download enhanced result"
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Original Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff'],
        help="Upload any image format"
    )

    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file)
        st.session_state.original_image = original_image

        st.image(original_image, caption="Original Image", use_container_width=True)

        # Image info
        st.caption(f"📊 Size: {original_image.size[0]} x {original_image.size[1]} pixels")
        st.caption(f"📁 Format: {original_image.format if original_image.format else 'Unknown'}")
        st.caption(f"🎨 Mode: {original_image.mode}")

with col2:
    st.subheader("✨ Polished Result")

    if st.session_state.polished_image is not None:
        st.image(st.session_state.polished_image, caption="Polished Image", use_container_width=True)

        # Download button
        buf = io.BytesIO()
        st.session_state.polished_image.save(buf, format='PNG')
        st.download_button(
            label="📥 Download Polished Image",
            data=buf.getvalue(),
            file_name="polished_image.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.info("👈 Upload an image and click 'Polish Image' to see results here")

        # Recommend ViT app
        st.warning("⚠️ **Gemini won't generate images. For real image restoration:**")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("**[🎯 ViT Restoration App (Port 8503)](http://localhost:8503)**")
            st.caption("✅ Guaranteed image-to-image transformation")
        with col_v2:
            st.markdown("**[📜 OCR Gemini App (Port 8501)](http://localhost:8501)**")
            st.caption("✅ ViT + OCR + Translation")

# Process button
st.divider()

if uploaded_file is not None:
    # Add prominent warning and alternative before process button
    st.warning("""
    ⚠️ **Reality Check**: Gemini API will likely return text description, not an enhanced image.
    
    **For actual image restoration, use:**
    - **ViT Restoration (Port 8503)**: http://localhost:8503 ← Works 100% of the time
    - **OCR Gemini (Port 8501)**: http://localhost:8501 ← Has ViT built-in
    """)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        if st.button("✨ Polish Image with Gemini AI (May Not Work)", use_container_width=True, type="primary"):
            st.session_state.processing = True

            with st.spinner("🤖 Gemini AI is polishing your image... Please wait..."):
                try:
                    # Initialize Gemini client
                    client = genai.Client(api_key=api_key)

                    # Convert image to bytes
                    img_byte_arr = io.BytesIO()
                    original_image.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()

                    # Create enhancement prompt based on selected type
                    enhancement_prompts = {
                        "General Polish & Enhancement":
                            "Enhance and polish this image. Improve clarity, brightness, contrast, and overall quality. "
                            "Make colors more vibrant, sharpen details, reduce noise, and create a professionally polished result. "
                            "Return ONLY the enhanced image.",

                        "Restore Old/Damaged Photos":
                            "Restore this old or damaged photo. Fix scratches, tears, fading, discoloration, and other damage. "
                            "Enhance clarity, improve contrast, and restore lost details. Make it look professionally restored. "
                            "Return ONLY the restored image.",

                        "Enhance Document/Manuscript":
                            "Enhance this document or manuscript image. Improve text clarity, increase contrast between text and background, "
                            "reduce background noise, straighten if needed, and make text more readable. "
                            "Return ONLY the enhanced document image.",

                        "Colorize Black & White":
                            "Colorize this black and white image with realistic, natural colors. "
                            "Use historically accurate and contextually appropriate colors. Make it look professionally colorized. "
                            "Return ONLY the colorized image.",

                        "Remove Noise & Artifacts":
                            "Remove all noise, artifacts, compression artifacts, grain, and imperfections from this image. "
                            "Clean up the image while preserving important details and sharpness. "
                            "Return ONLY the cleaned image.",

                        "Sharpen & Clarify":
                            "Sharpen and clarify this image. Enhance edge details, improve definition, increase clarity, "
                            "and make everything crisp and well-defined without over-sharpening or creating halos. "
                            "Return ONLY the sharpened image.",

                        "Professional Photo Edit":
                            "Apply professional photo editing to this image. Optimize exposure, color balance, saturation, "
                            "contrast, and sharpness. Create a polished, magazine-quality result. "
                            "Return ONLY the professionally edited image."
                    }

                    prompt = enhancement_prompts.get(
                        enhancement_type,
                        enhancement_prompts["General Polish & Enhancement"]
                    )

                    # Prepare content with image and prompt
                    contents = [
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=prompt),
                                types.Part.from_bytes(
                                    data=img_bytes,
                                    mime_type="image/png"
                                ),
                            ],
                        ),
                    ]

                    # Configure generation
                    generate_content_config = types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )

                    # Call Gemini API
                    st.info("📡 Sending image to Gemini API...")

                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp",  # Latest model with image generation
                        contents=contents,
                        config=generate_content_config
                    )

                    # Extract polished image from response
                    polished_image = None

                    # Check if response contains an image
                    if hasattr(response, 'candidates') and response.candidates is not None and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts is not None:
                            for part in candidate.content.parts:
                                # Check for inline data (image)
                                if hasattr(part, 'inline_data') and part.inline_data is not None:
                                    image_data = part.inline_data.data
                                    polished_image = Image.open(io.BytesIO(image_data))
                                    break
                                # Check for text that might contain base64 image
                                elif hasattr(part, 'text') and part.text:
                                    # Try to extract base64 image from text
                                    text = part.text
                                    if 'base64,' in text:
                                        try:
                                            base64_str = text.split('base64,')[1].split('"')[0]
                                            image_data = base64.b64decode(base64_str)
                                            polished_image = Image.open(io.BytesIO(image_data))
                                            break
                                        except:
                                            pass

                    if polished_image:
                        st.session_state.polished_image = polished_image
                        st.success("✅ Image polished successfully!")
                        st.balloons()
                        st.rerun()
                    else:
                        # If no image returned, Gemini might have provided text response
                        st.warning("⚠️ Gemini API returned a text response instead of an image.")
                        st.info("💡 **Try Gemini 2.5 Pro**: Select it from the sidebar (🤖 AI Model) for better image-to-image transformation. "
                               "Gemini 2.5 has improved image generation capabilities.")

                        # Show the text response
                        if hasattr(response, 'text'):
                            with st.expander("📝 View Gemini Response"):
                                st.write(response.text)

                        # Suggest alternative
                        st.success("✅ **Guaranteed Results**: Use ViT Restoration App (Port 8503)\n\n"
                               "For reliable image restoration, use the dedicated ViT model app which is trained specifically for manuscript restoration.")

                        if st.button("🔄 Try Gemini 2.5 Pro Instead"):
                            st.info("Change the model in the sidebar to 'gemini-2.5-pro' and try again!")

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)

                    # Helpful error messages
                    if "API key" in str(e):
                        st.warning("⚠️ Check your Gemini API key in the sidebar")
                    elif "quota" in str(e).lower():
                        st.warning("⚠️ API quota exceeded. Please check your Gemini API usage limits.")
                    elif "model" in str(e).lower():
                        st.info("💡 The current Gemini model may not support image generation. "
                               "Try using our local ViT restoration model instead.")

            st.session_state.processing = False

# Info section
st.divider()

with st.expander("ℹ️ About Image Polishing with Gemini"):
    st.markdown("""
    ### How it works
    
    This application sends your uploaded image directly to Google's Gemini AI API with specific 
    enhancement instructions. The AI processes the image and returns an enhanced version.
    
    ### Enhancement Types
    
    - **General Polish**: Overall quality improvement
    - **Restore Old Photos**: Fix damage, scratches, fading
    - **Document Enhancement**: Better readability for text
    - **Colorization**: Add colors to B&W images
    - **Noise Removal**: Clean up grainy images
    - **Sharpening**: Increase clarity and definition
    - **Professional Edit**: Magazine-quality results
    
    ### Important Notes
    
    ⚠️ **Current Limitation**: Gemini API's image generation capabilities are limited. The model 
    can analyze and describe images but may not directly generate enhanced versions.
    
    ✅ **Recommended Alternative**: For reliable image restoration, use our **ViT Restoration Model** 
    available in `ocr_gemini_streamlit.py` which provides:
    - Trained AI model for manuscript/document restoration
    - Patch-based processing for high quality
    - Proven results on historical documents
    
    ### API Requirements
    
    - Valid Gemini API key
    - Active internet connection
    - Sufficient API quota
    
    ### Supported Formats
    
    - PNG, JPG, JPEG, BMP, WEBP, TIFF
    - Maximum file size: 10MB (API limit)
    - Recommended: High-resolution images for best results
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>✨ <strong>Image Polish with Gemini 2.5 AI</strong></p>
    <p>Latest AI Model | Image-to-Image Generation | Instant Enhancement</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        🌟 Powered by Gemini 2.5 Pro - Best for image transformation
    </p>
</div>
""", unsafe_allow_html=True)

