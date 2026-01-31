"""
Sanskrit Manuscript Restoration & Translation
A scholarly web application for digital preservation of ancient Indian manuscripts
Frontend redesigned with formal academic aesthetic
"""
import os
import io
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from google import genai
from google.genai import types

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration (hardcoded for production)
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_MODEL = "gemini-2.5-flash"

# OCR System Prompt - Extract only Sanskrit text
OCR_PROMPT = """You are an expert Sanskrit scholar specializing in manuscript digitization.

From the provided manuscript image, extract all visible Sanskrit text accurately in Devanagari script.
Correct any obvious transcription errors.

Output only the Sanskrit text in Devanagari, nothing else."""

# Translation Prompt - For full translation
TRANSLATION_PROMPT = """You are an expert Sanskrit-to-English, Sanskrit-to-Hindi, and Sanskrit-to-Kannada translator.

Translate the following Sanskrit text into English, Hindi, and Kannada.
Preserve the scholarly and poetic meaning. Avoid literal word-by-word translation.
If verses are incomplete, provide context where possible.

Sanskrit Text:
{sanskrit_text}

Output format (STRICT):

**Extracted Sanskrit Text:**
{sanskrit_text}

**English Meaning:**
[English translation]

**हिंदी अर्थ:**
[Hindi translation in Devanagari]

**ಕನ್ನಡ ಅರ್ಥ:**
[Kannada translation]
"""

# ============================================================================
# CUSTOM THEME & STYLING
# ============================================================================

def apply_custom_theme():
    """Apply Saffron-heritage academic theme"""
    st.markdown("""
        <style>
        /* Import Serif Font */
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600;700&family=Noto+Serif+Devanagari:wght@400;600&family=Noto+Sans+Kannada:wght@400;600&display=swap');
        
        /* Global Background */
        .stApp {
            background-color: #FFF8EE;
        }
        
        /* Main Container */
        .main {
            background-color: #FFF8EE;
        }
        
        /* Title Styling */
        .main-title {
            font-family: 'Crimson Text', serif;
            font-size: 3rem;
            font-weight: 700;
            color: #D2691E;
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            letter-spacing: 1px;
        }
        
        .subtitle {
            font-family: 'Crimson Text', serif;
            font-size: 1.2rem;
            color: #5A3E1B;
            text-align: center;
            margin-bottom: 2rem;
            font-style: italic;
        }
        
        /* Section Headers */
        .section-header {
            font-family: 'Crimson Text', serif;
            font-size: 1.8rem;
            font-weight: 600;
            color: #8B4513;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #F4C430;
            padding-bottom: 0.5rem;
        }
        
        /* Card Containers */
        .card {
            background-color: #FAF0E6;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid #DEB887;
            box-shadow: 0 2px 8px rgba(90, 62, 27, 0.1);
        }
        
        /* Image Containers */
        .image-container {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 1rem;
            border: 2px solid #DEB887;
            margin: 1rem 0;
        }
        
        .image-label {
            font-family: 'Crimson Text', serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #5A3E1B;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        /* Text Areas */
        .sanskrit-text {
            font-family: 'Noto Serif Devanagari', serif;
            font-size: 1.3rem;
            line-height: 2;
            color: #2C1810;
            background-color: #FFFAF0;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #DEB887;
            text-align: justify;
        }
        
        .translation-card {
            background-color: #FFFAF0;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #F4C430;
        }
        
        .translation-title {
            font-family: 'Crimson Text', serif;
            font-size: 1.4rem;
            font-weight: 600;
            color: #8B4513;
            margin-bottom: 0.8rem;
        }
        
        .translation-text {
            font-size: 1.1rem;
            line-height: 1.8;
            color: #2C1810;
            text-align: justify;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #F4C430;
            color: #5A3E1B;
            font-family: 'Crimson Text', serif;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
            border: 2px solid #D2691E;
            padding: 0.7rem 2rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #D2691E;
            color: #FFFFFF;
            border-color: #8B4513;
        }
        
        /* File Uploader */
        .uploadedFile {
            background-color: #FAF0E6;
            border-radius: 8px;
        }
        
        /* Footer */
        .footer {
            font-family: 'Crimson Text', serif;
            font-size: 0.9rem;
            color: #8B7355;
            text-align: center;
            margin-top: 3rem;
            padding: 1rem;
            border-top: 1px solid #DEB887;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# IMAGE ENHANCEMENT FUNCTION
# ============================================================================

def enhance_manuscript_simple(image_pil):
    """
    Enhance manuscript using CLAHE + Unsharp Mask

    Args:
        image_pil: PIL Image

    Returns:
        Enhanced PIL Image
    """
    try:
        # Convert to numpy
        img = np.array(image_pil.convert("RGB"))

        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # Apply unsharp mask for sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Convert back to PIL
        return Image.fromarray(sharpened)
    except Exception as e:
        st.error(f"Enhancement failed: {e}")
        return image_pil

# ============================================================================
# OCR & TRANSLATION FUNCTION
# ============================================================================

def perform_ocr_translation(client, image_pil, prompt_text, model_name, temperature=0.3):
    """
    Perform OCR and translation using Gemini

    Returns:
        Translated text or None if failed
    """
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Prepare content
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_text),
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/png"
                    ),
                ],
            ),
        ]

        # Configure generation (without thinking mode to avoid errors)
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
        )

        # Generate content with streaming
        result_text = ""
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if hasattr(chunk, 'text') and chunk.text:
                result_text += chunk.text

        return result_text.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Apply custom theme
    apply_custom_theme()

    # Initialize session state with proper serialization handling
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'enhanced_image' not in st.session_state:
        st.session_state.enhanced_image = None
    if 'ocr_result' not in st.session_state:
        st.session_state.ocr_result = None
    if 'show_restoration' not in st.session_state:
        st.session_state.show_restoration = False
    if 'show_ocr' not in st.session_state:
        st.session_state.show_ocr = False
    if 'show_translation' not in st.session_state:
        st.session_state.show_translation = False
    if 'current_image_name' not in st.session_state:
        st.session_state.current_image_name = None

    # Header
    st.markdown('<div class="main-title">Sanskrit Manuscript Restoration & Translation</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Digital Preservation of Ancient Indian Manuscripts</div>', unsafe_allow_html=True)

    # ========================================================================
    # SECTION 1: Upload Manuscript Image
    # ========================================================================

    st.markdown('<div class="section-header">📜 Upload Manuscript Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Select a manuscript image to begin",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        try:
            # Check if this is a new image
            if uploaded.name != st.session_state.current_image_name:
                img_bytes = uploaded.read()
                image_original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                st.session_state.uploaded_image = image_original
                st.session_state.current_image_name = uploaded.name

                # Display uploaded image
                st.markdown('<div class="image-label">Original Manuscript</div>', unsafe_allow_html=True)
                st.image(image_original, use_container_width=True)

                # Reset downstream states
                st.session_state.show_restoration = False
                st.session_state.enhanced_image = None
                st.session_state.show_ocr = False
                st.session_state.extracted_text = None
                st.session_state.translation_result = None
                st.session_state.show_translation = False
            else:
                # Same image, just display it
                if st.session_state.uploaded_image:
                    st.markdown('<div class="image-label">Original Manuscript</div>', unsafe_allow_html=True)
                    st.image(st.session_state.uploaded_image, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load image: {e}")
            return

    # ========================================================================
    # SECTION 2: Image Restoration
    # ========================================================================

    if st.session_state.uploaded_image is not None:
        st.markdown('<div class="section-header">🧹 Image Restoration</div>', unsafe_allow_html=True)

        if st.button("🧹 Restore Manuscript Image", use_container_width=True):
            with st.spinner("Restoring manuscript..."):
                enhanced = enhance_manuscript_simple(st.session_state.uploaded_image)
                st.session_state.enhanced_image = enhanced
                st.session_state.show_restoration = True
                # Reset downstream
                st.session_state.show_ocr = False
                st.session_state.ocr_result = None
                st.session_state.show_translation = False

        # Display side-by-side if restored
        if st.session_state.show_restoration and st.session_state.enhanced_image:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="image-label">Original Manuscript</div>', unsafe_allow_html=True)
                st.image(st.session_state.uploaded_image, use_container_width=True)

            with col2:
                st.markdown('<div class="image-label">Restored Manuscript</div>', unsafe_allow_html=True)
                st.image(st.session_state.enhanced_image, use_container_width=True)

    # ========================================================================
    # SECTION 3: OCR Extraction
    # ========================================================================

    if st.session_state.show_restoration and st.session_state.enhanced_image:
        st.markdown('<div class="section-header">🔍 OCR Extraction</div>', unsafe_allow_html=True)

        if st.button("🔍 Extract Sanskrit Text (OCR)", use_container_width=True):
            with st.spinner("Extracting Sanskrit text from manuscript..."):
                try:
                    client = genai.Client(api_key=API_KEY)

                    # Perform OCR on enhanced image - Extract only Sanskrit text
                    result = perform_ocr_translation(
                        client,
                        st.session_state.enhanced_image,
                        OCR_PROMPT,
                        DEFAULT_MODEL,
                        temperature=0.3
                    )

                    st.session_state.extracted_text = result
                    st.session_state.show_ocr = True
                    st.session_state.show_translation = False

                except Exception as e:
                    st.error(f"OCR extraction failed: {e}")

        # Display ONLY extracted Sanskrit text
        if st.session_state.show_ocr and st.session_state.extracted_text:
            st.markdown('<div class="section-header">📖 Extracted Sanskrit Text</div>', unsafe_allow_html=True)

            # Display in a styled container - ONLY Sanskrit text
            st.markdown(f'<div class="sanskrit-text">{st.session_state.extracted_text}</div>', unsafe_allow_html=True)

    # ========================================================================
    # SECTION 4: Translation
    # ========================================================================

    if st.session_state.show_ocr and st.session_state.extracted_text:
        st.markdown('<div class="section-header">🌐 Translation</div>', unsafe_allow_html=True)

        if st.button("🌐 Translate Extracted Text", use_container_width=True):
            with st.spinner("Translating to English, Hindi, and Kannada..."):
                try:
                    client = genai.Client(api_key=API_KEY)

                    # Create translation prompt with extracted Sanskrit text
                    translation_prompt = TRANSLATION_PROMPT.format(
                        sanskrit_text=st.session_state.extracted_text
                    )

                    # Perform translation
                    translation = perform_ocr_translation(
                        client,
                        st.session_state.enhanced_image,
                        translation_prompt,
                        DEFAULT_MODEL,
                        temperature=0.3
                    )

                    st.session_state.translation_result = translation
                    st.session_state.show_translation = True

                except Exception as e:
                    st.error(f"Translation failed: {e}")

        # Display translations if completed
        if st.session_state.show_translation and st.session_state.translation_result:
            # Display the complete result in structured format
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(st.session_state.translation_result)
            st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown('<div class="footer">Designed for Academic & Cultural Heritage Preservation</div>', unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Sanskrit Manuscript Restoration",
        page_icon="📜",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    main()

