"""
Streamlit Application for Sanskrit Text Extraction using Tesseract OCR

Features:
- Drag and drop or browse to upload images
- Extract Sanskrit text using Tesseract OCR or Google Cloud Vision (Document & Standard modes)
- Display extracted text in the application
"""

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import tempfile
import os
import cv2
import numpy as np
import io

# New helpers for completeness
from ocr.diagnostics import compute_image_quality, summarize_metrics, is_low_quality
from ocr.heuristics import estimate_completeness, needs_fallback
from ocr.tiler import run_tiling_fallback
from ocr.merge import merge_segments

                    # Pre-OCR diagnostics
                    diag_metrics = compute_image_quality(image)
                    st.caption(f"🧪 Image Quality: {summarize_metrics(diag_metrics)}")
                    if is_low_quality(diag_metrics):
                        st.warning("⚠️ Low-quality indicators detected (blur/contrast/brightness). Fallbacks may engage.")

                    extracted_text = ""  # ensure variable exists for later fallback logic

                    # After primary extraction, evaluate completeness
                    completeness_info = estimate_completeness(extracted_text, diag_metrics['megapixels']) if extracted_text else {
                        'completeness_score': 0.0,
                        'char_count': 0,
                        'chars_per_mp': 0.0,
                        'devanagari_ratio': 0.0,
                        'avg_line_length': 0.0
                    }
                    if not extracted_text.strip() or needs_fallback(completeness_info):
                        st.warning("⚠️ Primary extraction may be incomplete. Initiating tiling fallback for COMPLETE TEXT.")
                        tile_engine = 'google' if 'Google' in ocr_engine else 'tesseract'
                        tiles_result = run_tiling_fallback(image, engine=tile_engine, lang=ocr_lang)
                        if tiles_result:
                            merged = merge_segments(tiles_result)
                            if len(merged) > len(extracted_text):
                                st.success(f"🔁 Fallback improved extraction: +{len(merged)-len(extracted_text)} characters (tiling)")
                                extracted_text = merged
                        else:
                            st.info("ℹ️ No additional text recovered from tiling fallback.")

                        # Recompute completeness after tiling
                        completeness_info = estimate_completeness(extracted_text, diag_metrics['megapixels']) if extracted_text else completeness_info

                        # Rotation fallback if still incomplete
                        if needs_fallback(completeness_info):
                            st.warning("🔄 Attempting rotation fallback (90°/180°/270°) for additional text.")
                            rotation_candidates = [90, 180, 270]
                            best_rotated = extracted_text
                            for angle in rotation_candidates:
                                rotated_img = image.rotate(angle, expand=True)
                                if 'Google' in ocr_engine:
                                    try:
                                        rot_text, _ = extract_text_google_vision(rotated_img, use_document_mode=use_document_mode)
                                    except Exception:
                                        rot_text = ''
                                else:
                                    rot_text = pytesseract.image_to_string(rotated_img, lang=ocr_lang, config='--oem 1 --psm 3 -c preserve_interword_spaces=1')
                                # If rotation produced more text, accept
                                if len(rot_text.strip()) > len(best_rotated.strip()):
                                    st.success(f"✨ Rotation {angle}° improved extraction: +{len(rot_text)-len(best_rotated)} chars")
                                    best_rotated = rot_text
                            if len(best_rotated) > len(extracted_text):
                                extracted_text = best_rotated
                                completeness_info = estimate_completeness(extracted_text, diag_metrics['megapixels'])
                            else:
                                st.info("ℹ️ Rotation fallback did not yield additional text.")

                    # Show completeness metrics
                    with st.expander("📊 Extraction Completeness Metrics", expanded=False):
                        st.write({k: v for k, v in completeness_info.items()})
                        if needs_fallback(completeness_info):
                            st.warning("⚠️ Overall completeness score below threshold. Some text may still be missing.")
                        else:
                            st.success("✅ Completeness heuristics passed.")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Try to import Google Cloud Vision
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Sanskrit OCR - Text Extraction",
    page_icon="🕉️",
    layout="wide"
)

# Initialize session state at the start
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'ocr_completed' not in st.session_state:
    st.session_state.ocr_completed = False
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# Custom CSS for better UI
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
    .sub-header {
        font-size: 1.5rem;
        color: #4A5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .image-container {
        text-align: center;
        padding: 20px;
        border: 2px solid #E2E8F0;
        border-radius: 10px;
        background: #F7FAFC;
        margin-bottom: 20px;
    }
    .text-output {
        padding: 20px;
        border: 2px solid #E2E8F0;
        border-radius: 10px;
        background: #FFFFFF;
        font-size: 1.2rem;
        line-height: 2;
        direction: ltr;
        text-align: left;
        min-height: 200px;
        font-family: 'Noto Sans Devanagari', 'Arial Unicode MS', sans-serif;
    }
    .info-box {
        padding: 15px;
        border-left: 4px solid #4299E1;
        background: #EBF8FF;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        padding: 15px;
        border-left: 4px solid #48BB78;
        background: #F0FFF4;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

def preprocess_image(image, contrast=1.5, sharpness=2.0, denoise=True, threshold=True):
    """
    Preprocess image for better OCR text extraction

    Args:
        image: PIL Image object
        contrast: Contrast enhancement factor
        sharpness: Sharpness enhancement factor
        denoise: Apply denoising
        threshold: Apply adaptive thresholding

    Returns:
        Preprocessed PIL Image
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)

    # Convert to OpenCV format for advanced processing
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Apply denoising
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Apply adaptive thresholding for better text extraction
    if threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    # Convert back to PIL Image
    processed_image = Image.fromarray(gray)

    return processed_image

def extract_text_google_vision(image, use_document_mode=True):
    """Extract text using Google Cloud Vision (standard or document mode).

    Args:
        image: PIL Image
        use_document_mode: bool - if True, use document_text_detection (better for full pages)
    Returns:
        Extracted text string (may be empty if none detected)
    """
    if not GOOGLE_VISION_AVAILABLE:
        raise ImportError("Google Cloud Vision not available. Install with: pip install google-cloud-vision")

    # Initialize client (prefer service account / ADC)
    try:
        api_key = os.getenv('GOOGLE_CLOUD_API_KEY') or os.getenv('GOOGLE_VISION_API_KEY')
        if api_key:
            # API key auth (Note: Some endpoints may require service account) - attempt client_options
            from google.cloud.vision_v1 import ImageAnnotatorClient
            from google.api_core.client_options import ClientOptions
            client_options = ClientOptions(api_key=api_key)
            client = ImageAnnotatorClient(client_options=client_options)
        else:
            client = vision.ImageAnnotatorClient()
    except Exception as e:
        raise Exception(f"Failed to initialize Google Cloud Vision client: {e}")

    # Prepare image bytes
    img_bytes_io = io.BytesIO()
    image.save(img_bytes_io, format='PNG')
    content = img_bytes_io.getvalue()
    vision_image = vision.Image(content=content)

    extracted_text = ""
    errors = []

    # Primary attempt: Document mode (better layout, multi-page capability)
    if use_document_mode:
        try:
            response = client.document_text_detection(image=vision_image, image_context={"language_hints": ["sa", "hi", "mr", "en"]})
            if response.error.message:
                errors.append(f"Document mode error: {response.error.message}")
            else:
                if response.full_text_annotation and response.full_text_annotation.text:
                    extracted_text = response.full_text_annotation.text
        except Exception as e:
            errors.append(f"Document detection exception: {e}")

    # Fallback attempt: Standard text detection if document failed or empty
    if not extracted_text.strip():
        try:
            response_std = client.text_detection(image=vision_image, image_context={"language_hints": ["sa", "hi", "mr", "en"]})
            if response_std.error.message:
                errors.append(f"Standard mode error: {response_std.error.message}")
            else:
                texts = response_std.text_annotations
                if texts:
                    extracted_text = texts[0].description
        except Exception as e:
            errors.append(f"Standard detection exception: {e}")

    # Trim trailing spaces but keep line breaks
    return extracted_text, errors

def check_google_vision_credentials():
    """Check if Google Cloud Vision credentials are available"""
    if not GOOGLE_VISION_AVAILABLE:
        return False, "Library not installed"

    # Check for API key
    api_key = os.getenv('GOOGLE_CLOUD_API_KEY') or os.getenv('GOOGLE_VISION_API_KEY')
    if api_key:
        return True, f"API Key (from .env): {api_key[:10]}..."

    # Check for credentials file
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if cred_path and os.path.exists(cred_path):
        return True, f"Credentials file: {cred_path}"

    # Check if gcloud default credentials exist
    try:
        client = vision.ImageAnnotatorClient()
        return True, "Default credentials (gcloud)"
    except:
        return False, "No credentials found"

# Gemini integration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_PROMPT_TEMPLATE = (
    "You are an expert Sanskrit-to-Hindi and Sanskrit-to-English translator.\n"
    "Translate the following Sanskrit text into both Hindi and English.\n"
    "Preserve poetic meaning and avoid literal word-by-word translation.\n"
    "If any verse is incomplete, intelligently reconstruct and translate meaningfully.\n\n"
    "Sanskrit Text:\n---------------------------------------\n"
    "{text}\n"
    "---------------------------------------\n\n"
    "Output format:\n"
    "1. Corrected Sanskrit (if needed)\n"
    "2. Hindi Meaning:\n"
    "3. English Meaning:\n"
)


def translate_with_gemini(text: str) -> str:
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini not available or GEMINI_API_KEY missing")
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    prompt = GEMINI_PROMPT_TEMPLATE.format(text=text)
    resp = model.generate_content(prompt)
    if hasattr(resp, 'text') and resp.text:
        return resp.text.strip()
    if hasattr(resp, 'candidates') and resp.candidates:
        parts = []
        for c in resp.candidates:
            if hasattr(c, 'content') and hasattr(c.content, 'parts'):
                for p in c.content.parts:
                    if hasattr(p, 'text'):
                        parts.append(p.text)
        if parts:
            return "\n".join(parts).strip()
    return ""

# Title
st.markdown('<div class="main-header">🕉️ Sanskrit Text Extraction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Extract COMPLETE TEXT from entire images using Google Cloud Vision or Tesseract OCR</div>', unsafe_allow_html=True)

# Add info banner
st.info("📄 Configured for COMPLETE PAGE text extraction - extracts ALL text from the entire image!")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    ### How to use:
    1. **Upload an image** containing Sanskrit text
    2. **Choose OCR Engine** (Google Vision recommended)
    3. Click **Extract Text** button
    4. View the extracted Sanskrit text
    
    ### Supported formats:
    - PNG, JPG, JPEG
    - Images with Devanagari script
    - Any resolution (higher is better)
    
    ### 🎯 Best Practices:
    - **Use ORIGINAL images** - No preprocessing needed!
    - **Google Cloud Vision**: Best for all image types
    - **Tesseract OCR**: Good alternative, works offline
    - **Preprocessing**: Keep it OFF (disabled by default)
    """)

    st.divider()

    st.header("⚙️ OCR Settings")

    # OCR Engine Selection
    ocr_engine_options = []
    default_index = 0

    # Add Google Vision first if available (makes it default)
    if GOOGLE_VISION_AVAILABLE:
        ocr_engine_options.append("Google Cloud Vision (Google Lens)")
        default_index = 0  # Google Vision is default

    # Add Tesseract
    ocr_engine_options.append("Tesseract OCR")

    ocr_engine = st.selectbox(
        "OCR Engine",
        options=ocr_engine_options,
        index=default_index,
        help="Choose the OCR engine to use for text extraction"
    )

    # Google Vision document mode toggle
    if GOOGLE_VISION_AVAILABLE and "Google" in ocr_engine:
        use_document_mode = st.checkbox(
            "Use Google Document Mode (Full Page)",
            value=True,
            help="Enable advanced Document Text Detection for complete multi-line page extraction"
        )
    else:
        use_document_mode = False

    # Show Google Vision status
    if "Google" in ocr_engine:
        cred_available, cred_info = check_google_vision_credentials()
        if cred_available:
            st.success("✅ Using Google Cloud Vision (Recommended)")
            st.caption("Advanced Google Lens technology for superior accuracy")
            with st.expander("🔑 Credentials Info"):
                st.info(f"Authentication: {cred_info}")
        else:
            st.error("❌ Google Cloud Vision credentials not configured")
            st.warning(f"Issue: {cred_info}")
            with st.expander("How to fix"):
                st.markdown("""
                **Option 1: Use API Key (Easiest)**
                Add to `.env` file:
                ```
                GOOGLE_CLOUD_API_KEY=your_api_key_here
                ```
                
                **Option 2: Use Service Account**
                ```bash
                export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
                ```
                
                **Option 3: Use gcloud**
                ```bash
                gcloud auth application-default login
                ```
                """)
    elif GOOGLE_VISION_AVAILABLE:
        cred_available, cred_info = check_google_vision_credentials()
        if cred_available:
            st.info("💡 Google Cloud Vision is available - switch to it for better accuracy!")
            st.caption(f"Ready: {cred_info}")
        else:
            st.warning("⚠️ Google Cloud Vision installed but not configured")
            st.caption(f"Issue: {cred_info}")
    else:
        st.warning("⚠️ Google Cloud Vision not available")
        with st.expander("How to enable Google Cloud Vision"):
            st.code("pip install google-cloud-vision")
            st.markdown("Also set up credentials:")
            st.code("export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")

    # Language selection (only for Tesseract)
    if ocr_engine == "Tesseract OCR":
        ocr_lang = st.selectbox(
            "OCR Language",
            options=["san", "san+eng", "hin", "hin+eng"],
            index=0,
            help="san: Sanskrit, hin: Hindi, eng: English"
        )

        # OCR Engine Mode
        oem_mode = st.selectbox(
            "OCR Engine Mode",
            options=[
                "1 - Neural nets LSTM only (Best)",
                "3 - Default (Legacy + LSTM)",
                "0 - Legacy engine only",
                "2 - Legacy + LSTM"
            ],
            index=0,
            help="Choose the OCR engine mode"
        )
        oem = int(oem_mode.split(" - ")[0])

        # Page Segmentation Mode
        psm_mode = st.selectbox(
            "Page Segmentation Mode",
            options=[
                "3 - Fully automatic page segmentation (COMPLETE TEXT)",
                "1 - Automatic with OSD (COMPLETE TEXT + orientation)",
                "6 - Single uniform block of text",
                "4 - Single column of text",
                "11 - Sparse text",
                "12 - Sparse text with OSD"
            ],
            index=0,
            help="PSM 3 or 1 extracts ALL text from the entire image"
        )
        psm = int(psm_mode.split(" - ")[0])
    else:
        # Default values for Google Vision (not used but needed for scope)
        ocr_lang = "san"
        oem = 1
        psm = 3

    st.divider()

    st.header("🖼️ Image Preprocessing")
    st.markdown("**Enhance image for Tesseract OCR (optional):**")

    # Preprocessing options - DISABLED by default
    apply_preprocessing = st.checkbox(
        "Enable preprocessing (for Tesseract only)",
        value=False,
        help="Apply image enhancements - NOT recommended, use original image instead"
    )

    st.info("💡 Best practice: Use original images without preprocessing for both OCR engines")

    # Initialize default values
    contrast_factor = 1.5
    sharpness_factor = 2.0
    apply_denoise = True
    apply_threshold = True

    if apply_preprocessing:
        # Contrast enhancement
        contrast_factor = st.slider(
            "Contrast",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Increase contrast for better text visibility"
        )

        # Sharpness enhancement
        sharpness_factor = st.slider(
            "Sharpness",
            min_value=0.5,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="Increase sharpness for clearer text"
        )

        # Denoise
        apply_denoise = st.checkbox(
            "Remove noise",
            value=True,
            help="Remove image noise for cleaner text"
        )

        # Binarization
        apply_threshold = st.checkbox(
            "Apply adaptive threshold",
            value=True,
            help="Convert to black and white for better OCR"
        )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")

    # File uploader
    uploaded_file = st.file_uploader(
        "Drag and drop or browse to upload an image",
        type=["png", "jpg", "jpeg"],
        help="Upload an image containing Sanskrit text"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Image information
        st.markdown(f"""
        <div class="info-box">
        <strong>Image Info:</strong><br>
        📏 Size: {image.size[0]} x {image.size[1]} pixels<br>
        🎨 Mode: {image.mode}<br>
        📝 Format: {image.format}
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📝 Extracted Text")

    if uploaded_file is not None:
        # Extract button
        col_extract1, col_extract2 = st.columns(2)

        with col_extract1:
            if "Google" in ocr_engine:
                extract_button = st.button("🔍 Extract Text (Google Lens)", type="primary", use_container_width=True)
            else:
                extract_button = st.button("🔍 Extract Text (Tesseract)", type="primary", use_container_width=True)

        with col_extract2:
            if ocr_engine == "Tesseract OCR":
                auto_retry = st.button("🔄 Auto-Retry (Multiple Configs)", use_container_width=True)
            else:
                auto_retry = False  # Not applicable for Google Vision

        if extract_button or auto_retry:
            with st.spinner("Extracting text from image..."):
                try:
                    # Load image
                    image = Image.open(uploaded_file)

                    # Choose OCR engine
                    if "Google" in ocr_engine:
                        st.info("🔍 Using Google Cloud Vision (Google Lens) for OCR...")
                        if use_document_mode:
                            st.info("📄 Document Mode Enabled: Attempting full page structured extraction")
                        else:
                            st.info("📝 Standard Text Detection Mode")
                        st.info("📸 Sending ORIGINAL image to Google Cloud Vision (no preprocessing)")

                        try:
                            extracted_text, gv_errors = extract_text_google_vision(image, use_document_mode=use_document_mode)
                            if extracted_text.strip():
                                lines = len(extracted_text.split('\n'))
                                words = len(extracted_text.split())
                                chars = len(extracted_text)
                                st.success(f"✅ Google Cloud Vision extraction successful! Lines={lines}, Words={words}, Characters={chars}")
                                if gv_errors:
                                    with st.expander("⚠️ Warnings / Errors (Non-blocking)"):
                                        for err in gv_errors:
                                            st.warning(err)
                            else:
                                st.warning("⚠️ Google Vision returned no text.")
                                if gv_errors:
                                    with st.expander("🛠 Google Vision Error Details"):
                                        for err in gv_errors:
                                            st.error(err)
                                st.info("💡 Try: Ensure credentials (service account) & enable Vision API. You can also switch to Tesseract OCR or toggle Document Mode.")
                        except Exception as e:
                            st.error(f"❌ Google Vision Fatal Error: {str(e)}")
                            st.info("💡 Falling back to Tesseract OCR (PSM 3)")
                            extracted_text = pytesseract.image_to_string(image, lang='san', config='--oem 1 --psm 3 -c preserve_interword_spaces=1')
                        # Store original image (no preprocessing for Google Vision)
                        st.session_state.processed_image = image

                    elif auto_retry:
                        # Tesseract: Try multiple configurations on ORIGINAL image
                        st.info("🔄 Trying multiple Tesseract OCR configurations for COMPLETE TEXT extraction...")
                        st.info("📸 Using ORIGINAL image (no preprocessing)")

                        # Enhanced configs for complete page text extraction
                        configs_to_try = [
                            ("PSM 3: Full auto page (COMPLETE TEXT)", image, ocr_lang, '--oem 1 --psm 3 -c preserve_interword_spaces=1'),
                            ("PSM 1: Auto with OSD (COMPLETE TEXT)", image, ocr_lang, '--oem 1 --psm 1 -c preserve_interword_spaces=1'),
                            (f"PSM {psm}: User selected", image, ocr_lang, f'--oem {oem} --psm {psm} -c preserve_interword_spaces=1'),
                            ("PSM 6: Single block", image, ocr_lang, '--oem 1 --psm 6 -c preserve_interword_spaces=1'),
                            ("PSM 4: Single column", image, ocr_lang, '--oem 1 --psm 4 -c preserve_interword_spaces=1'),
                        ]

                        best_text = ""
                        best_config = ""
                        all_results = []

                        progress_bar = st.progress(0)
                        for idx, (desc, img, lang, config) in enumerate(configs_to_try):
                            st.caption(f"Trying: {desc}")
                            text = pytesseract.image_to_string(img, lang=lang, config=config)

                            if len(text.strip()) > len(best_text.strip()):
                                best_text = text
                                best_config = desc
                                st.success(f"✨ Better result: {len(text)} chars")

                            progress_bar.progress((idx + 1) / len(configs_to_try))

                        extracted_text = best_text
                        if best_text.strip():
                            st.success(f"🎯 Best configuration: {best_config}")

                        st.session_state.processed_image = image  # Original image

                    else:
                        # Single Tesseract configuration - USE ORIGINAL IMAGE
                        if apply_preprocessing:
                            st.warning("⚠️ Preprocessing enabled - but NOT recommended")
                            st.info("⚙️ Preprocessing image for Tesseract OCR...")
                            processed_image = preprocess_image(
                                image,
                                contrast=contrast_factor,
                                sharpness=sharpness_factor,
                                denoise=apply_denoise,
                                threshold=apply_threshold
                            )
                        else:
                            st.info("📸 Using ORIGINAL image (no preprocessing) - RECOMMENDED")
                            processed_image = image

                        # Enhanced config for complete text extraction
                        custom_config = (
                            f'--oem {oem} --psm {psm} '
                            f'-c preserve_interword_spaces=1 '
                            f'-c page_separator="" '
                        )
                        st.info(f"🔍 Running Tesseract OCR with: Language={ocr_lang}, OEM={oem}, PSM={psm}")
                        st.info("📄 Configured for COMPLETE PAGE text extraction")

                        # Perform OCR - extract all text from entire image
                        extracted_text = pytesseract.image_to_string(
                            processed_image,
                            lang=ocr_lang,
                            config=custom_config
                        )

                        if extracted_text.strip():
                            lines = len(extracted_text.split('\n'))
                            words = len(extracted_text.split())
                            chars = len(extracted_text)
                            st.success(f"✅ Extracted COMPLETE TEXT: {lines} lines, {words} words, {chars} characters")
                        else:
                            st.warning("⚠️ No text detected.")
                            st.info("💡 Try clicking 'Auto-Retry' to test multiple PSM modes")

                        st.session_state.processed_image = processed_image

                    # Store in session state
                    st.session_state.extracted_text = extracted_text
                    st.session_state.ocr_completed = True

                    # Success message
                    if extracted_text.strip():
                        st.markdown("""
                        <div class="success-box">
                        ✅ Text extraction completed successfully!
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error during text extraction: {str(e)}")
                    import traceback
                    with st.expander("🐛 Debug Information"):
                        st.code(traceback.format_exc())
                    st.info("💡 Make sure Tesseract is installed with Sanskrit language data.")
                    st.code("sudo apt-get install tesseract-ocr tesseract-ocr-san")

        # Display extracted text if available
        if hasattr(st.session_state, 'extracted_text') and st.session_state.ocr_completed:
            extracted_text = st.session_state.extracted_text

            # Show image info based on OCR engine used
            if st.session_state.processed_image is not None:
                if "Google" in ocr_engine:
                    with st.expander("📸 Image Sent to Google Cloud Vision", expanded=False):
                        st.image(st.session_state.processed_image, caption="Original Image (No Preprocessing)", use_container_width=True)
                        st.info("✨ Google Cloud Vision works directly on the original image!")
                else:
                    # Check if preprocessing was actually applied
                    if apply_preprocessing:
                        with st.expander("🔍 View Preprocessed Image", expanded=False):
                            st.image(st.session_state.processed_image, caption="Preprocessed Image Used for Tesseract OCR", use_container_width=True)
                            st.warning("⚠️ Preprocessing was applied (not recommended)")
                    else:
                        with st.expander("📸 Image Sent to Tesseract OCR", expanded=False):
                            st.image(st.session_state.processed_image, caption="Original Image (No Preprocessing)", use_container_width=True)
                            st.success("✅ Using original image - RECOMMENDED approach")

            # Display text in a styled container
            if extracted_text.strip():
                st.markdown(f"""
                <div class="text-output">
                {extracted_text}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No text detected in the image")
                if "Google" in ocr_engine:
                    st.markdown("""
                    ### Troubleshooting Google Cloud Vision:
                    1. Check if the image contains clear text
                    2. Verify your API key is valid and has Cloud Vision API enabled
                    3. Try switching to Tesseract OCR as alternative
                    """)
                else:
                    st.markdown("""
                    ### Troubleshooting Tesseract:
                    1. Click **Auto-Retry** to try multiple PSM modes automatically
                    2. Try different **PSM modes** manually (PSM 3, 6, 4, 11)
                    3. Verify the image contains Sanskrit/Devanagari script
                    4. Check if Tesseract has Sanskrit language data: `tesseract --list-langs`
                    5. Switch to **Google Cloud Vision** for better accuracy
                    
                    **Note:** Preprocessing is usually NOT helpful - keep it disabled!
                    """)

            # Text statistics
            if extracted_text.strip():
                word_count = len(extracted_text.split())
                char_count = len(extracted_text)
                line_count = len(extracted_text.split('\n'))

                st.markdown(f"""
                <div class="info-box">
                <strong>Text Statistics:</strong><br>
                📊 Lines: {line_count}<br>
                📝 Words: {word_count}<br>
                🔤 Characters: {char_count}
                </div>
                """, unsafe_allow_html=True)

                # Download button
                st.download_button(
                    label="⬇️ Download Extracted Text",
                    data=extracted_text,
                    file_name="extracted_sanskrit_text.txt",
                    mime="text/plain",
                    help="Download the extracted text as a .txt file"
                )

                # Copy to clipboard button (text area for easy copy)
                with st.expander("📋 Copy Text"):
                    st.text_area(
                        "Copy from here:",
                        value=extracted_text,
                        height=200,
                        help="Select all and copy the text"
                    )
    else:
        st.info("👆 Please upload an image to extract Sanskrit text")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #718096; padding: 20px;">
    <p>Powered by Tesseract OCR & Google Cloud Vision | Sanskrit Text Extraction</p>
    <p style="font-size: 0.8rem;">For best results, use clear, high-resolution images with good contrast</p>
</div>
""", unsafe_allow_html=True)
