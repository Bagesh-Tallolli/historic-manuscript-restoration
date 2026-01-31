"""
Sanskrit Manuscript OCR and Translation with Gemini AI + Image Restoration
- Upload a manuscript image.
- Optionally restore degraded manuscripts using trained AI model.
- Extract and correct Sanskrit text.
- Translate to Hindi and English.
Dependencies:
pip install google-genai streamlit pillow torch torchvision opencv-python einops
"""
import io
import os
import sys
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from google import genai
from google.genai import types

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import restoration utilities
from models.vit_restorer import create_vit_restorer
from utils.image_restoration_enhanced import create_enhanced_restorer
# --- Configuration ---
st.set_page_config(
    page_title="Sanskrit OCR & Translation",
    page_icon="📜",
    layout="wide",
)

# Hardcoded API Key
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Default model (gemini-3-pro-preview) - used in background, not shown to user
DEFAULT_MODEL = "gemini-3-pro-preview"

# Restoration Model Configuration
RESTORATION_CHECKPOINT_PATHS = [
    "checkpoints/kaggle/final_converted.pth",
    "checkpoints/kaggle/final.pth",
    "models/trained_models/final.pth",
]
RESTORATION_MODEL_SIZE = "base"
RESTORATION_IMG_SIZE = 256
# System prompt for OCR and translation
OCR_PROMPT = """You are an expert Sanskrit-to-Hindi and Sanskrit-to-English translator.
From the provided image, first extract all Sanskrit text present.
Then translate the Sanskrit text into both Hindi and English.
Preserve poetic meaning and avoid literal word-by-word translation.
If any verse is incomplete, intelligently reconstruct and translate meaningfully.
Sanskrit Text:
---------------------------------------
<EXTRACT TEXT FROM IMAGE>
---------------------------------------
Output format:
1. Corrected Sanskrit (if needed)
2. Hindi Meaning:
3. English Meaning:
4. kannada Meaning:
"""

# --- Helper Functions ---

@st.cache_resource
def load_restoration_model():
    """
    Load the pre-trained image restoration model
    Returns: (model, enhanced_restorer, device) or (None, None, None) if loading fails
    """
    try:
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Find checkpoint
        checkpoint_path = None
        for path in RESTORATION_CHECKPOINT_PATHS:
            full_path = os.path.join(PROJECT_ROOT, path)
            if os.path.exists(full_path):
                checkpoint_path = full_path
                break

        if checkpoint_path is None:
            st.warning("⚠️ Restoration model checkpoint not found. Restoration feature will be disabled.")
            return None, None, None

        # Load checkpoint first to determine format
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Determine which format to use based on checkpoint keys
        has_patch_recon = any(k.startswith('patch_recon.') for k in state_dict.keys())
        has_head = any(k.startswith('head.') for k in state_dict.keys())

        # Create model with appropriate format
        if has_patch_recon:
            # New format with patch_recon
            model = create_vit_restorer(
                model_size=RESTORATION_MODEL_SIZE,
                img_size=RESTORATION_IMG_SIZE,
                use_simple_head=False  # Use patch_recon decoder
            )
        else:
            # Old format with simple head
            model = create_vit_restorer(
                model_size=RESTORATION_MODEL_SIZE,
                img_size=RESTORATION_IMG_SIZE,
                use_simple_head=True  # Use simple head decoder
            )

        # Load state dict
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        # Create enhanced restorer
        enhanced_restorer = create_enhanced_restorer(
            model, device=device, patch_size=RESTORATION_IMG_SIZE, overlap=32
        )

        return model, enhanced_restorer, device

    except Exception as e:
        st.error(f"❌ Error loading restoration model: {e}")
        return None, None, None


def restore_image_quality(image_pil, enhanced_restorer):
    """
    Restore degraded manuscript image using trained model

    Args:
        image_pil: PIL Image
        enhanced_restorer: EnhancedRestoration instance

    Returns:
        Restored PIL Image
    """
    try:
        # Convert PIL to numpy array (RGB)
        img_array = np.array(image_pil)

        # If grayscale, convert to RGB
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Determine if we should use patch-based processing
        h, w = img_array.shape[:2]
        use_patches = (h > 512 or w > 512)

        # Restore image
        restored_array = enhanced_restorer.restore_image(
            img_array,
            use_patches=use_patches,
            apply_postprocess=True
        )

        # Convert back to PIL
        restored_pil = Image.fromarray(restored_array)

        return restored_pil

    except Exception as e:
        st.error(f"❌ Error during restoration: {e}")
        return image_pil  # Return original on error

# --- UI Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #4A4A4A;
        text-align: center;
        padding: 1rem;
        font-weight: bold;
    }
    .section-header {
        color: #D46228;
        border-bottom: 2px solid #D3D3D3;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        font-size: 1.5rem;
    }
    .stButton>button {
        background-color: #D46228;
        color: white;
        font-size: 1.1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
# --- Application ---
st.markdown('<div class="main-header">📜 Sanskrit Manuscript OCR & Translation</div>', unsafe_allow_html=True)
# --- Load Restoration Model ---
with st.spinner("🔧 Loading restoration model..."):
    model, enhanced_restorer, device = load_restoration_model()

restoration_available = (model is not None and enhanced_restorer is not None)

# --- Sidebar for controls ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Restoration toggle
    if restoration_available:
        use_restoration = st.checkbox(
            "🔧 Enable Image Restoration",
            value=True,
            help="Restore degraded manuscripts for better visualization (OCR uses original image)"
        )
        if device:
            st.caption(f"Restoration Device: {device.upper()}")
    else:
        use_restoration = False
        st.warning("Restoration model not available")

    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3, 0.1)

    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.info(
        "1. Upload an image of a Sanskrit manuscript.\n"
        "2. (Optional) Enable image restoration to view enhanced version.\n"
        "3. Click 'Analyze & Translate' (uses original image).\n"
        "4. View the extracted text and translations.\n"
        "5. Download restored image if needed."
    )
# --- Main Layout ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 class="section-header">📤 Upload Image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a manuscript image...",
        type=["png", "jpg", "jpeg", "bmp"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Manuscript", use_container_width=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'restored_image' not in st.session_state:
    st.session_state.restored_image = None

# --- Image Restoration (if enabled) ---
if uploaded_file and use_restoration and restoration_available:
    with col2:
        st.markdown('<h3 class="section-header">🔧 Restored Image</h3>', unsafe_allow_html=True)

        if st.session_state.restored_image is None:
            with st.spinner("🔧 Restoring image quality..."):
                restored_image = restore_image_quality(image, enhanced_restorer)
                st.session_state.restored_image = restored_image

        st.image(st.session_state.restored_image, caption="AI-Restored Manuscript", use_container_width=True)

        # Option to download restored image
        buf = io.BytesIO()
        st.session_state.restored_image.save(buf, format='PNG')
        st.download_button(
            label="📥 Download Restored Image",
            data=buf.getvalue(),
            file_name="restored_manuscript.png",
            mime="image/png",
        )

# --- Analysis Trigger ---
if uploaded_file:
    if st.button("🔍 Analyze & Translate", use_container_width=True):
        with st.spinner("🤖 Processing your manuscript..."):
            try:
                # Initialize Gemini client
                client = genai.Client(api_key=GEMINI_API_KEY)

                # Always use original image for OCR (restoration is for display only)
                image_to_analyze = image

                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image_to_analyze.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                # Prepare content with image and prompt
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=OCR_PROMPT),
                            types.Part.from_bytes(
                                data=img_bytes,
                                mime_type=f"image/{image.format.lower() if image.format else 'png'}"
                            ),
                        ],
                    ),
                ]
                # Configure generation with thinking mode
                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="HIGH"
                    ),
                )
                # Generate content with streaming (model name not shown to user)
                result_text = ""
                for chunk in client.models.generate_content_stream(
                    model=DEFAULT_MODEL,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if hasattr(chunk, 'text') and chunk.text:
                        result_text += chunk.text
                st.session_state.analysis_result = result_text

                # Show success message
                if use_restoration and st.session_state.restored_image:
                    st.success("✅ Analysis Complete! (OCR from original image, restoration for display)")
                else:
                    st.success("✅ Analysis Complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.analysis_result = None
# --- Display Results ---
st.markdown("---")
st.markdown('<h3 class="section-header">📊 Analysis Results</h3>', unsafe_allow_html=True)

if st.session_state.analysis_result:
    st.markdown(st.session_state.analysis_result)
    # Add a download button for the results
    st.download_button(
        label="📥 Download Results",
        data=st.session_state.analysis_result,
        file_name="ocr_analysis.txt",
        mime="text/plain",
    )
else:
    st.info("Analysis results will be displayed here after processing.")
