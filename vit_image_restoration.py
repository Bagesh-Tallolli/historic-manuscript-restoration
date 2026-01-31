"""
Sanskrit Manuscript Image Restoration - ViT Model
Upload blurry/unclear Sanskrit manuscript → Get clear, polished, readable image
Uses trained Vision Transformer (ViT) model for image-to-image restoration
"""

import io
import os
import sys
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import restoration utilities
from models.vit_restorer import create_vit_restorer
from utils.image_restoration_enhanced import create_enhanced_restorer

# Page configuration
st.set_page_config(
    page_title="Sanskrit Image Restoration - ViT AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-size: 1.3rem;
        border-radius: 10px;
        padding: 1rem 3rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
    }
    .metric-box {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E7D32;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Restoration Model Configuration
RESTORATION_CHECKPOINT_PATHS = [
    "checkpoints/kaggle/final_converted.pth",
    "checkpoints/kaggle/final.pth",
    "models/trained_models/final.pth",
]
RESTORATION_MODEL_SIZE = "base"
RESTORATION_IMG_SIZE = 256

@st.cache_resource
def load_restoration_model():
    """Load the pre-trained ViT restoration model"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Find checkpoint
        checkpoint_path = None
        for path in RESTORATION_CHECKPOINT_PATHS:
            full_path = os.path.join(PROJECT_ROOT, path)
            if os.path.exists(full_path):
                checkpoint_path = full_path
                break

        if checkpoint_path is None:
            st.error("❌ ViT model checkpoint not found!")
            return None, None, None

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Determine format
        has_patch_recon = any(k.startswith('patch_recon.') for k in state_dict.keys())

        # Create model
        model = create_vit_restorer(
            model_size=RESTORATION_MODEL_SIZE,
            img_size=RESTORATION_IMG_SIZE,
            use_simple_head=(not has_patch_recon)
        )

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

def restore_image(image_pil, enhanced_restorer):
    """Restore blurry/unclear manuscript image"""
    try:
        # Convert PIL to numpy array (RGB)
        img_array = np.array(image_pil)

        # Ensure RGB
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        # Determine processing method
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

        return restored_pil, True

    except Exception as e:
        st.error(f"❌ Error during restoration: {e}")
        return image_pil, False

# Initialize session state
if 'restored_image' not in st.session_state:
    st.session_state.restored_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'restoration_success' not in st.session_state:
    st.session_state.restoration_success = False

# Header
st.markdown('<h1 class="main-header">✨ Sanskrit Manuscript Image Restoration</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">🔍 Upload Blurry/Unclear Image → 🎯 Get Clear, Polished, Readable Image</p>', unsafe_allow_html=True)

# Load model
with st.spinner("🔧 Loading ViT restoration model..."):
    model, enhanced_restorer, device = load_restoration_model()

restoration_available = (model is not None and enhanced_restorer is not None)

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Information")

    if restoration_available:
        st.success("✅ ViT Restoration Model Loaded")
        st.info(f"🖥️ Device: {device.upper()}")
        st.info(f"📐 Model: {RESTORATION_MODEL_SIZE.upper()}")
        st.info(f"🔢 Patch Size: {RESTORATION_IMG_SIZE}x{RESTORATION_IMG_SIZE}")
    else:
        st.error("❌ Model not available")

    st.divider()

    st.subheader("🎯 What This Does")
    st.markdown("""
    **Vision Transformer (ViT) Model:**
    - Trained specifically for manuscript restoration
    - Enhances blurry/unclear text
    - Removes noise and artifacts
    - Improves contrast and readability
    - Preserves original text structure
    """)

    st.divider()

    st.subheader("✨ Features")
    st.markdown("""
    ✅ Patch-based processing
    ✅ High-resolution support
    ✅ Trained on Sanskrit manuscripts
    ✅ No API limits
    ✅ Fast local processing
    ✅ Consistent results
    """)

    st.divider()

    st.subheader("📖 Instructions")
    st.markdown("""
    1. Upload manuscript image
    2. Click "Restore Image"
    3. Compare before/after
    4. Download restored image
    """)

# Main content
if not restoration_available:
    st.error("❌ ViT restoration model is not available. Please check model checkpoint files.")
    st.stop()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Original Image (Blurry/Unclear)")

    uploaded_file = st.file_uploader(
        "Upload Sanskrit manuscript image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
        help="Upload a blurry, unclear, or degraded Sanskrit manuscript image"
    )

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.session_state.original_image = original_image

        st.image(original_image, caption="Original (Before Restoration)", use_container_width=True)

        # Image info
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown(f"**📊 Image Info:**")
        st.write(f"• Size: {original_image.size[0]} x {original_image.size[1]} pixels")
        st.write(f"• Format: {original_image.format if original_image.format else 'Unknown'}")
        st.write(f"• Mode: {original_image.mode}")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("✨ Restored Image (Clear/Polished)")

    if st.session_state.restored_image is not None:
        st.image(st.session_state.restored_image, caption="Restored (After ViT Processing)", use_container_width=True)

        # Success indicator
        if st.session_state.restoration_success:
            st.success("✅ Image restored successfully!")

        # Download button
        buf = io.BytesIO()
        st.session_state.restored_image.save(buf, format='PNG')
        st.download_button(
            label="📥 Download Restored Image",
            data=buf.getvalue(),
            file_name="restored_manuscript.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.info("👈 Upload an image and click 'Restore Image' to see the clear result here")

# Restore button
st.markdown("---")

if uploaded_file is not None:
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        if st.button("✨ Restore Image with ViT Model", use_container_width=True, type="primary"):
            with st.spinner("🔄 Restoring your manuscript image... Please wait..."):
                try:
                    restored_image, success = restore_image(original_image, enhanced_restorer)

                    if success:
                        st.session_state.restored_image = restored_image
                        st.session_state.restoration_success = True
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Restoration failed. Please try again.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Comparison section
if st.session_state.original_image is not None and st.session_state.restored_image is not None:
    st.markdown("---")
    st.subheader("📊 Before & After Comparison")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.image(st.session_state.original_image, caption="❌ Before: Blurry/Unclear", use_container_width=True)

    with comp_col2:
        st.image(st.session_state.restored_image, caption="✅ After: Clear/Polished", use_container_width=True)

    # Quality improvement info
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown("**✨ Restoration Applied:**")
    st.write("• Enhanced text clarity and sharpness")
    st.write("• Improved contrast between text and background")
    st.write("• Reduced noise and artifacts")
    st.write("• Preserved original text structure")
    st.write("• Made text more readable")
    st.markdown('</div>', unsafe_allow_html=True)

# Info section
st.markdown("---")

with st.expander("ℹ️ About ViT Image Restoration Model"):
    st.markdown("""
    ### What is ViT (Vision Transformer)?
    
    Vision Transformer is a state-of-the-art deep learning architecture that applies transformer 
    models (originally designed for natural language processing) to computer vision tasks.
    
    ### How It Works for Manuscript Restoration:
    
    1. **Patch-based Processing**: Divides image into small patches
    2. **Transformer Encoding**: Analyzes each patch in context of surrounding patches
    3. **Feature Enhancement**: Learns to enhance degraded features
    4. **Reconstruction**: Reconstructs a clearer version of the image
    5. **Post-processing**: Applies final sharpening and enhancement
    
    ### Trained Specifically For:
    
    - 📜 Sanskrit manuscripts
    - 📝 Historical documents
    - 🖼️ Degraded text images
    - 🔍 Low-contrast images
    - 🌫️ Blurry or unclear text
    
    ### Advantages Over Cloud APIs:
    
    ✅ **No API Limits**: Process unlimited images
    ✅ **Fast**: Runs locally on your hardware
    ✅ **Consistent**: Always produces restoration results
    ✅ **Privacy**: Images never leave your system
    ✅ **Reliable**: Trained model specifically for this task
    ✅ **Free**: No API costs
    
    ### Model Details:
    
    - **Architecture**: Vision Transformer (Base)
    - **Input Size**: 256x256 patches with overlap
    - **Training Data**: Sanskrit manuscript dataset
    - **Purpose**: Image-to-image restoration
    - **Output**: Polished, readable image
    """)

with st.expander("🎓 Tips for Best Results"):
    st.markdown("""
    ### Upload Guidelines:
    
    - ✅ **Good**: Scanned manuscripts, old documents, degraded images
    - ✅ **Good**: Blurry text, low contrast, noisy backgrounds
    - ✅ **Good**: Sanskrit, Devanagari script, historical texts
    - ⚠️ **Avoid**: Extremely low resolution (< 100x100 pixels)
    - ⚠️ **Avoid**: Heavily damaged/torn pages (partial text only)
    
    ### For Best Results:
    
    1. Upload highest resolution available
    2. Ensure text is somewhat visible (even if blurry)
    3. Crop to focus on text area if possible
    4. Try different images to see improvement
    5. Compare before/after carefully
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>✨ <strong>Sanskrit Manuscript Image Restoration with ViT</strong></p>
    <p>Trained AI Model | Local Processing | Reliable Results</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        🔍 Blur/Unclear Input → ✨ ViT Model → 🎯 Clear/Polished Output
    </p>
</div>
""", unsafe_allow_html=True)

