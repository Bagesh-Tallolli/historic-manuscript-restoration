"""
Sanskrit Manuscript Restoration & Translation
Multi-Page Professional Application
Home Page
"""
import streamlit as st
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ui_components import apply_custom_theme

# Page configuration
st.set_page_config(
    page_title="Sanskrit Manuscript Restoration",
    page_icon="📜",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply custom theme
apply_custom_theme()

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'translation_result' not in st.session_state:
    st.session_state.translation_result = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="main-title">📜 Sanskrit<br/>Manuscript</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🧭 Navigation")

    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'Home'

    if st.button("📤 Upload Manuscript", use_container_width=True):
        st.switch_page("pages/1_📤_Upload.py")

    if st.button("🔧 Image Restoration", use_container_width=True):
        st.switch_page("pages/2_🔧_Restoration.py")

    if st.button("📖 OCR Extraction", use_container_width=True):
        st.switch_page("pages/3_📖_OCR.py")

    if st.button("🌐 Translation", use_container_width=True):
        st.switch_page("pages/4_🌐_Translation.py")

    if st.button("📚 History", use_container_width=True):
        st.switch_page("pages/5_📚_History.py")

    st.markdown("---")

    # Status indicators
    st.markdown("### 📊 Workflow Status")

    status_upload = "✅" if st.session_state.uploaded_image else "⏳"
    status_restore = "✅" if st.session_state.enhanced_image else "⏳"
    status_ocr = "✅" if st.session_state.extracted_text else "⏳"
    status_translate = "✅" if st.session_state.translation_result else "⏳"

    st.markdown(f"{status_upload} Upload Image")
    st.markdown(f"{status_restore} Restore Image")
    st.markdown(f"{status_ocr} Extract Text")
    st.markdown(f"{status_translate} Translate")

# Main Content
st.markdown('<div class="main-title">Sanskrit Manuscript Restoration & Translation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Digital Preservation of Ancient Indian Manuscripts</div>', unsafe_allow_html=True)

# Hero Section
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2 style="color: #8B4513; font-family: 'Crimson Text', serif;">
            Preserving India's Heritage Through Modern Technology
        </h2>
        <p style="font-size: 1.1rem; color: #5A3E1B; line-height: 1.8;">
            Transform ancient Sanskrit manuscripts into searchable, translatable digital texts
            using advanced AI-powered OCR and image restoration techniques.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# How It Works Section
st.markdown('<div class="section-header">🔄 How It Works</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card" style="text-align: center; height: 220px;">
        <h3 style="color: #D2691E; font-size: 2.5rem;">📤</h3>
        <h4 style="color: #8B4513;">Step 1</h4>
        <p style="color: #5A3E1B;">Upload your manuscript image</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card" style="text-align: center; height: 220px;">
        <h3 style="color: #D2691E; font-size: 2.5rem;">🔧</h3>
        <h4 style="color: #8B4513;">Step 2</h4>
        <p style="color: #5A3E1B;">Enhance image quality with CLAHE</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card" style="text-align: center; height: 220px;">
        <h3 style="color: #D2691E; font-size: 2.5rem;">📖</h3>
        <h4 style="color: #8B4513;">Step 3</h4>
        <p style="color: #5A3E1B;">Extract Sanskrit text using AI OCR</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card" style="text-align: center; height: 220px;">
        <h3 style="color: #D2691E; font-size: 2.5rem;">🌐</h3>
        <h4 style="color: #8B4513;">Step 4</h4>
        <p style="color: #5A3E1B;">Translate to English, Hindi, Kannada</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Features Section
st.markdown('<div class="section-header">✨ Key Features</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
        <strong>🎯 AI-Powered OCR</strong><br>
        Advanced neural network model for accurate Sanskrit text recognition in Devanagari script.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>🖼️ Image Enhancement</strong><br>
        CLAHE and Unsharp Mask techniques for clarity improvement.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <strong>🌍 Multi-Language Translation</strong><br>
        Translate Sanskrit to English, Hindi, and Kannada with scholarly accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>📚 Academic Focus</strong><br>
        Designed for researchers, historians, and heritage institutions.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Call to Action
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #F4C430 0%, #D2691E 100%); 
    padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h3 style="color: #FFFFFF; margin-bottom: 1rem;">Ready to Begin?</h3>
        <p style="color: #FFFFFF;">Start processing your Sanskrit manuscript now</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("📤 Start with Upload →", use_container_width=True, type="primary"):
        st.switch_page("pages/1_📤_Upload.py")

st.markdown("<br>", unsafe_allow_html=True)

# About Section
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### About Sanskrit Manuscript Restoration
    
    This application is designed to help preserve and digitize ancient Sanskrit manuscripts, 
    making them accessible to modern scholars and researchers worldwide.
    
    **Core Technologies:**
    - **Python 3** - Core programming language with strong AI and image processing support
    - **PyTorch** - Deep learning framework for model implementation and training
    - **Vision Transformer (ViT)** - Advanced model for restoring degraded manuscript images
    - **Tesseract OCR** - Extracts Sanskrit text in Devanagari script from restored images
    - **mBART** - Pretrained multilingual neural machine translation model
    - **OpenCV** - Image preprocessing (CLAHE, Unsharp Mask)
    - **Streamlit** - Interactive web interface
    
    **Model Architecture:**
    - **Image Restoration:** Vision Transformer (ViT) model trained on degraded manuscripts
    - **OCR Engine:** Tesseract OCR fine-tuned for Devanagari script recognition
    - **Translation:** mBART multilingual NMT model for Sanskrit-to-English/Hindi/Kannada
    - **Training Environment:** GPU-enabled Kaggle environment for efficient processing
    
    **Processing Pipeline:**
    1. Image enhancement using OpenCV preprocessing
    2. ViT-based restoration of degraded manuscript regions
    3. Tesseract OCR extraction of Devanagari text
    4. mBART neural translation preserving semantic meaning
    
    **Purpose:**
    - Digital preservation of cultural heritage
    - Making Sanskrit texts accessible globally
    - Supporting academic research in Indology
    - Bridging ancient wisdom with modern technology
    
    **Suitable For:**
    - Academic researchers
    - Heritage institutions
    - Digital libraries
    - Sanskrit scholars and students
    """)

# Footer
st.markdown('<div class="footer">Designed for Academic & Cultural Heritage Preservation</div>', unsafe_allow_html=True)

