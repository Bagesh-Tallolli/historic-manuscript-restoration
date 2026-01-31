"""
UI Styling and Theme Configuration
"""
import streamlit as st


def apply_custom_theme():
    """Apply Saffron-heritage academic theme"""
    st.markdown("""
        <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600;700&family=Noto+Serif+Devanagari:wght@400;600&family=Noto+Sans+Kannada:wght@400;600&display=swap');
        
        /* Global Background */
        .stApp {
            background-color: #FFF8EE;
        }
        
        .main {
            background-color: #FFF8EE;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #FAF0E6;
            border-right: 2px solid #DEB887;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #5A3E1B;
        }
        
        /* Title Styling */
        .main-title {
            font-family: 'Crimson Text', serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #D2691E;
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            letter-spacing: 1px;
        }
        
        .page-title {
            font-family: 'Crimson Text', serif;
            font-size: 2.2rem;
            font-weight: 700;
            color: #8B4513;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 3px solid #F4C430;
            padding-bottom: 0.5rem;
        }
        
        .subtitle {
            font-family: 'Crimson Text', serif;
            font-size: 1.1rem;
            color: #5A3E1B;
            text-align: center;
            margin-bottom: 2rem;
            font-style: italic;
        }
        
        /* Section Headers */
        .section-header {
            font-family: 'Crimson Text', serif;
            font-size: 1.6rem;
            font-weight: 600;
            color: #8B4513;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #F4C430;
            padding-bottom: 0.5rem;
        }
        
        /* Step Indicator */
        .step-indicator {
            background: linear-gradient(135deg, #F4C430 0%, #D2691E 100%);
            color: #FFFFFF;
            font-family: 'Crimson Text', serif;
            font-size: 1rem;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            display: inline-block;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(90, 62, 27, 0.2);
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
        
        .info-box {
            background-color: #FFF9E6;
            border-left: 4px solid #F4C430;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 5px;
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
            width: 100%;
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
        
        /* Success/Error Messages */
        .stSuccess, .stError, .stWarning, .stInfo {
            border-radius: 8px;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)


def show_header(title, subtitle=None):
    """Display page header"""
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def show_step_indicator(step_num, step_name):
    """Display step indicator"""
    st.markdown(
        f'<div class="step-indicator">📍 Step {step_num}: {step_name}</div>',
        unsafe_allow_html=True
    )


def show_info_box(message, icon="ℹ️"):
    """Display information box"""
    st.markdown(
        f'<div class="info-box">{icon} {message}</div>',
        unsafe_allow_html=True
    )

