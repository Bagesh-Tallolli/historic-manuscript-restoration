"""
Sanskrit Manuscript Analyzer with Gemini AI
- Upload manuscript image
- Extract and correct Sanskrit text
- Translate to Hindi and English
- Estimate time period
- Display side-by-side comparison
"""

import io
import base64
import streamlit as st
from PIL import Image
import google.generativeai as genai

# Configure Gemini API
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# System prompt for manuscript analysis
MANUSCRIPT_PROMPT = """You are an expert Sanskrit scholar and paleographer specializing in ancient manuscripts.

Analyze this Sanskrit manuscript image and provide:

1. **Extracted Sanskrit Text**: Extract all visible Sanskrit text from the manuscript
2. **Corrected Sanskrit**: Correct any unclear or damaged portions intelligently
3. **Hindi Translation**: Provide a meaningful Hindi translation (not literal word-by-word)
4. **English Translation**: Provide a meaningful English translation preserving poetic/philosophical meaning
5. **Time Period**: Estimate the approximate time period (century/era) based on script style, language features, and manuscript characteristics
6. **Script Type**: Identify the script used (Devanagari, Sharada, Grantha, etc.)
7. **Manuscript Details**: Note any visible details about condition, decoration, or notable features

Format your response as follows:

### EXTRACTED TEXT
[Original extracted text]

### CORRECTED SANSKRIT
[Corrected Sanskrit text]

### HINDI TRANSLATION
[Hindi translation]

### ENGLISH TRANSLATION
[English translation]

### TIME PERIOD
[Estimated period with reasoning]

### SCRIPT TYPE
[Script identification]

### MANUSCRIPT DETAILS
[Additional observations]
"""

# Page configuration
st.set_page_config(
    page_title="Sanskrit Manuscript Analyzer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #8B4513;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #FFF8DC, #FFE4B5);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #8B4513;
        border-bottom: 2px solid #DEB887;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .stButton>button {
        background-color: #8B4513;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #A0522D;
    }
    .info-box {
        background-color: #FFF8DC;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #8B4513;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📜 Sanskrit Manuscript Analyzer 📜</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.info("""
    This application uses Gemini AI to:
    - Extract Sanskrit text from manuscripts
    - Correct unclear portions
    - Translate to Hindi and English
    - Estimate time period
    - Identify script type
    """)

    st.markdown("### Settings")
    model_choice = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro-vision-latest"],
        index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

    st.markdown("### Instructions")
    st.markdown("""
    1. Upload a clear image of the manuscript
    2. Click 'Analyze Manuscript'
    3. View extracted text and translations
    4. Review time period estimation
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h3 class="section-header">📤 Upload Manuscript</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a manuscript image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload a clear image of the Sanskrit manuscript"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Manuscript", use_container_width=True)

        # Image info
        with st.expander("Image Information"):
            st.write(f"**Format:** {image.format}")
            st.write(f"**Size:** {image.size}")
            st.write(f"**Mode:** {image.mode}")

with col2:
    st.markdown('<h3 class="section-header">ℹ️ Analysis Status</h3>', unsafe_allow_html=True)
    status_placeholder = st.empty()

    if uploaded_file is None:
        status_placeholder.info("📸 Please upload a manuscript image to begin analysis")
    else:
        status_placeholder.success("✅ Image uploaded successfully! Click 'Analyze Manuscript' below.")

# Analyze button
if uploaded_file is not None:
    st.markdown("---")
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_button = st.button("🔍 Analyze Manuscript", use_container_width=True)

    if analyze_button:
        with st.spinner("🔄 Analyzing manuscript... This may take a moment..."):
            try:
                # Prepare image for Gemini
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                # Initialize Gemini model
                model = genai.GenerativeModel(model_choice)

                # Prepare the image part
                image_parts = [
                    {
                        "mime_type": "image/png",
                        "data": img_byte_arr.getvalue()
                    }
                ]

                # Generate content
                response = model.generate_content(
                    [MANUSCRIPT_PROMPT, image_parts[0]],
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                    )
                )

                # Parse response
                response_text = response.text

                # Display results
                st.markdown("---")
                st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)

                # Split response into sections
                sections = {}
                current_section = None
                current_content = []

                for line in response_text.split('\n'):
                    if line.startswith('###'):
                        if current_section:
                            sections[current_section] = '\n'.join(current_content).strip()
                        current_section = line.replace('#', '').strip()
                        current_content = []
                    else:
                        current_content.append(line)

                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()

                # Display in organized layout
                st.markdown("### 📜 Original & Corrected Text")
                text_col1, text_col2 = st.columns(2)

                with text_col1:
                    st.markdown("#### Extracted Text")
                    if 'EXTRACTED TEXT' in sections:
                        st.text_area("", sections['EXTRACTED TEXT'], height=200, key="extracted", disabled=True)
                    else:
                        st.info("No extracted text found")

                with text_col2:
                    st.markdown("#### Corrected Sanskrit")
                    if 'CORRECTED SANSKRIT' in sections:
                        st.text_area("", sections['CORRECTED SANSKRIT'], height=200, key="corrected", disabled=True)
                    else:
                        st.info("No corrected text found")

                # Translations side by side
                st.markdown("### 🌐 Translations")
                trans_col1, trans_col2 = st.columns(2)

                with trans_col1:
                    st.markdown("#### 🇮🇳 Hindi Translation")
                    if 'HINDI TRANSLATION' in sections:
                        st.text_area("", sections['HINDI TRANSLATION'], height=250, key="hindi", disabled=True)
                    else:
                        st.info("No Hindi translation found")

                with trans_col2:
                    st.markdown("#### 🇬🇧 English Translation")
                    if 'ENGLISH TRANSLATION' in sections:
                        st.text_area("", sections['ENGLISH TRANSLATION'], height=250, key="english", disabled=True)
                    else:
                        st.info("No English translation found")

                # Additional details
                st.markdown("### 📅 Manuscript Information")
                info_col1, info_col2, info_col3 = st.columns(3)

                with info_col1:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("#### ⏳ Time Period")
                    if 'TIME PERIOD' in sections:
                        st.write(sections['TIME PERIOD'])
                    else:
                        st.info("Not determined")
                    st.markdown('</div>', unsafe_allow_html=True)

                with info_col2:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("#### ✍️ Script Type")
                    if 'SCRIPT TYPE' in sections:
                        st.write(sections['SCRIPT TYPE'])
                    else:
                        st.info("Not identified")
                    st.markdown('</div>', unsafe_allow_html=True)

                with info_col3:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("#### 📝 Details")
                    if 'MANUSCRIPT DETAILS' in sections:
                        st.write(sections['MANUSCRIPT DETAILS'])
                    else:
                        st.info("No additional details")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Download option
                st.markdown("---")
                st.markdown("### 💾 Export Results")

                export_text = f"""SANSKRIT MANUSCRIPT ANALYSIS REPORT
{'='*60}

EXTRACTED TEXT:
{sections.get('EXTRACTED TEXT', 'N/A')}

CORRECTED SANSKRIT:
{sections.get('CORRECTED SANSKRIT', 'N/A')}

HINDI TRANSLATION:
{sections.get('HINDI TRANSLATION', 'N/A')}

ENGLISH TRANSLATION:
{sections.get('ENGLISH TRANSLATION', 'N/A')}

TIME PERIOD:
{sections.get('TIME PERIOD', 'N/A')}

SCRIPT TYPE:
{sections.get('SCRIPT TYPE', 'N/A')}

MANUSCRIPT DETAILS:
{sections.get('MANUSCRIPT DETAILS', 'N/A')}

{'='*60}
Generated by Sanskrit Manuscript Analyzer
"""

                st.download_button(
                    label="📥 Download Analysis Report",
                    data=export_text,
                    file_name="manuscript_analysis.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                st.success("✅ Analysis completed successfully!")

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Sanskrit Manuscript Analyzer | Powered by Google Gemini AI</p>
    <p style='font-size: 0.8rem;'>For educational and research purposes</p>
</div>
""", unsafe_allow_html=True)

