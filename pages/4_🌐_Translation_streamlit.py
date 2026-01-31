"""
Translation Page
Step 4 of 4
"""
import streamlit as st
import sys
import os
from google import genai

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ui_components import apply_custom_theme, show_header, show_step_indicator, show_info_box
from utils.backend import perform_ocr_translation, API_KEY, DEFAULT_MODEL, TRANSLATION_PROMPT

# Apply custom theme
apply_custom_theme()

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'translation_result' not in st.session_state:
    st.session_state.translation_result = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None

# Header
show_header("🌐 Translation", "Translate Sanskrit text to English, Hindi, and Kannada")
show_step_indicator(4, "Translate Text")

st.markdown("---")

# Check if text is extracted
if st.session_state.extracted_text is None:
    st.warning("⚠️ No extracted text available!")
    show_info_box(
        "Please complete the OCR extraction step first before proceeding to translation.",
        icon="📖"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Go to OCR Page", use_container_width=True):
            st.switch_page("pages/3_📖_OCR.py")

    st.stop()

# Instructions
show_info_box(
    "Click the button below to translate the extracted Sanskrit text into English, Hindi, and Kannada using AI-powered translation.",
    icon="💡"
)

# Display extracted text
st.markdown('<div class="section-header">📜 Original Sanskrit Text</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sanskrit-text">{st.session_state.extracted_text}</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# Translation button
st.markdown('<div class="section-header">🤖 AI-Powered Translation</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🌐 Translate to Multiple Languages", use_container_width=True, type="primary"):
        with st.spinner("🔄 Translating to English, Hindi, and Kannada... Please wait..."):
            try:
                client = genai.Client(api_key=API_KEY)

                # Create translation prompt
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

                st.success("✅ Translation completed successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Translation failed: {e}")
                st.info("Please try again or check your API configuration.")

st.markdown("---")

# Display translations if completed
if st.session_state.translation_result:
    st.markdown('<div class="section-header">📚 Translations</div>', unsafe_allow_html=True)

    # Use tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📝 All Translations", "🇬🇧 English", "🇮🇳 Hindi", "🇮🇳 Kannada"])

    with tab1:
        st.markdown("### Complete Translation Output")
        st.markdown(
            f'<div class="card">{st.session_state.translation_result}</div>',
            unsafe_allow_html=True
        )

        # Download button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Complete Translation",
                data=st.session_state.translation_result,
                file_name="sanskrit_translation.txt",
                mime="text/plain",
                use_container_width=True
            )

    with tab2:
        st.markdown("### English Translation")
        st.markdown("""
        <div class="translation-card">
            <p style="font-size: 1.1rem; line-height: 1.8; color: #2C1810;">
                View the complete output in the "All Translations" tab for the English translation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### हिंदी अनुवाद")
        st.markdown("""
        <div class="translation-card">
            <p style="font-family: 'Noto Serif Devanagari', serif; font-size: 1.1rem; line-height: 1.8; color: #2C1810;">
                संपूर्ण हिंदी अनुवाद के लिए "सभी अनुवाद" टैब देखें।
            </p>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### ಕನ್ನಡ ಅನುವಾದ")
        st.markdown("""
        <div class="translation-card">
            <p style="font-family: 'Noto Sans Kannada', sans-serif; font-size: 1.1rem; line-height: 1.8; color: #2C1810;">
                ಸಂಪೂರ್ಣ ಕನ್ನಡ ಅನುವಾದಕ್ಕಾಗಿ "ಎಲ್ಲಾ ಅನುವಾದಗಳು" ಟ್ಯಾಬ್ ನೋಡಿ.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Translation details
    with st.expander("🔬 Translation Technology Details"):
        st.markdown("""
        **AI Model:** Gemini 2.5 Flash
        
        **Languages Supported:**
        - English (scholarly translation)
        - Hindi (Devanagari script)
        - Kannada (Kannada script)
        
        **Translation Approach:**
        - Contextual meaning preservation
        - Scholarly interpretation
        - Cultural nuance consideration
        - Avoids literal word-by-word translation
        
        **Accuracy:** Optimized for Sanskrit classical texts and manuscripts
        """)

    st.markdown("---")

    # Completion message
    st.markdown('<div class="section-header">✅ Workflow Complete!</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #F4C430 0%, #D2691E 100%); 
    border-radius: 15px; margin: 1rem 0;">
        <h3 style="color: #FFFFFF; margin-bottom: 1rem;">🎉 Manuscript Processing Complete!</h3>
        <p style="color: #FFFFFF; font-size: 1.1rem;">
            Your Sanskrit manuscript has been successfully digitized, extracted, and translated.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🏠 Return to Home", use_container_width=True):
            st.switch_page("Home.py")

    with col2:
        if st.button("📚 View History", use_container_width=True):
            st.switch_page("pages/5_📚_History.py")

else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background-color: #FFF9E6; 
    border-radius: 15px; border-left: 4px solid #F4C430; margin-top: 2rem;">
        <p style="color: #8B4513; font-size: 1.1rem;">
            ⏳ Click the "Translate to Multiple Languages" button above to generate translations
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Step 4 of 4: Multi-Language Translation</div>', unsafe_allow_html=True)

