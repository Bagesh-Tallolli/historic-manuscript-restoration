"""
History Page
View previously processed manuscripts
"""
import streamlit as st
import sys
import os
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ui_components import apply_custom_theme, show_header, show_info_box

# Apply custom theme
apply_custom_theme()

# Initialize history in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
show_header("📚 Processing History", "View previously processed Sanskrit manuscripts")

st.markdown("---")

# Current session status
st.markdown('<div class="section-header">📊 Current Session</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    status_upload = "✅ Complete" if st.session_state.get('uploaded_image') else "⏳ Pending"
    st.metric("Upload", status_upload)

with col2:
    status_restore = "✅ Complete" if st.session_state.get('enhanced_image') else "⏳ Pending"
    st.metric("Restoration", status_restore)

with col3:
    status_ocr = "✅ Complete" if st.session_state.get('extracted_text') else "⏳ Pending"
    st.metric("OCR", status_ocr)

with col4:
    status_translate = "✅ Complete" if st.session_state.get('translation_result') else "⏳ Pending"
    st.metric("Translation", status_translate)

st.markdown("---")

# Current manuscript details
if st.session_state.get('uploaded_image'):
    st.markdown('<div class="section-header">📄 Current Manuscript</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="image-label">Current Image</div>', unsafe_allow_html=True)

        if st.session_state.get('enhanced_image'):
            st.image(st.session_state.enhanced_image, use_container_width=True)
        else:
            st.image(st.session_state.uploaded_image, use_container_width=True)

    with col2:
        st.markdown("### Processing Details")

        # Image info
        if st.session_state.get('current_image_name'):
            st.text(f"📁 File: {st.session_state.current_image_name}")

        st.text(f"📅 Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Status checklist
        st.markdown("**Status Checklist:**")
        if st.session_state.get('uploaded_image'):
            st.success("✅ Image uploaded")
        if st.session_state.get('enhanced_image'):
            st.success("✅ Image restored")
        if st.session_state.get('extracted_text'):
            st.success("✅ Text extracted")
        if st.session_state.get('translation_result'):
            st.success("✅ Translation completed")

        # Quick actions
        st.markdown("---")
        st.markdown("**Quick Actions:**")

        if st.session_state.get('extracted_text'):
            with st.expander("📖 View Extracted Text"):
                st.text_area(
                    "Sanskrit Text",
                    value=st.session_state.extracted_text,
                    height=200,
                    label_visibility="collapsed"
                )

        if st.session_state.get('translation_result'):
            with st.expander("🌐 View Translations"):
                st.markdown(st.session_state.translation_result)

    # Save to history
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💾 Save Current Session to History", use_container_width=True, type="primary"):
            # Add current session to history
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'filename': st.session_state.get('current_image_name', 'Unknown'),
                'has_ocr': bool(st.session_state.get('extracted_text')),
                'has_translation': bool(st.session_state.get('translation_result')),
                'text': st.session_state.get('extracted_text', ''),
                'translation': st.session_state.get('translation_result', '')
            }
            st.session_state.history.append(history_entry)
            st.success("✅ Session saved to history!")
            st.balloons()

else:
    show_info_box(
        "No manuscript currently loaded. Upload a new manuscript to begin processing.",
        icon="📤"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📤 Go to Upload Page", use_container_width=True):
            st.switch_page("pages/1_📤_Upload.py")

st.markdown("---")

# Historical sessions
st.markdown('<div class="section-header">🗂️ Previous Sessions</div>', unsafe_allow_html=True)

if st.session_state.history:
    st.info(f"📊 Total saved sessions: {len(st.session_state.history)}")

    # Display each history entry
    for idx, entry in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"📄 Session {len(st.session_state.history) - idx + 1}: {entry['filename']}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.text(f"📅 Date: {datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                st.text(f"📁 File: {entry['filename']}")

            with col2:
                st.text(f"OCR: {'✅ Yes' if entry['has_ocr'] else '❌ No'}")
                st.text(f"Translation: {'✅ Yes' if entry['has_translation'] else '❌ No'}")

            if entry['text']:
                st.markdown("**Extracted Text:**")
                st.text_area(
                    "Text",
                    value=entry['text'],
                    height=150,
                    key=f"text_{idx}",
                    label_visibility="collapsed"
                )

            if entry['translation']:
                st.markdown("**Translation:**")
                with st.expander("View Translation"):
                    st.markdown(entry['translation'])

    # Clear history
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.success("✅ History cleared!")
            st.rerun()

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background-color: #FAF0E6; 
    border-radius: 15px; border: 2px dashed #DEB887;">
        <h3 style="color: #8B7355;">📁 No Historical Sessions Yet</h3>
        <p style="color: #5A3E1B; font-size: 1.1rem;">
            Process and save manuscripts to see them here
        </p>
    </div>
    """, unsafe_allow_html=True)

# Usage statistics
st.markdown("---")
st.markdown('<div class="section-header">📈 Usage Statistics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Sessions", len(st.session_state.history))

with col2:
    ocr_count = sum(1 for entry in st.session_state.history if entry['has_ocr'])
    st.metric("OCR Extractions", ocr_count)

with col3:
    trans_count = sum(1 for entry in st.session_state.history if entry['has_translation'])
    st.metric("Translations", trans_count)

# Footer
st.markdown('<div class="footer">Processing History & Session Management</div>', unsafe_allow_html=True)

