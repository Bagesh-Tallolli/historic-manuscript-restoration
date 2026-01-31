# 🎨 Sanskrit Manuscript - Multi-Page Streamlit Application

## ✅ REDESIGN COMPLETE

Your Sanskrit Manuscript Restoration & Translation application has been redesigned into a **professional, multi-page Streamlit application** with clean navigation and academic styling.

---

## 📁 Project Structure

```
EL-project/
│
├── Home.py                          # Main landing page
├── pages/                           # Streamlit multipage structure
│   ├── 1_📤_Upload.py              # Step 1: Upload manuscript
│   ├── 2_🔧_Restoration.py         # Step 2: Image enhancement
│   ├── 3_📖_OCR.py                 # Step 3: Text extraction
│   ├── 4_🌐_Translation.py         # Step 4: Multi-language translation
│   └── 5_📚_History.py             # View processing history
│
└── utils/                           # Utility modules
    ├── backend.py                   # All backend logic (UNCHANGED)
    └── ui_components.py             # UI styling functions

```

---

## 🎯 What Changed

### ✅ Frontend UI Only (As Requested)
- **Multi-page structure** instead of single page
- **Sidebar navigation** for easy page switching
- **Step indicators** showing progress (Step 1/4, etc.)
- **Clean sections** with proper spacing
- **Professional academic styling** with saffron-heritage theme

### ❌ Backend Logic (UNCHANGED)
- All processing functions remain exactly the same
- `enhance_manuscript_simple()` - Image restoration
- `perform_ocr_translation()` - OCR and translation
- Gemini AI integration unchanged
- All prompts identical

---

## 🌟 New Features

### 1. **Home Page** (`Home.py`)
- Project title and mission
- "How It Works" visual workflow (4 steps)
- Feature highlights
- Call-to-action button to start
- About section in expander

### 2. **Upload Page** (Step 1 of 4)
- File uploader with format validation
- Image preview in centered layout
- Image metadata display (dimensions, format)
- "Proceed to Restoration" button

### 3. **Restoration Page** (Step 2 of 4)
- Restore button
- Side-by-side comparison (original vs restored)
- Enhancement techniques explanation
- "Proceed to OCR" button

### 4. **OCR Page** (Step 3 of 4)
- Display restored image
- Extract text button
- Side-by-side image and text view
- Copy text functionality
- Formatted Sanskrit text display
- "Proceed to Translation" button

### 5. **Translation Page** (Step 4 of 4)
- Display original Sanskrit text
- Translate button
- Tabbed view for translations
- Complete output display
- Download translation option
- Completion celebration

### 6. **History Page**
- Current session status
- Save session to history
- View all previous sessions
- Usage statistics
- Clear history option

---

## 🎨 Design Improvements

### Sidebar Navigation
- Logo and project name
- Navigation buttons for all pages
- Workflow status indicators (✅/⏳)
- Always visible for easy navigation

### Step Indicators
- Clear step numbers (Step 1/4, etc.)
- Progress visualization
- Contextual guidance

### Section Organization
- Headers with icons
- Info boxes for instructions
- Card-style containers
- Proper spacing and alignment

### Color Scheme
- **Background**: Parchment beige (#FFF8EE)
- **Primary**: Saffron gold (#F4C430)
- **Accent**: Heritage brown (#D2691E)
- **Text**: Dark brown (#5A3E1B)

### Typography
- **Headings**: Crimson Text (serif)
- **Sanskrit**: Noto Serif Devanagari
- **Kannada**: Noto Sans Kannada
- **UI**: System sans-serif

---

## 🚀 How to Run

### Current Status: ✅ **RUNNING**
- **URL**: http://localhost:8501
- **Port**: 8501
- **Process**: Running (PID: 81649)

### Start Application
```bash
cd /home/bagesh/EL-project
source venv/bin/activate
streamlit run Home.py
```

### Stop Application
```bash
pkill -f "streamlit run"
```

---

## 📖 User Workflow

### Complete Processing Flow:

1. **Home Page** → Click "Start with Upload"
   
2. **Upload Page** (Step 1/4)
   - Upload manuscript image (PNG/JPG/JPEG)
   - Preview image
   - Click "Proceed to Image Restoration"

3. **Restoration Page** (Step 2/4)
   - Click "Restore Manuscript Image"
   - View before/after comparison
   - Click "Proceed to OCR Extraction"

4. **OCR Page** (Step 3/4)
   - Click "Extract Sanskrit Text (OCR)"
   - View extracted text
   - Click "Proceed to Translation"

5. **Translation Page** (Step 4/4)
   - Click "Translate to Multiple Languages"
   - View translations (English, Hindi, Kannada)
   - Download results
   - Return to Home or view History

6. **History Page**
   - View current session status
   - Save session to history
   - Browse previous sessions

---

## 🔄 Session State Management

The application maintains workflow state across pages:

```python
st.session_state.uploaded_image       # Original image
st.session_state.enhanced_image       # Restored image
st.session_state.extracted_text       # OCR result
st.session_state.translation_result   # Translations
st.session_state.history              # Processing history
```

---

## 🎯 Key Improvements Over Single-Page Version

| Feature | Old (Single Page) | New (Multi-Page) |
|---------|------------------|------------------|
| **Navigation** | Scrolling | Sidebar menu |
| **Progress** | Unclear | Step indicators (1/4, 2/4, etc.) |
| **Organization** | Cluttered | Clean pages |
| **User Flow** | All-at-once | Step-by-step guidance |
| **Instructions** | Inline | Info boxes |
| **History** | None | Dedicated page |
| **Professional Look** | Basic | Academic styling |
| **Responsiveness** | Limited | Fully responsive |

---

## 📊 Features Comparison

### ✅ Retained (Backend Unchanged)
- Image enhancement (CLAHE + Unsharp Mask)
- Gemini AI OCR extraction
- Multi-language translation
- All processing algorithms
- API integration
- Error handling

### ✨ Added (Frontend Only)
- Multi-page structure
- Sidebar navigation
- Step-by-step workflow
- Progress indicators
- Info boxes and instructions
- History tracking
- Session management
- Download functionality
- Tabbed views
- Better spacing and layout
- Professional styling

---

## 🎨 Custom CSS Styling

The application uses custom CSS for:
- Saffron-heritage color scheme
- Serif fonts for academic look
- Card-style containers
- Button styling
- Image labels
- Text area formatting
- Footer design

All styling is in `utils/ui_components.py` → `apply_custom_theme()`

---

## 🔧 Customization Guide

### Change Colors
Edit `utils/ui_components.py`:
```python
# Global Background
.stApp {
    background-color: #FFF8EE;  # Change this
}

# Primary buttons
.stButton > button {
    background-color: #F4C430;  # Change this
}
```

### Add New Page
1. Create `pages/6_📊_NewPage.py`
2. Add navigation button in sidebar (all pages)
3. Follow same structure as existing pages

### Modify Layout
Each page uses:
- `st.columns()` for side-by-side layouts
- `st.expander()` for collapsible sections
- `st.tabs()` for organized content
- Custom HTML/CSS for styling

---

## 📝 File Descriptions

### `Home.py`
- Main landing page
- Project overview
- "How It Works" section
- Navigation to other pages

### `utils/backend.py`
- All backend functions
- Image enhancement
- OCR and translation
- API integration
- **DO NOT MODIFY**

### `utils/ui_components.py`
- Theme and styling
- Helper functions for UI
- Custom CSS
- Reusable components

### Page Files
Each page follows this structure:
```python
# Imports
# Apply theme
# Initialize session state
# Check prerequisites
# Display content
# Navigation buttons
# Footer
```

---

## 🌐 Accessing the Application

### Local Access
```
http://localhost:8501
```

### Network Access (if needed)
```
http://YOUR_IP_ADDRESS:8501
```

To find your IP:
```bash
hostname -I
```

---

## 📱 Responsive Design

The application is fully responsive:
- Desktop: Full layout with sidebar
- Tablet: Adapted column widths
- Mobile: Single column, collapsible sidebar

---

## 🎓 Academic Presentation Tips

### Demo Flow (5 minutes):
1. **Show Home** - Explain project mission (30s)
2. **Upload** - Drag image, show preview (30s)
3. **Restoration** - Click restore, show comparison (1min)
4. **OCR** - Extract text, show Sanskrit (1.5min)
5. **Translation** - Translate, show 3 languages (1.5min)
6. **History** - Show session tracking (30s)

### Highlight:
- "Multi-page professional architecture"
- "Step-by-step guided workflow"
- "Academic heritage-inspired design"
- "Real-time AI processing"

---

## 🐛 Troubleshooting

### Application won't start:
```bash
# Check if port 8501 is in use
netstat -tlnp | grep 8501

# Kill existing process
pkill -f "streamlit run"

# Restart
streamlit run Home.py
```

### Page navigation not working:
- Ensure all files are in correct locations
- Check file naming (includes emojis and underscores)
- Verify imports in each file

### Styling not applied:
- Check `utils/ui_components.py` exists
- Verify `apply_custom_theme()` is called in each page
- Clear browser cache

---

## 📚 Dependencies

All unchanged from original:
```
streamlit
pillow
numpy
opencv-python
google-genai
```

---

## ✅ Testing Checklist

- [x] Home page loads correctly
- [x] Sidebar navigation works
- [x] File upload accepts images
- [x] Image restoration processes correctly
- [x] OCR extraction works
- [x] Translation generates output
- [x] History saves sessions
- [x] All styling applied
- [x] Step indicators show correctly
- [x] Responsive on different screens

---

## 🎉 Summary

Your application is now:
- ✅ Multi-page with clean navigation
- ✅ Step-by-step guided workflow
- ✅ Professional academic styling
- ✅ Better organized and spaced
- ✅ History tracking
- ✅ **Backend logic completely unchanged**

**Access at: http://localhost:8501**

---

*Designed for Academic & Cultural Heritage Preservation*

