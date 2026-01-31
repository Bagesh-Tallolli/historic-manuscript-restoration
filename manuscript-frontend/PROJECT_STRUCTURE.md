# 🏛️ Sanskrit Manuscript Restoration - Complete Project Structure

```
EL-project/
│
├── 📱 STREAMLIT APP (Currently Running ✅)
│   ├── streamlit_app.py              # Main Streamlit application (Port 8501)
│   ├── venv/                         # Python virtual environment
│   │   ├── lib/python3.12/
│   │   │   └── site-packages/
│   │   │       ├── streamlit/
│   │   │       ├── google-genai/
│   │   │       ├── opencv-python/
│   │   │       └── pillow/
│   │   └── bin/
│   │       ├── streamlit
│   │       └── python3
│   └── requirements.txt
│
├── 🌐 REACT FRONTEND (New Professional UI)
│   └── manuscript-frontend/
│       │
│       ├── 📄 Configuration Files
│       │   ├── package.json           # Dependencies & scripts
│       │   ├── vite.config.js        # Build configuration
│       │   ├── tailwind.config.js    # Custom theme (saffron/heritage colors)
│       │   ├── postcss.config.js     # CSS processing
│       │   ├── .npmrc                # npm configuration for WSL
│       │   ├── .gitignore            # Git ignore rules
│       │   ├── Dockerfile            # Docker deployment
│       │   └── index.html            # HTML template
│       │
│       ├── 📚 Documentation
│       │   ├── README.md             # Complete project documentation
│       │   ├── QUICKSTART.md         # Quick setup guide
│       │   ├── DEPLOYMENT_GUIDE.md   # Deployment & troubleshooting
│       │   └── setup.sh              # Automated setup script
│       │
│       └── src/                      # Source code
│           │
│           ├── 📄 Entry Points
│           │   ├── main.jsx          # React entry point
│           │   └── App.jsx           # Main routing component
│           │
│           ├── 📑 Pages (7 pages)
│           │   ├── Home.jsx          # Landing page
│           │   │   ├── Hero section
│           │   │   ├── How it works (4 steps)
│           │   │   ├── Feature highlights
│           │   │   └── Call-to-action
│           │   │
│           │   ├── Upload.jsx        # Step 1: Upload manuscript
│           │   │   ├── Drag-drop upload
│           │   │   ├── File validation
│           │   │   ├── Image preview
│           │   │   └── Metadata form
│           │   │
│           │   ├── Restore.jsx       # Step 2: Image restoration
│           │   │   ├── Restore button
│           │   │   ├── Side-by-side comparison
│           │   │   ├── Zoom controls
│           │   │   └── Download options
│           │   │
│           │   ├── OCR.jsx           # Step 3: Text extraction
│           │   │   ├── Extract button
│           │   │   ├── Sanskrit text display
│           │   │   ├── Edit mode
│           │   │   └── Copy functionality
│           │   │
│           │   ├── Translate.jsx     # Step 4: Translation
│           │   │   ├── Language selection
│           │   │   ├── English translation
│           │   │   ├── Hindi translation
│           │   │   ├── Kannada translation
│           │   │   └── Download all
│           │   │
│           │   ├── History.jsx       # Archive page
│           │   │   ├── Manuscript list
│           │   │   ├── Thumbnail grid
│           │   │   └── Download/delete actions
│           │   │
│           │   └── About.jsx         # About project
│           │       ├── Mission statement
│           │       ├── Academic impact
│           │       ├── Technology stack
│           │       └── Team roles
│           │
│           ├── 🧩 Components (3 components)
│           │   ├── Navbar.jsx        # Navigation bar
│           │   │   ├── Logo
│           │   │   ├── Nav links
│           │   │   └── Active state
│           │   │
│           │   ├── Footer.jsx        # Page footer
│           │   │   ├── About section
│           │   │   ├── Quick links
│           │   │   └── Social links
│           │   │
│           │   └── StepProgressBar.jsx  # Workflow progress
│           │       ├── 4-step indicator
│           │       ├── Checkmarks
│           │       └── Connector lines
│           │
│           ├── 🔄 Context (State Management)
│           │   └── ManuscriptContext.jsx
│           │       ├── uploadedImage
│           │       ├── restoredImage
│           │       ├── extractedText
│           │       ├── translations
│           │       ├── metadata
│           │       └── history
│           │
│           └── 🎨 Styles
│               └── index.css         # Tailwind + custom CSS
│                   ├── Google Fonts import
│                   ├── Tailwind directives
│                   ├── Custom components (.btn-primary, .card, etc.)
│                   └── Animations
│
├── 📊 Data & Assets
│   ├── sample_manuscripts/           # Test images
│   ├── outputs/                      # Generated results
│   └── logs/                         # Application logs
│
└── 📋 Project Documentation
    ├── AGENT_SETUP_GUIDE.md
    ├── COMPLETE_PROJECT_GUIDE.md
    ├── FINAL_PIPELINE_DOCUMENTATION.md
    └── API_KEYS_AND_LIBRARIES_GUIDE.md
```

---

## 🎯 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────┐         ┌───────────────────┐       │
│  │   STREAMLIT UI    │         │   REACT FRONTEND  │       │
│  │  (Port 8501)      │         │   (Port 3000)     │       │
│  │  ✅ Running       │         │   ⏳ To Deploy    │       │
│  └─────────┬─────────┘         └─────────┬─────────┘       │
│            │                               │                 │
└────────────┼───────────────────────────────┼─────────────────┘
             │                               │
             └───────────┬───────────────────┘
                         ▼
            ┌────────────────────────┐
            │   APPLICATION LOGIC    │
            ├────────────────────────┤
            │                        │
            │  📸 Image Enhancement  │
            │     (OpenCV CLAHE)     │
            │                        │
            │  🔍 OCR Extraction     │
            │     (Gemini AI)        │
            │                        │
            │  🌐 Translation        │
            │     (Gemini AI)        │
            │                        │
            └────────────┬───────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   EXTERNAL SERVICES    │
            ├────────────────────────┤
            │                        │
            │  🤖 Google Gemini API  │
            │     (OCR & NLP)        │
            │                        │
            └────────────────────────┘
```

---

## 🔄 User Workflow Comparison

### STREAMLIT (Single Page):
```
Upload Image
     ↓
[Restore Button]
     ↓
Original ↔ Restored (side-by-side)
     ↓
[Extract OCR Button]
     ↓
Sanskrit Text Display
     ↓
[Translate Button]
     ↓
English + Hindi + Kannada
```

### REACT (Multi-Page):
```
Home (/)
  ↓
Upload (/upload)  ← Step 1/4
  ↓
Restore (/restore)  ← Step 2/4
  ↓
OCR (/ocr)  ← Step 3/4
  ↓
Translate (/translate)  ← Step 4/4
  ↓
History (/history)  ← Archive
  ↓
About (/about)  ← Documentation
```

---

## 📦 Component Tree (React)

```
<App>
├── <Router>
│   ├── <ManuscriptProvider>  [Context]
│   │   ├── <Navbar>
│   │   ├── <Routes>
│   │   │   ├── <Home>
│   │   │   │   ├── Hero section
│   │   │   │   ├── Workflow cards (4)
│   │   │   │   └── Feature cards (4)
│   │   │   │
│   │   │   ├── <Upload>
│   │   │   │   ├── <StepProgressBar>
│   │   │   │   ├── <Dropzone>
│   │   │   │   └── Metadata form
│   │   │   │
│   │   │   ├── <Restore>
│   │   │   │   ├── <StepProgressBar>
│   │   │   │   ├── Original image
│   │   │   │   └── Restored image
│   │   │   │
│   │   │   ├── <OCR>
│   │   │   │   ├── <StepProgressBar>
│   │   │   │   └── Sanskrit text area
│   │   │   │
│   │   │   ├── <Translate>
│   │   │   │   ├── <StepProgressBar>
│   │   │   │   └── Translation cards (3)
│   │   │   │
│   │   │   ├── <History>
│   │   │   │   └── Manuscript cards
│   │   │   │
│   │   │   └── <About>
│   │   │       └── Info sections
│   │   │
│   │   ├── <Footer>
│   │   └── <ToastContainer>
│   │
│   └── [Global State]
│       ├── currentStep
│       ├── uploadedImage
│       ├── restoredImage
│       ├── extractedText
│       ├── translations
│       └── history[]
```

---

## 🎨 Design System

### Color Variables (Tailwind)

```javascript
parchment: {
  50: '#FFFBF5',   // Lightest background
  100: '#FFF8EE',  // Main background
  300: '#F5E6D3',  // Borders
}

saffron: {
  400: '#F4C430',  // Primary buttons
  500: '#D2691E',  // Hover states
  600: '#A0522D',  // Active states
}

heritage: {
  400: '#8B7355',  // Secondary text
  600: '#5A3E1B',  // Headings
  700: '#2C1810',  // Body text
}
```

### Typography

- **Headings**: Crimson Text (serif)
- **Body**: Inter (sans-serif)
- **Sanskrit**: Noto Serif Devanagari
- **Kannada**: Noto Sans Kannada

---

## 🚀 Deployment Targets

| Platform | Type | URL Format | Cost |
|----------|------|------------|------|
| **Streamlit Cloud** | Backend | yourapp.streamlit.app | Free |
| **Vercel** | React Frontend | yourapp.vercel.app | Free |
| **Netlify** | React Frontend | yourapp.netlify.app | Free |
| **GitHub Pages** | Static Site | username.github.io/repo | Free |
| **Railway** | Full Stack | yourapp.railway.app | Free tier |
| **Render** | Full Stack | yourapp.render.com | Free tier |

---

## ✅ Completion Checklist

### Backend (Streamlit):
- ✅ Image upload
- ✅ CLAHE enhancement
- ✅ Gemini OCR
- ✅ Multi-language translation
- ✅ Heritage theme
- ✅ Running on port 8501

### Frontend (React):
- ✅ 7 pages created
- ✅ 3 components created
- ✅ Context state management
- ✅ Tailwind theme configured
- ✅ Routing setup
- ✅ Documentation complete
- ⏳ npm install (pending)
- ⏳ Deployment (pending)

---

## 📞 Support Commands

### Check if Streamlit is running:
```bash
ps aux | grep streamlit
netstat -tlnp | grep 8501
```

### Restart Streamlit:
```bash
pkill streamlit
cd /home/bagesh/EL-project
source venv/bin/activate
streamlit run streamlit_app.py
```

### Install React dependencies (WSL fix):
```bash
# Method 1: Pure WSL Node
sudo apt install nodejs npm
cd /home/bagesh/EL-project/manuscript-frontend
npm install

# Method 2: With .npmrc
cd /home/bagesh/EL-project/manuscript-frontend
npm install

# Method 3: Docker
cd /home/bagesh/EL-project/manuscript-frontend
docker build -t manuscript-frontend .
docker run -p 3000:3000 manuscript-frontend
```

---

**🎉 Your complete Sanskrit manuscript preservation system is ready!**

**Next Steps:**
1. Fix npm installation using one of the methods above
2. Run `npm run dev` to start React frontend
3. Test complete workflow with sample manuscript
4. Deploy to cloud for academic presentation

*Documentation created with ❤️ for academic excellence*

