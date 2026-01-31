# 🚀 Quick Start Guide

## Setup (5 minutes)

### Step 1: Install Dependencies
```bash
cd /home/bagesh/EL-project/manuscript-frontend
npm install
```

### Step 2: Start Development Server
```bash
npm run dev
```

### Step 3: Open in Browser
Navigate to: `http://localhost:3000`

---

## File Structure Quick Reference

```
src/
├── pages/              # Each page is a route
│   ├── Home.jsx       # Landing page (/)
│   ├── Upload.jsx     # Upload page (/upload)
│   ├── Restore.jsx    # Restoration (/restore)
│   ├── OCR.jsx        # OCR extraction (/ocr)
│   ├── Translate.jsx  # Translation (/translate)
│   ├── History.jsx    # Archive (/history)
│   └── About.jsx      # About page (/about)
├── components/        # Reusable components
├── contexts/          # Global state
└── styles/            # Tailwind CSS
```

---

## Common Customizations

### Change Colors
Edit `tailwind.config.js`:
```javascript
colors: {
  saffron: {
    400: '#YOUR_COLOR', // Primary button color
  },
}
```

### Add New Page
1. Create `src/pages/NewPage.jsx`
2. Add route in `src/App.jsx`:
```javascript
<Route path="/new" element={<NewPage />} />
```
3. Add nav link in `src/components/Navbar.jsx`

### Modify API Endpoints
Update fetch URLs in:
- `Restore.jsx` (line ~30)
- `OCR.jsx` (line ~25)
- `Translate.jsx` (line ~35)

---

## Backend Integration

### Connect to Python Backend
1. Start your Streamlit/FastAPI backend on port 8501
2. Frontend automatically proxies `/api/*` requests
3. Update `vite.config.js` if using different port

### Example Backend Adapter (FastAPI)
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/api/restore")
async def restore(data: dict):
    image = data['image']
    # Your enhancement logic here
    return {"restoredImage": enhanced_image}
```

---

## Deployment

### Build for Production
```bash
npm run build
# Output: dist/ folder
```

### Deploy to Vercel (Free)
```bash
npm i -g vercel
vercel --prod
```

### Deploy to Netlify
1. Push to GitHub
2. Connect repo to Netlify
3. Build command: `npm run build`
4. Publish directory: `dist`

---

## Troubleshooting

**Port already in use?**
```bash
# Change port in vite.config.js
server: { port: 3001 }
```

**Tailwind styles not working?**
```bash
# Rebuild Tailwind
npm run dev
```

**API calls failing?**
- Check backend is running
- Verify CORS is enabled
- Check browser console for errors

---

## Demo Tips

**Show workflow step-by-step:**
1. Home → Click "Start Processing"
2. Upload → Drag sample manuscript
3. Restore → Click restore, show comparison
4. OCR → Extract text, show Devanagari
5. Translate → Show English/Hindi/Kannada
6. History → Show archived manuscripts

**Highlight features:**
- Multi-page professional design
- Heritage-inspired aesthetics
- Real-time progress tracking
- Responsive layout
- Academic-grade UI

---

## Need Help?

**React**: https://react.dev  
**Tailwind**: https://tailwindcss.com  
**React Router**: https://reactrouter.com

---

✅ **Your professional, production-ready Sanskrit manuscript application is ready!**

