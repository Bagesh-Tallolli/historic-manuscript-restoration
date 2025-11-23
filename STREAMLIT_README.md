# 🌐 Streamlit Frontend - Quick Reference

## ✅ Installation Complete!

Your Sanskrit Manuscript Pipeline now has a beautiful web interface built with Streamlit!

---

## 🚀 How to Run

### Simple Method (Recommended)
```bash
cd /home/bagesh/EL-project
./run_app.sh
```

### Manual Method
```bash
cd /home/bagesh/EL-project
source venv/bin/activate
streamlit run app.py
```

The app will open automatically at: **http://localhost:8501**

---

## 📁 Files Created

```
/home/bagesh/EL-project/
├── app.py                    # Main Streamlit application
├── run_app.sh               # Launch script
├── STREAMLIT_GUIDE.md       # Complete user guide
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── output/
    └── streamlit/           # Output directory for results
```

---

## ✨ Features

### 🎨 User Interface
- **Modern Design**: Clean, professional interface
- **Responsive**: Works on desktop and tablets
- **Intuitive**: Easy to navigate tabs
- **Interactive**: Real-time processing feedback

### 🔧 Functionality
- **Image Upload**: Drag & drop or browse
- **Real-time Processing**: See results immediately
- **Multiple Outputs**: Images, text, JSON, reports
- **Configuration**: Customize all settings
- **Downloads**: Export results in multiple formats

### 📊 Display Options
- Side-by-side image comparison
- Devanagari text display
- Romanized transliteration
- English translation
- Quality metrics
- Processing statistics

---

## 🎯 Main Tabs

### 1️⃣ Upload & Process
- Upload manuscript images
- Configure processing options
- Start processing with one click
- View upload preview

### 2️⃣ Results
- View processed images
- Read extracted text
- See translations
- Download outputs
- Check metrics

### 3️⃣ About
- Project information
- Technology stack
- Features overview
- Version details

### 4️⃣ Help
- Step-by-step guide
- Configuration help
- Troubleshooting
- FAQs

---

## ⚙️ Configuration Options

### Sidebar Settings

**Restoration Model**
- Enable/disable restoration
- Upload custom models
- Auto-configuration

**OCR Engine**
- Tesseract (fast)
- TrOCR (accurate)
- Ensemble (best)

**Translation Method**
- Google Translate (online)
- IndicTrans (offline)
- Ensemble (comprehensive)

**Advanced Options**
- Save intermediate results
- Show quality metrics
- Display processing stages

---

## 📖 Usage Example

### Quick Processing

1. **Start the app**
   ```bash
   ./run_app.sh
   ```

2. **Upload an image**
   - Click "Browse files"
   - Select your manuscript image
   - See instant preview

3. **Configure settings** (optional)
   - Choose OCR engine
   - Select translation method
   - Enable features you need

4. **Process**
   - Click "🚀 Process Manuscript"
   - Wait 2-10 seconds
   - View results automatically

5. **Download results**
   - Click download buttons
   - Get JSON, text, or full report
   - Save processed images

---

## 🎨 Customization

### Change Theme Colors
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B35"        # Orange
backgroundColor = "#FFFFFF"      # White
secondaryBackgroundColor = "#F0F2F6"  # Light gray
textColor = "#262730"           # Dark gray
```

### Modify Port
```toml
[server]
port = 8501  # Change to your preferred port
```

---

## 🔍 Troubleshooting

### App won't start
```bash
# Reinstall Streamlit
source venv/bin/activate
pip install --force-reinstall streamlit
```

### Port already in use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Module not found errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt
```

---

## 📊 Performance Tips

### For Speed
- Use Tesseract OCR
- Disable restoration
- Skip intermediate saves
- Use smaller images

### For Accuracy
- Use Ensemble OCR
- Enable restoration
- Use Ensemble translation
- Enable all metrics

### Balanced
- Use TrOCR
- Enable restoration (if trained)
- Use Google Translate
- Show key metrics only

---

## 🔒 Privacy

### Data Storage
- Images: Temporary only (deleted after processing)
- Results: Saved to `output/streamlit/` (optional)
- No cloud storage by default

### Internet Usage
- Google Translate: Requires internet
- IndicTrans: Fully offline
- Everything else: Local processing

### Offline Mode
Set translation to "IndicTrans" for 100% offline operation.

---

## 🎓 Integration with Training

### Use Your Trained Models

1. Train a model:
   ```bash
   python train.py --train_dir data/raw --epochs 100
   ```

2. Find the checkpoint:
   ```bash
   ls models/checkpoints/best_model.pth
   ```

3. Upload in Streamlit:
   - Enable restoration in sidebar
   - Click "Upload Model Checkpoint"
   - Select `best_model.pth`

4. Process with restoration enabled!

---

## 📱 Access from Other Devices

### Same Network
```bash
# Find your IP address
hostname -I

# Access from other device
http://YOUR_IP:8501
```

### Configure in config.toml
```toml
[server]
address = "0.0.0.0"  # Allow external connections
```

---

## 🚦 Status Indicators

### Processing States
- ⏳ **Loading**: Initializing pipeline
- 🔄 **Processing**: Running inference
- ✅ **Success**: Processing complete
- ❌ **Error**: Something went wrong

### Quality Indicators
- 🟢 High confidence (>90%)
- 🟡 Medium confidence (70-90%)
- 🔴 Low confidence (<70%)

---

## 📈 Next Steps

### After First Run

1. **Test with Sample**
   - Use provided test image
   - Verify all features work
   - Understand the workflow

2. **Try Your Data**
   - Upload your manuscripts
   - Compare different settings
   - Save best configurations

3. **Train Models** (Optional)
   - Collect dataset
   - Train restoration model
   - Upload and test in app

4. **Share Results**
   - Download outputs
   - Share with colleagues
   - Build your corpus

---

## 🔗 Related Files

- **STREAMLIT_GUIDE.md**: Comprehensive user guide
- **QUICKSTART.md**: General project commands
- **README.md**: Project overview
- **DATASET_REQUIREMENTS.md**: Training data guide

---

## 🎉 You're All Set!

Your Streamlit frontend is ready to use!

**Quick Start:**
```bash
./run_app.sh
```

**Full Guide:**
```bash
cat STREAMLIT_GUIDE.md
```

**Help:**
Open the app and click the "Help" tab

---

**Made with ❤️ for Sanskrit Digital Humanities**

🕉️ Happy Processing! 📜

