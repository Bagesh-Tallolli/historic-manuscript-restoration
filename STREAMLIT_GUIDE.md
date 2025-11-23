# 🌐 Streamlit Web Interface - User Guide

**Sanskrit Manuscript Pipeline - Interactive Web Application**

---

## 🚀 Quick Start

### Option 1: Using the Launch Script (Recommended)

```bash
cd /home/bagesh/EL-project
./run_app.sh
```

The app will automatically:
- Activate the virtual environment
- Install Streamlit if needed
- Start the web server
- Open in your default browser at http://localhost:8501

### Option 2: Manual Launch

```bash
cd /home/bagesh/EL-project
source venv/bin/activate
streamlit run app.py
```

---

## 📱 Application Features

### 1. **Upload & Process Tab**

Upload manuscript images and process them through the AI pipeline.

**Features:**
- Drag & drop image upload
- Support for JPEG, PNG, TIFF, BMP
- Real-time processing status
- Example image demonstration
- Instant visual feedback

**How to Use:**
1. Click "Browse files" or drag image into upload area
2. Preview your uploaded image
3. Click "🚀 Process Manuscript" button
4. Wait for processing to complete (2-10 seconds)
5. View results in the Results tab

---

### 2. **Results Tab**

View comprehensive processing results with multiple output formats.

**Displayed Information:**

#### 🖼️ Images
- **Original**: Your uploaded image
- **Restored**: AI-enhanced version (if restoration enabled)
- **Comparison**: Side-by-side comparison

#### 📝 Text Outputs
- **Sanskrit (Devanagari)**: Extracted text in original script
- **Romanized (IAST)**: Transliterated version
- **English Translation**: Translated text

#### 📊 Metrics
- Word count
- Sentence count
- OCR confidence score
- Image quality metrics (PSNR, SSIM)

#### 💾 Download Options
- JSON format (structured data)
- Text format (human-readable)
- Full report with images
- Individual images

---

### 3. **Configuration Sidebar**

Customize processing options to suit your needs.

#### ⚙️ Restoration Model
- **Enable/Disable**: Toggle image restoration
- **Upload Model**: Provide custom trained model (.pth file)
- **Auto-skip**: Works without model (OCR only)

#### 🔍 OCR Engine Selection
- **Tesseract**: Fast, reliable for clear text
  - Best for: Well-preserved manuscripts
  - Speed: ⚡⚡⚡ Very Fast
  - Accuracy: ⭐⭐⭐ Good

- **TrOCR**: Transformer-based, better for degraded text
  - Best for: Damaged or faded manuscripts
  - Speed: ⚡⚡ Moderate
  - Accuracy: ⭐⭐⭐⭐ Excellent

- **Ensemble**: Combines both engines
  - Best for: Maximum accuracy
  - Speed: ⚡ Slower
  - Accuracy: ⭐⭐⭐⭐⭐ Best

#### 🌐 Translation Method
- **Google Translate**: Fast, requires internet
  - Speed: ⚡⚡⚡ Very Fast
  - Quality: ⭐⭐⭐⭐ Very Good
  - Requires: Internet connection

- **IndicTrans**: Local processing, offline
  - Speed: ⚡⚡ Moderate
  - Quality: ⭐⭐⭐⭐ Very Good
  - Requires: Nothing (fully offline)

- **Ensemble**: Uses multiple methods
  - Speed: ⚡ Slower
  - Quality: ⭐⭐⭐⭐⭐ Best
  - Requires: Internet (recommended)

#### 🔧 Advanced Settings
- **Save Intermediate Results**: Keep processing stages
- **Show Quality Metrics**: Display PSNR, SSIM, etc.
- **Show Processing Stages**: Visualize each step

---

## 📖 Detailed Usage Guide

### Processing Your First Manuscript

#### Step 1: Prepare Your Image
- **Format**: JPEG, PNG, TIFF, or BMP
- **Resolution**: 512×512 pixels or higher recommended
- **Content**: Sanskrit text in Devanagari script
- **Quality**: Clear, readable text (can be degraded)

#### Step 2: Configure Settings
1. Open the sidebar (if collapsed)
2. Choose OCR engine based on image quality
3. Select translation method
4. Enable restoration if you have a model
5. Adjust advanced settings if needed

#### Step 3: Upload & Process
1. Go to "Upload & Process" tab
2. Click "Browse files" or drag image
3. Preview appears automatically
4. Click "🚀 Process Manuscript"
5. Watch progress indicator

#### Step 4: Review Results
1. Navigate to "Results" tab
2. Review extracted text
3. Check translation accuracy
4. Examine quality metrics
5. Download results if needed

---

## 🎯 Use Cases

### Use Case 1: Quick OCR
**Goal**: Extract text from a clear manuscript

**Settings:**
- Restoration: Disabled
- OCR Engine: Tesseract
- Translation: Google

**Time**: ~2-3 seconds

### Use Case 2: Damaged Manuscript Restoration
**Goal**: Restore and extract text from damaged manuscript

**Settings:**
- Restoration: Enabled (with model)
- OCR Engine: Ensemble
- Translation: Ensemble

**Time**: ~8-10 seconds

### Use Case 3: Offline Processing
**Goal**: Process without internet connection

**Settings:**
- Restoration: Optional
- OCR Engine: Tesseract or TrOCR
- Translation: IndicTrans

**Time**: ~5-7 seconds

### Use Case 4: Research & Documentation
**Goal**: Generate comprehensive report

**Settings:**
- Restoration: Enabled
- OCR Engine: Ensemble
- Save Intermediate: Yes
- Show Metrics: Yes

**Time**: ~10-15 seconds

---

## 🔧 Advanced Features

### Batch Processing
While the web UI processes one image at a time, you can:
- Process multiple images sequentially
- Download results for each
- Compare results across images

### Custom Models
Upload your own trained restoration models:
1. Train model using `train.py`
2. Find checkpoint in `models/checkpoints/`
3. Upload .pth file in sidebar
4. Enable restoration

### API Integration
The app uses the same pipeline as the CLI:
```python
from main import ManuscriptPipeline

pipeline = ManuscriptPipeline(
    restoration_model_path='path/to/model.pth',
    ocr_engine='ensemble',
    translation_method='ensemble'
)

result = pipeline.process('image.jpg')
```

---

## 📊 Understanding Metrics

### Image Quality Metrics

#### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 20-50 dB
- **Good**: >30 dB
- **Excellent**: >35 dB
- **Meaning**: Higher = better restoration quality

#### SSIM (Structural Similarity Index)
- **Range**: 0-1
- **Good**: >0.8
- **Excellent**: >0.9
- **Meaning**: Higher = better structure preservation

### OCR Metrics

#### Confidence Score
- **Range**: 0-100%
- **Good**: >80%
- **Excellent**: >90%
- **Meaning**: OCR's confidence in extracted text

#### Word Count
- Number of words detected
- Useful for tracking completeness

#### Sentence Count
- Number of sentences identified
- Helps validate text structure

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "App won't start"
**Solutions:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall Streamlit
pip install --force-reinstall streamlit

# Check port availability
netstat -tuln | grep 8501
```

#### Issue: "Image upload fails"
**Solutions:**
- Check file format (must be JPEG, PNG, TIFF, BMP)
- Verify file size (<10MB recommended)
- Try converting image to JPEG
- Check file permissions

#### Issue: "Processing hangs"
**Solutions:**
- Reduce image size
- Disable restoration
- Use single OCR engine (not ensemble)
- Check system resources (RAM, CPU)

#### Issue: "No text detected"
**Solutions:**
- Verify image contains visible text
- Try different OCR engine
- Increase image resolution
- Adjust image brightness/contrast

#### Issue: "Translation failed"
**Solutions:**
- Check internet connection (for Google)
- Switch to IndicTrans (offline)
- Verify extracted text is Sanskrit
- Try ensemble method

#### Issue: "Model upload fails"
**Solutions:**
- Verify .pth file is valid
- Check file size (<500MB)
- Ensure model architecture matches
- Try re-downloading model

---

## 💡 Tips & Best Practices

### For Best Results

1. **Image Quality**
   - Use highest resolution available
   - Ensure good lighting and contrast
   - Avoid shadows and glare
   - Keep text clearly visible

2. **OCR Selection**
   - Clear text → Tesseract (fastest)
   - Degraded text → TrOCR (accurate)
   - Critical work → Ensemble (best)

3. **Translation**
   - Quick work → Google (fast)
   - Offline → IndicTrans (local)
   - Research → Ensemble (thorough)

4. **Performance**
   - Smaller images process faster
   - Disable features you don't need
   - Use single engines for speed
   - Enable all for accuracy

### Workflow Optimization

**Fast Processing:**
```
Tesseract + Google + No Restoration
→ 2-3 seconds per image
```

**Balanced:**
```
TrOCR + IndicTrans + Restoration
→ 5-7 seconds per image
```

**Maximum Quality:**
```
Ensemble + Ensemble + Restoration + All Metrics
→ 10-15 seconds per image
```

---

## 🔒 Privacy & Security

### Data Handling
- All processing happens locally (except Google Translate)
- Images are not stored permanently
- Temporary files are cleaned up automatically
- No data is sent to external servers (except for Google Translate)

### Offline Mode
- Use IndicTrans for translation
- Use Tesseract or TrOCR for OCR
- Disable all cloud features
- Complete privacy guaranteed

---

## 🚀 Performance Optimization

### For Faster Processing

1. **Reduce Image Size**
   ```python
   # Recommended max size: 1024×1024
   # Resize before upload if larger
   ```

2. **Disable Unused Features**
   - Turn off restoration if not needed
   - Use single OCR engine
   - Skip intermediate results

3. **Hardware Acceleration**
   - Use GPU if available (automatic)
   - Close other applications
   - Allocate more RAM

### For Better Accuracy

1. **Use Ensemble Methods**
   - Combines multiple engines
   - Better error correction
   - More robust results

2. **Enable All Metrics**
   - Monitor quality scores
   - Validate results
   - Fine-tune settings

3. **Use Restoration**
   - Improves degraded images
   - Better OCR accuracy
   - Clearer text

---

## 📞 Support & Resources

### Documentation
- **README.md**: Project overview
- **QUICKSTART.md**: Quick commands
- **DATASET_REQUIREMENTS.md**: Training data guide
- **RUN_REPORT.md**: System status

### Getting Help
1. Check troubleshooting section
2. Review documentation files
3. Check GitHub issues
4. Contact support

### Useful Commands

```bash
# Start app
./run_app.sh

# Stop app
Ctrl+C

# Check logs
streamlit logs

# Clear cache
streamlit cache clear

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

## 🎓 Training Integration

### Using Custom Models

After training a model with `train.py`:

1. **Find Model**
   ```bash
   ls models/checkpoints/
   # Look for best_model.pth
   ```

2. **Upload in App**
   - Enable restoration in sidebar
   - Click "Upload Model Checkpoint"
   - Select your .pth file

3. **Process**
   - Upload manuscript image
   - Model will be used automatically
   - Results include restoration metrics

---

## 🎨 Customization

### Modify UI Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

### Add Custom Features

Edit `app.py`:
```python
# Add new tabs
tab5 = st.tabs(["My Custom Feature"])

# Add custom processing
def my_custom_function():
    # Your code here
    pass
```

---

## 📈 Analytics

### Processing Statistics

The app tracks:
- Total images processed
- Average processing time
- Success/failure rates
- Most used features

View in session state or logs.

---

## 🌟 Future Features

Coming soon:
- [ ] Batch upload and processing
- [ ] Real-time video processing
- [ ] Cloud storage integration
- [ ] API endpoints
- [ ] Mobile responsive design
- [ ] Multi-language interface
- [ ] Advanced analytics dashboard

---

## ✅ Checklist for First Use

- [ ] Virtual environment activated
- [ ] Streamlit installed
- [ ] App launches successfully
- [ ] Can upload images
- [ ] Processing works
- [ ] Results display correctly
- [ ] Downloads work
- [ ] Understand configuration options

---

## 📝 Feedback

Help us improve! Report:
- Bugs and issues
- Feature requests
- UI/UX suggestions
- Performance problems

---

**Happy Processing! 🕉️📜**

*For more information, see README.md and other documentation files.*

