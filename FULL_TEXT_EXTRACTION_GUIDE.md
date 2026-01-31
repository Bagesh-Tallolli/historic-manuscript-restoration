# Full Text Extraction Guide

## How to Extract Complete Text from Images

The updated `app_sanskrit_ocr.py` application now includes advanced features to ensure complete text extraction from images.

## New Features for Complete Text Extraction

### 1. **Image Preprocessing (Enabled by Default)**

The application now includes preprocessing options that significantly improve text extraction:

- **Contrast Enhancement**: Increases visibility of faint text
- **Sharpness Enhancement**: Makes text edges clearer
- **Noise Removal**: Cleans up image artifacts
- **Adaptive Thresholding**: Converts to optimal black and white for OCR

### 2. **Optimized OCR Configuration**

The OCR is configured with:
- `preserve_interword_spaces=1`: Maintains spacing between words
- **LSTM Neural Network** (default): Best accuracy for complete text extraction
- **Automatic Page Segmentation**: Detects all text regions

### 3. **Adjustable Settings in Sidebar**

You can fine-tune extraction by adjusting:

#### OCR Settings:
- **Language**: Choose Sanskrit, Hindi, or combinations
- **OCR Engine Mode**: LSTM (Best) is now the default
- **Page Segmentation Mode**: Try different modes for different layouts

#### Preprocessing Settings:
- **Contrast** (0.5 - 3.0): Default 1.5
- **Sharpness** (0.5 - 3.0): Default 2.0
- **Remove Noise**: Enabled by default
- **Adaptive Threshold**: Enabled by default

## Tips for Complete Text Extraction

### ✅ Best Practices:

1. **Use High-Resolution Images**
   - Minimum 300 DPI
   - Larger images = better accuracy

2. **Enable Preprocessing**
   - Keep preprocessing enabled (default)
   - Adjust contrast and sharpness if needed

3. **Choose Right PSM Mode**
   - **PSM 3**: Best for full pages (default)
   - **PSM 6**: For single blocks of text
   - **PSM 4**: For single columns
   - **PSM 11**: For scattered text

4. **Try Different Settings**
   - If text is missing, try:
     - Increasing contrast (1.8 - 2.5)
     - Increasing sharpness (2.5 - 3.0)
     - Different PSM modes

### 🔧 Troubleshooting Incomplete Text:

#### Problem: Some text is missing
**Solutions:**
1. Enable preprocessing
2. Increase contrast to 2.0-2.5
3. Try PSM 6 (Single block) instead of PSM 3
4. Ensure image is not rotated or skewed

#### Problem: Text is garbled
**Solutions:**
1. Reduce contrast to 1.2-1.5
2. Disable adaptive threshold
3. Try different OCR engine modes

#### Problem: Extra spaces or broken words
**Solutions:**
1. Try PSM 4 (Single column)
2. Ensure preprocessing is enabled
3. Check if image has good resolution

## How to Use the Updated Application

1. **Start the application:**
   ```bash
   cd /home/bagesh/EL-project
   source venv/bin/activate
   streamlit run app_sanskrit_ocr.py
   ```

2. **Upload your image** (drag & drop or browse)

3. **Check preprocessing settings** (sidebar):
   - Default settings work well for most images
   - Adjust if needed based on your image quality

4. **Click "Extract Sanskrit Text"**

5. **View results:**
   - See preprocessed image (if enabled)
   - View extracted text
   - Check text statistics (lines, words, characters)
   - Download or copy the text

6. **If text is incomplete:**
   - Adjust contrast/sharpness sliders
   - Try different PSM modes
   - Click "Extract Sanskrit Text" again

## Advanced Configuration

### For Dense Text (Books, Documents):
- PSM: 3 (Fully automatic)
- Contrast: 1.5
- Sharpness: 2.0
- Enable all preprocessing

### For Single Paragraphs:
- PSM: 6 (Single block)
- Contrast: 1.8
- Sharpness: 2.5
- Enable all preprocessing

### For Low-Quality Images:
- Contrast: 2.0-2.5
- Sharpness: 2.5-3.0
- Enable noise removal
- Enable adaptive threshold
- Try PSM 6 or PSM 4

### For High-Quality Scans:
- Contrast: 1.2
- Sharpness: 1.5
- Disable adaptive threshold
- PSM: 3

## Technical Details

### Preprocessing Pipeline:
1. Convert to RGB
2. Enhance contrast
3. Enhance sharpness
4. Convert to grayscale
5. Apply denoising (if enabled)
6. Apply adaptive thresholding (if enabled)

### OCR Configuration:
```
--oem 1              # LSTM Neural Network
--psm 3              # Fully automatic
-c preserve_interword_spaces=1  # Maintain spacing
```

## Verification

To verify you're getting complete text:

1. **Check Text Statistics**
   - Lines, words, and character counts are shown
   - Compare with expected values

2. **View Preprocessed Image**
   - Expand "View Preprocessed Image"
   - Ensure text is clear and readable
   - If not, adjust preprocessing settings

3. **Compare with Original**
   - Download extracted text
   - Manually verify against original image

## Quick Settings for Common Scenarios

### Scenario 1: Ancient Manuscript (Faded Text)
```
Contrast: 2.5
Sharpness: 3.0
Remove Noise: Yes
Adaptive Threshold: Yes
PSM: 6
```

### Scenario 2: Modern Printed Book
```
Contrast: 1.5
Sharpness: 2.0
Remove Noise: Yes
Adaptive Threshold: Yes
PSM: 3
```

### Scenario 3: Handwritten Notes
```
Contrast: 2.0
Sharpness: 2.5
Remove Noise: Yes
Adaptive Threshold: Yes
PSM: 11
```

### Scenario 4: Digital/Typed Text
```
Contrast: 1.2
Sharpness: 1.5
Remove Noise: No
Adaptive Threshold: No
PSM: 3
```

## Need Help?

If you're still not getting complete text extraction:

1. Check image quality (should be clear and high-res)
2. Try all PSM modes (3, 4, 6, 11)
3. Experiment with preprocessing settings
4. Ensure Tesseract Sanskrit data is installed
5. Check that the image contains Devanagari script

---

**Last Updated**: November 29, 2025  
**Application**: app_sanskrit_ocr.py  
**Purpose**: Complete Sanskrit text extraction from images

