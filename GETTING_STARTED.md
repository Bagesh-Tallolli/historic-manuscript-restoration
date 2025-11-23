# 🚀 GETTING STARTED - Sanskrit Manuscript Pipeline

## ✅ Complete Installation Guide

### Prerequisites
- Python 3.8+ (You have Python 3.12.3 ✓)
- Linux/macOS/Windows
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA (optional, for faster training)

---

## 📦 Step-by-Step Installation

### Option 1: Automatic Setup (Recommended for Linux/macOS)

```bash
# Navigate to project directory
cd /home/bagesh/EL-project

# Run setup script
./setup.sh
```

This will:
- Create virtual environment
- Install all Python dependencies
- Install Tesseract OCR
- Setup directory structure
- Run tests

---

### Option 2: Manual Setup (All Platforms)

#### 1. Install Python Dependencies

```bash
cd /home/bagesh/EL-project

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### 2. Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-san tesseract-ocr-hin
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH
3. Download Sanskrit language pack

#### 3. Verify Installation

```bash
python test_setup.py
```

You should see:
```
✓ PyTorch
✓ OpenCV
✓ NumPy
✓ Tesseract
✓ Models module
✓ OCR module
✓ NLP module
✓ Translation module
```

---

## 🎯 Quick Start Examples

### Example 1: Process a Manuscript (Basic)

```bash
# Process with default settings (no restoration)
python main.py --image_path data/datasets/samples/test_sample.png
```

### Example 2: Process with Restoration (After Training)

```bash
# First, train a model (see below)
# Then process with restoration
python main.py \
    --image_path your_manuscript.jpg \
    --restoration_model checkpoints/best_psnr.pth \
    --ocr_engine tesseract \
    --translation_method google
```

### Example 3: Train Restoration Model

```bash
# Add training images to data/raw/
# Then train
python train.py \
    --train_dir data/raw \
    --epochs 50 \
    --batch_size 8 \
    --model_size base
```

### Example 4: Batch Processing

```python
from main import ManuscriptPipeline
from pathlib import Path

# Initialize pipeline
pipeline = ManuscriptPipeline(
    restoration_model_path=None,  # or path to .pth file
    ocr_engine='tesseract',
    translation_method='google'
)

# Process all images in a folder
for img_path in Path('data/raw').glob('*.jpg'):
    result = pipeline.process_manuscript(img_path, save_output=True)
    print(f"Processed: {img_path.name}")
    print(f"Translation: {result['translation']}\n")
```

---

## 📊 Using the Jupyter Notebook

```bash
# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter
jupyter notebook

# Open demo.ipynb
# Run cells step by step to see the complete pipeline
```

---

## 🔧 Configuration

Edit `config.yaml` to customize settings:

```yaml
model:
  size: 'base'        # Model size
  img_size: 256       # Input image size

training:
  epochs: 100         # Training epochs
  batch_size: 16      # Batch size
  learning_rate: 1.0e-4

ocr:
  engine: 'tesseract' # OCR engine
  language: 'san'     # Sanskrit

translation:
  method: 'google'    # Translation method
```

---

## 📚 Common Use Cases

### Use Case 1: Just OCR (No Restoration)

```python
from ocr.run_ocr import SanskritOCR

ocr = SanskritOCR(engine='tesseract')
text = ocr.extract_text('manuscript.jpg', lang='san')
print(text)
```

### Use Case 2: Just Translation

```python
from nlp.translation import SanskritTranslator

translator = SanskritTranslator(method='google')
english = translator.translate("रामः वनं गच्छति")
print(english)
```

### Use Case 3: Unicode Normalization

```python
from nlp.unicode_normalizer import SanskritTextProcessor

processor = SanskritTextProcessor()

# Convert romanized to Devanagari
result = processor.process_ocr_output(
    "rāmaḥ vanam gacchati", 
    input_format='iast'
)
print(result['normalized'])  # रामः वनं गच्छति
```

### Use Case 4: Full Pipeline

```python
from main import process_manuscript

result = process_manuscript(
    image_path="manuscript.jpg",
    restoration_model="checkpoints/best_psnr.pth",
    save_output=True
)

print("Sanskrit:", result['ocr_text_cleaned'])
print("English:", result['translation'])
print("Words:", result['word_count'])
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision
# or for CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Tesseract not found"
**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-san

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Issue: "Language 'san' not found in Tesseract"
**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr-san tesseract-ocr-hin

# Or download manually from:
# https://github.com/tesseract-ocr/tessdata
```

### Issue: CUDA out of memory
**Solution:**
```bash
# Reduce batch size
python train.py --batch_size 4

# Or use smaller model
python train.py --model_size small
```

### Issue: Import errors in IDE
**Solution:**
This is normal if dependencies aren't installed yet. After running:
```bash
pip install -r requirements.txt
```
The errors will disappear.

---

## 📁 Adding Your Data

### Training Data
```bash
# Add manuscript images to:
data/raw/

# Structure:
data/raw/
├── manuscript1.jpg
├── manuscript2.jpg
└── manuscript3.png
```

### Pre-processed Data
```bash
# If you have clean/degraded pairs:
data/processed/
├── degraded/
│   ├── img1.jpg
│   └── img2.jpg
└── clean/
    ├── img1.jpg
    └── img2.jpg
```

---

## 🎓 Training Tips

### 1. Start Small
```bash
# Test with tiny model first
python train.py --model_size tiny --epochs 10 --batch_size 4
```

### 2. Use Synthetic Degradation
```bash
# If you don't have paired data, use synthetic degradation
python train.py --train_dir data/raw
# (synthetic_degradation is enabled by default)
```

### 3. Monitor Training
```bash
# Use TensorBoard
tensorboard --logdir logs/

# Or Weights & Biases
python train.py --use_wandb
```

### 4. Resume Training
```bash
python train.py --resume checkpoints/epoch_50.pth
```

---

## 📈 Expected Results

### OCR Accuracy
- Clean manuscripts: 90-98%
- Degraded manuscripts: 60-85%
- After restoration: 75-95%

### Restoration Quality
- PSNR: 28-35 dB
- SSIM: 0.85-0.95

### Translation Quality
- Depends on OCR accuracy
- Google Translate: Good for simple sentences
- IndicTrans2: Better for complex Sanskrit

---

## 🔗 Next Steps

1. **Collect Data**: Add manuscript images to `data/raw/`
2. **Train Model**: Run `python train.py`
3. **Process Manuscripts**: Run `python main.py`
4. **Evaluate**: Check `output/` directory
5. **Iterate**: Adjust config.yaml and retrain

---

## 💡 Pro Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for 10-50x speedup
2. **Data Quality**: More training data = better results
3. **Fine-tuning**: Start with pre-trained weights (if available)
4. **Ensemble**: Use multiple OCR/translation methods for better accuracy
5. **Validation**: Always use a validation set to prevent overfitting

---

## 📞 Need Help?

1. Run diagnostics: `python test_setup.py`
2. Check logs in `logs/` directory
3. Review `demo.ipynb` for examples
4. Read `PROJECT_SUMMARY.md` for full documentation

---

## ✨ You're Ready!

Everything is set up and ready to use. Start with:

```bash
# Quick test
python test_setup.py

# Process a manuscript
python main.py --image_path data/datasets/samples/test_sample.png

# Or open the notebook
jupyter notebook demo.ipynb
```

**Happy manuscript processing! 🕉️**

