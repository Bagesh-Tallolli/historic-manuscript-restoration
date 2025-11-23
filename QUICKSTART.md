# 🚀 Quick Start Guide - Sanskrit Manuscript Pipeline

## ✅ Everything is Ready!

Your Sanskrit Manuscript Pipeline is fully installed and operational.

---

## 🎯 Quick Commands

### Activate Virtual Environment
```bash
cd /home/bagesh/EL-project
source venv/bin/activate
```

### Process a Manuscript
```bash
python main.py --image_path data/datasets/samples/test_sample.png
```

### Test the Setup
```bash
python test_setup.py
```

---

## 📊 What Just Happened

✅ **Installed:**
- Virtual environment with Python 3.12.3
- 200+ packages including PyTorch, transformers, OpenCV
- Tesseract OCR with Sanskrit support

✅ **Tested:**
- All modules working correctly
- OCR extracting Devanagari text
- Translation pipeline operational
- Sample image processed successfully

✅ **Generated:**
- Test results in `output/` directory
- JSON and text format outputs
- Visual comparisons of original/restored images

---

## 🎓 Your First Manuscript Processing

### 1. Place Your Manuscript Image
```bash
# Copy your manuscript to:
cp your_manuscript.jpg /home/bagesh/EL-project/data/raw/
```

### 2. Run the Pipeline
```bash
cd /home/bagesh/EL-project
source venv/bin/activate
python main.py --image_path data/raw/your_manuscript.jpg
```

### 3. Check Results
```bash
# Results will be in:
ls -lh output/
cat output/your_manuscript_results.txt
```

---

## 🔧 Advanced Usage

### Use Different OCR Engine
```bash
# TrOCR (transformer-based)
python main.py --image_path data/raw/manuscript.jpg --ocr_engine trocr

# Both engines (ensemble)
python main.py --image_path data/raw/manuscript.jpg --ocr_engine ensemble
```

### Use Different Translation
```bash
# IndicTrans (local model)
python main.py --image_path data/raw/manuscript.jpg --translation indictrans

# Multiple methods
python main.py --image_path data/raw/manuscript.jpg --translation ensemble
```

### With Trained Restoration Model
```bash
python main.py \
    --image_path data/raw/manuscript.jpg \
    --restoration_model models/checkpoints/best_model.pth
```

---

## 📚 Interactive Exploration

### Start Jupyter Notebook
```bash
cd /home/bagesh/EL-project
source venv/bin/activate
jupyter notebook demo.ipynb
```

This will open an interactive demo in your browser.

---

## 🎯 Training Your Own Model

### Download Datasets
```bash
python dataset_downloader.py
```

### Start Training
```bash
python train.py \
    --train_dir data/raw \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4
```

### Monitor with TensorBoard
```bash
# In a new terminal
tensorboard --logdir logs/
# Then open http://localhost:6006
```

---

## 📖 Python API Usage

### Simple Example
```python
from main import ManuscriptPipeline

# Initialize pipeline
pipeline = ManuscriptPipeline(
    ocr_engine='tesseract',
    translation_method='google'
)

# Process image
result = pipeline.process('path/to/manuscript.jpg')

# Access results
print(f"Sanskrit: {result['sanskrit_text']}")
print(f"English: {result['translation']}")
print(f"Words: {result['word_count']}")
```

### Advanced Example
```python
from main import ManuscriptPipeline

pipeline = ManuscriptPipeline(
    restoration_model_path='models/checkpoints/best_model.pth',
    ocr_engine='ensemble',
    translation_method='ensemble',
    device='cuda'
)

results = pipeline.process(
    image_path='manuscript.jpg',
    save_intermediate=True,
    output_dir='custom_output/'
)
```

---

## 🎨 Output Files Explained

After processing, you'll find in `output/`:

| File | Description |
|------|-------------|
| `*_original.jpg` | Original input image |
| `*_restored.jpg` | AI-restored image (if model used) |
| `*_comparison.jpg` | Side-by-side comparison |
| `*_results.txt` | Human-readable results |
| `*_results.json` | Machine-readable JSON |

---

## 🛠️ Common Tasks

### Task 1: Extract Text Only
```python
from ocr.run_ocr import SanskritOCR

ocr = SanskritOCR(engine='tesseract')
text = ocr.extract_text('manuscript.jpg')
print(text)
```

### Task 2: Translate Text Only
```python
from nlp.translation import SanskritTranslator

translator = SanskritTranslator(method='google')
english = translator.translate("रामः वनं गच्छति")
print(english)
```

### Task 3: Normalize Unicode
```python
from nlp.unicode_normalizer import SanskritTextProcessor

processor = SanskritTextProcessor()
clean_text = processor.normalize("रामः वनं गच्छति")
romanized = processor.to_iast(clean_text)
print(romanized)  # "rāmaḥ vanaṃ gacchati"
```

---

## 📊 Example Output

### Input
Sanskrit manuscript image with degraded text

### Output (JSON)
```json
{
  "ocr_raw": "रामः वनं गच्छति।",
  "ocr_cleaned": "रामः वनं गच्छति।",
  "romanized": "rāmaḥ vanaṃ gacchati",
  "translation": "Rama goes to the forest",
  "word_count": 3,
  "sentences": ["रामः वनं गच्छति।"]
}
```

---

## 🔍 Troubleshooting

### Problem: Module not found
```bash
# Make sure venv is activated
source venv/bin/activate
which python  # Should show venv path
```

### Problem: Tesseract not found
```bash
# Already installed, but verify:
tesseract --version
```

### Problem: Out of memory
```bash
# Reduce batch size
python train.py --batch_size 4
```

### Problem: Slow processing
```bash
# Use GPU if available
python main.py --image_path file.jpg --device cuda

# Or reduce image size
python main.py --image_path file.jpg --max_size 1024
```

---

## 📚 Learn More

- **README.md** - Full documentation
- **GETTING_STARTED.md** - Detailed setup guide
- **PROJECT_SUMMARY.md** - Architecture overview
- **RUN_REPORT.md** - Latest run report
- **demo.ipynb** - Interactive tutorial

---

## 🎉 You're All Set!

The pipeline is ready for production use. Start processing your Sanskrit manuscripts!

**Commands to remember:**
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Process manuscript
python main.py --image_path your_manuscript.jpg

# 3. Check results
cat output/your_manuscript_results.txt
```

**Happy manuscript processing! 🕉️📜**

