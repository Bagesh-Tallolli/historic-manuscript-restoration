# 🕉️ Sanskrit Manuscript Restoration & Translation Pipeline

**Complete end-to-end pipeline for restoring, extracting text from, and translating ancient Sanskrit manuscripts.**

---

## 🌟 Features

✅ **ViT-based Image Restoration** — Removes noise, fading, and blur from manuscript images  
✅ **OCR (Tesseract + TrOCR)** — Extracts Devanagari text from restored images  
✅ **Unicode Normalization** — Converts romanized/broken text to proper Devanagari Unicode  
✅ **Sanskrit→English Translation** — Translates normalized Sanskrit to English  
✅ **Full Pipeline Integration** — Single function processes entire workflow  

---

## 📁 Project Structure

```
project/
├── data/
│   ├── raw/              # Original manuscript images
│   ├── processed/        # Restored images
│   └── datasets/         # Downloaded datasets
├── models/
│   ├── vit_restorer.py   # Vision Transformer restoration model
│   └── checkpoints/      # Saved model weights
├── ocr/
│   ├── preprocess.py     # Image preprocessing for OCR
│   └── run_ocr.py        # OCR pipeline (Tesseract + TrOCR)
├── nlp/
│   ├── unicode_normalizer.py  # Devanagari normalization
│   └── translation.py    # Sanskrit→English translation
├── utils/
│   ├── metrics.py        # PSNR, SSIM, etc.
│   ├── dataset_loader.py # Data loading utilities
│   └── visualization.py  # Plotting & visualization
├── dataset_downloader.py # Auto-download datasets
├── train.py              # Training script for ViT
├── inference.py          # Run inference on single image
├── main.py               # Full pipeline execution
├── demo.ipynb            # Jupyter demo notebook
└── requirements.txt      # Dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

**✅ Dependency conflicts fixed!** The installation now works without any conflicts.

**Quick install:**
```bash
pip install -r requirements.txt
```

**Or use the installation script:**
```bash
./install.sh
```

**See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for detailed instructions and troubleshooting.**

### 2. Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-san
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Download Datasets

**Option A: Roboflow Sanskrit OCR Dataset (Recommended)**
```bash
# Interactive setup
bash setup_roboflow.sh

# Or with API key directly
python3 download_roboflow_dataset.py --api-key YOUR_API_KEY
```

See **ROBOFLOW_SETUP.md** for detailed instructions.

**Option B: Sample Datasets**
```bash
python dataset_downloader.py
```

See **DATASET_REQUIREMENTS.md** for more dataset sources.

### 4. Train the ViT Restoration Model

```bash
python train.py --epochs 100 --batch_size 16 --lr 1e-4
```

### 5. Run Full Pipeline on a Manuscript Image

```bash
python main.py --image_path data/raw/manuscript.jpg
```

Or use Python:

```python
from main import process_manuscript

result = process_manuscript("data/raw/manuscript.jpg")
print("Translation:", result['translation'])
```

---

## 📊 Pipeline Stages

### Stage 1: Image Restoration (ViT)
- Input: Noisy/faded manuscript image
- Output: Cleaned, enhanced image
- Metrics: PSNR, SSIM

### Stage 2: OCR
- Input: Restored image
- Output: Raw Devanagari/romanized text
- Tools: Tesseract, TrOCR

### Stage 3: Unicode Normalization
- Input: Raw OCR output
- Output: Clean Unicode Devanagari
- Handles: IAST, ITRANS, broken Unicode

### Stage 4: Translation
- Input: Normalized Sanskrit
- Output: English translation
- Model: IndicTrans2 / Google Translate API

---

## 🎯 Model Architecture

**Vision Transformer (ViT) for Image Restoration:**
- Patch size: 16×16
- Embedding dim: 768
- Transformer layers: 12
- Attention heads: 12
- Skip connections for detail preservation
- Loss: L1 + Perceptual loss

---

## 📈 Training

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/
```

Or use Weights & Biases:

```bash
wandb login
python train.py --use_wandb
```

---

## 🧪 Example Usage

```python
from main import process_manuscript

# Process a manuscript image
result = process_manuscript(
    image_path="data/raw/palm_leaf.jpg",
    save_output=True
)

print("OCR Raw:", result['ocr_text_raw'])
print("OCR Cleaned:", result['ocr_text_cleaned'])
print("Translation:", result['translation'])

# Visualize restoration
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(result['original_image'])
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(result['restored_image'])
plt.title('Restored')
plt.show()
```

---

## 📚 Supported Datasets

- e-Granthalaya Manuscripts
- Sanskrit Palm Leaf Dataset
- Kaggle Devanagari OCR Dataset
- IAM Handwriting Database

---

## 🛠️ Advanced Configuration

Edit `config.yaml` to customize:
- Model hyperparameters
- OCR settings
- Translation API keys
- Data augmentation parameters

---

## 📝 Citation

If you use this project, please cite:

```bibtex
@software{sanskrit_manuscript_pipeline,
  title={Sanskrit Manuscript Restoration and Translation Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/sanskrit-manuscript-pipeline}
}
```

---

## 📄 License

MIT License — See LICENSE file for details.

---

## 🙏 Acknowledgments

- HuggingFace Transformers
- PyTorch Team
- Tesseract OCR
- Indic NLP Library
- AI4Bharat (IndicTrans2)

---

**Built with ❤️ for preserving ancient Sanskrit manuscripts**

