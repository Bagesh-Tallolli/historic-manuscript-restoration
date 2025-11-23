

# 📚 Dataset Requirements for Training - Sanskrit Manuscript Pipeline

**Last Updated:** November 21, 2025

---

## 📊 Overview

The Sanskrit Manuscript Restoration model requires image datasets to learn how to restore degraded manuscript images. This document details the dataset requirements, formats, and sources.

---

## 🎯 Dataset Types

### Option 1: Paired Dataset (Recommended)
**Description:** Pairs of degraded and clean versions of the same manuscript

**Structure:**
```
data/
├── raw/
│   ├── clean/
│   │   ├── manuscript_001.jpg
│   │   ├── manuscript_002.jpg
│   │   └── ...
│   └── degraded/
│       ├── manuscript_001.jpg
│       ├── manuscript_002.jpg
│       └── ...
```

**Use Case:**
- Best for supervised learning
- Highest quality restoration
- Requires manual pairing or professional scanning

---

### Option 2: Single Clean Images with Synthetic Degradation (Default)
**Description:** Clean manuscript images that will be synthetically degraded during training

**Structure:**
```
data/
└── raw/
    ├── manuscript_001.jpg
    ├── manuscript_002.jpg
    ├── manuscript_003.png
    └── ...
```

**Use Case:**
- Easier to collect (only clean images needed)
- System automatically creates degraded versions
- **Currently implemented and recommended for getting started**

---

## 📋 Detailed Requirements

### 1. Image Format
**Supported Formats:**
- ✅ JPEG/JPG (`.jpg`, `.jpeg`)
- ✅ PNG (`.png`)
- ✅ TIFF (`.tif`, `.tiff`)
- ✅ BMP (`.bmp`)

**Recommended:**
- PNG or TIFF for best quality (lossless)
- JPEG is acceptable but may have compression artifacts

---

### 2. Image Size

**Minimum Resolution:**
- 256 × 256 pixels (absolute minimum)
- 512 × 512 pixels (recommended minimum)
- 1024 × 1024 pixels or higher (ideal)

**Training Size:**
- Images will be resized to 256×256 during training (configurable)
- Original aspect ratio is maintained during preprocessing
- Higher resolution = better quality but slower training

**Note:** The model can process images of any size during inference, but training uses fixed-size patches.

---

### 3. Image Content

**Required:**
- ✅ Sanskrit manuscripts in Devanagari script
- ✅ Historical documents with Sanskrit text
- ✅ Scanned palm leaf manuscripts
- ✅ Ancient texts on paper/parchment

**Also Works With:**
- ✅ Hindi text in Devanagari
- ✅ Other Indic scripts (requires OCR language change)
- ✅ Mixed script documents

**Avoid:**
- ❌ Blank or mostly empty pages
- ❌ Images with excessive watermarks
- ❌ Extremely low resolution (<256px)
- ❌ Completely illegible documents

---

### 4. Dataset Size

**Minimum:** 50-100 images (for testing/demo)
**Recommended:** 1,000+ images (for good results)
**Ideal:** 5,000+ images (for production quality)

**Why more is better:**
- Better generalization to different degradation types
- Improved handling of various writing styles
- More robust to edge cases

---

### 5. Degradation Types Supported

The synthetic degradation system simulates:

#### Physical Degradation
- 📄 **Fading:** Reduced contrast and brightness
- 🌫️ **Blur:** Gaussian blur (aging effect)
- 📝 **Stains:** Random dark spots and marks
- 🎨 **Color Shift:** Yellowish aging tint

#### Digital Noise
- 🔊 **Gaussian Noise:** Random pixel variations
- ⚡ **Salt & Pepper:** Random black/white pixels
- 📉 **Contrast Loss:** Reduced dynamic range

**Customizable:** All degradation parameters are adjustable in `utils/dataset_loader.py`

---

## 🗂️ Directory Structure

### Recommended Setup

```
/home/bagesh/EL-project/
└── data/
    ├── raw/                    # Training data
    │   ├── train/              # Training images
    │   │   ├── manuscript_001.jpg
    │   │   ├── manuscript_002.jpg
    │   │   └── ...
    │   ├── val/                # Validation images (optional)
    │   │   ├── val_001.jpg
    │   │   └── ...
    │   └── test/               # Test images (optional)
    │       ├── test_001.jpg
    │       └── ...
    ├── processed/              # Auto-generated during training
    └── datasets/               # Downloaded datasets
        └── samples/
            └── test_sample.png
```

### Alternative Simple Setup

```
/home/bagesh/EL-project/
└── data/
    └── raw/                    # All images here
        ├── *.jpg
        ├── *.png
        └── ...
```

The system will automatically split into train/val if not pre-separated.

---

## 📥 Where to Get Datasets

### 1. Public Datasets

#### **Roboflow Sanskrit OCR Dataset (Recommended - NEW!)**
- **URL:** https://universe.roboflow.com/sanskritocr/yoyoyo-mptyx/browse
- **Description:** Sanskrit OCR dataset with annotations
- **Content:** Sanskrit manuscript images in Devanagari script
- **Format:** Images with annotations (various formats available)
- **License:** Check project page for license details
- **Setup:** Use `python download_roboflow_dataset.py --api-key YOUR_KEY`

**Quick Start:**
```bash
# Install Roboflow package
pip install roboflow

# Download dataset (see instructions below)
python download_roboflow_dataset.py --instructions

# Or download directly with API key
python download_roboflow_dataset.py --api-key YOUR_API_KEY
```

#### **e-Granthalaya (GRETIL)**
- **URL:** https://gretil.sub.uni-goettingen.de/
- **Description:** Digital library of Sanskrit texts
- **Content:** Thousands of Sanskrit manuscripts
- **Format:** Various (PDF, images)
- **License:** Academic use

#### **Sanskrit Documents**
- **URL:** https://sanskritdocuments.org/
- **Description:** Collection of Sanskrit texts in various scripts
- **Content:** Multiple scripts, large collection
- **Format:** PDF, text, images

#### **Digital Library of India**
- **URL:** https://dli.sanskritdictionary.com/
- **Description:** Scanned manuscripts and texts
- **Content:** Historical manuscripts
- **Format:** Page scans

#### **Kaggle Devanagari Datasets**
- **URL:** https://www.kaggle.com/search?q=devanagari
- **Description:** Various Devanagari OCR datasets
- **Popular:**
  - Devanagari Character Dataset
  - Devanagari Handwritten Character Dataset
- **Format:** Images, CSV

#### **IIIT Handwritten Dataset**
- **URL:** https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data
- **Description:** Handwritten Indic script datasets
- **Content:** Multiple Indic scripts including Devanagari
- **Format:** Images with annotations

---

### 2. Create Your Own Dataset

#### **Scan Physical Manuscripts**
```bash
# Scanning recommendations:
- DPI: 300-600 (higher for historical documents)
- Color: RGB (even for black/white documents)
- Format: TIFF or PNG (lossless)
```

#### **Photograph Manuscripts**
```bash
# Photography tips:
- Use good lighting (diffused, not direct)
- Keep camera parallel to page
- Avoid shadows and glare
- Minimum resolution: 2048×2048
```

#### **Generate Synthetic Data**
```python
# The system can work with clean images only
# Place clean manuscript images in data/raw/
# System will automatically degrade them during training
```

---

## 🚀 Quick Start with Datasets

### Method 1: Download Samples (Automatic)

```bash
cd /home/bagesh/EL-project
source venv/bin/activate
python dataset_downloader.py
```

This will:
- Show dataset sources
- Setup directory structure
- Provide download instructions

---

### Method 2: Manual Dataset Setup

#### Step 1: Create Directories
```bash
mkdir -p data/raw/train
mkdir -p data/raw/val
```

#### Step 2: Add Your Images
```bash
# Copy your manuscript images
cp /path/to/your/manuscripts/*.jpg data/raw/train/
cp /path/to/validation/images/*.jpg data/raw/val/
```

#### Step 3: Verify
```bash
# Check number of images
ls data/raw/train/ | wc -l
ls data/raw/val/ | wc -l
```

---

### Method 4: Use Kaggle Datasets

#### Step 1: Install Kaggle CLI
```bash
pip install kaggle
```

#### Step 2: Setup Kaggle Credentials
```bash
# Get API token from https://www.kaggle.com/account
mkdir -p ~/.kaggle
# Copy kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Step 3: Download Dataset
```bash
# Example: Download Devanagari dataset
kaggle datasets download -d ashokpant/devanagari-character-dataset
unzip devanagari-character-dataset.zip -d data/raw/
```

---

## 🎓 Training with Your Dataset

### Basic Training (Synthetic Degradation)

```bash
cd /home/bagesh/EL-project
source venv/bin/activate

python train.py \
    --train_dir data/raw \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4
```

### Advanced Training (Custom Settings)

```bash
python train.py \
    --train_dir data/raw/train \
    --val_dir data/raw/val \
    --img_size 512 \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_dir models/checkpoints \
    --use_wandb
```

### Training Parameters Explained

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--train_dir` | Training images directory | `data/raw` | Your path |
| `--val_dir` | Validation directory | `None` | 10-20% of data |
| `--img_size` | Training image size | `256` | 256-512 |
| `--epochs` | Number of epochs | `100` | 100-200 |
| `--batch_size` | Batch size | `16` | 8-32 (based on GPU) |
| `--lr` | Learning rate | `1e-4` | 1e-4 to 1e-5 |
| `--use_wandb` | Log to W&B | `False` | True (for monitoring) |

---

## ⚙️ Dataset Configuration

### Edit `config.yaml`

```yaml
# Data Configuration
data:
  train_dir: 'data/raw/train'
  val_dir: 'data/raw/val'
  synthetic_degradation: true  # Auto-degrade clean images
  augmentation: true           # Apply data augmentation

# Training Configuration
training:
  epochs: 100
  batch_size: 16
  img_size: 256
```

### Custom Degradation

Edit `utils/dataset_loader.py` to customize degradation:

```python
# In _degrade_image() method:

# Adjust noise levels
noise_sigma = random.uniform(0.01, 0.08)  # Increase for more noise

# Adjust blur amount
kernel_size = random.choice([3, 5, 7, 9])  # Larger = more blur

# Adjust fading
alpha = random.uniform(0.5, 0.9)  # Lower = more fading
```

---

## 📊 Dataset Quality Checklist

Before training, verify:

- [ ] **Sufficient Quantity:** 100+ images minimum
- [ ] **Consistent Format:** All images in supported formats
- [ ] **Adequate Resolution:** 512×512 or higher
- [ ] **Proper Content:** Sanskrit/Devanagari text visible
- [ ] **Organized Structure:** Files in correct directories
- [ ] **No Corruption:** All images load correctly
- [ ] **Varied Content:** Different styles, ages, conditions

**Quick Test:**
```bash
# Test if images load correctly
python -c "
import cv2
from pathlib import Path

images = list(Path('data/raw').glob('*.jpg'))
print(f'Found {len(images)} images')

for img_path in images[:5]:
    img = cv2.imread(str(img_path))
    if img is not None:
        print(f'✓ {img_path.name}: {img.shape}')
    else:
        print(f'✗ {img_path.name}: FAILED TO LOAD')
"
```

---

## 🔧 Troubleshooting

### Issue: "No images found"
**Solution:**
```bash
# Check directory structure
ls -R data/raw/

# Verify file extensions
find data/raw -name "*.jpg" -o -name "*.png"
```

### Issue: "Out of memory during training"
**Solution:**
```bash
# Reduce batch size
python train.py --batch_size 4

# Or reduce image size
python train.py --img_size 128
```

### Issue: "Images are too small"
**Solution:**
```bash
# Find images smaller than 256x256
python -c "
import cv2
from pathlib import Path

for img_path in Path('data/raw').glob('*.jpg'):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    if h < 256 or w < 256:
        print(f'{img_path.name}: {w}x{h} - TOO SMALL')
"
```

### Issue: "Dataset is imbalanced"
**Solution:**
- Use data augmentation (automatically enabled)
- Add more diverse samples
- Use synthetic degradation with various parameters

---

## 📈 Expected Results by Dataset Size

| Dataset Size | Training Time | Expected PSNR | Quality |
|--------------|---------------|---------------|---------|
| 50-100 images | 1-2 hours | 20-25 dB | Demo |
| 500-1000 images | 5-10 hours | 25-30 dB | Good |
| 1000-5000 images | 10-24 hours | 30-35 dB | Very Good |
| 5000+ images | 24-48 hours | 35+ dB | Excellent |

*Times based on single GPU training*

---

## 🎯 Best Practices

### Data Collection
1. **Diversity:** Include various manuscript conditions
2. **Quality:** Higher resolution is better
3. **Quantity:** More is better, but quality matters
4. **Balance:** Mix of different degradation levels

### Data Preparation
1. **Organize:** Proper directory structure
2. **Clean:** Remove corrupted/duplicate images
3. **Validate:** Test-load all images before training
4. **Backup:** Keep original copies

### Training Strategy
1. **Start Small:** Test with 100 images first
2. **Iterate:** Train, evaluate, adjust
3. **Monitor:** Use TensorBoard or W&B
4. **Save Often:** Regular checkpoints

---

## 📚 Additional Resources

### Documentation
- **README.md** - General project information
- **GETTING_STARTED.md** - Setup instructions
- **QUICKSTART.md** - Quick commands reference
- **RUN_REPORT.md** - Latest run results

### Code Files
- **utils/dataset_loader.py** - Dataset loading logic
- **train.py** - Training script
- **config.yaml** - Configuration settings

### External Links
- **PyTorch Datasets:** https://pytorch.org/vision/stable/datasets.html
- **Image Augmentation:** https://albumentations.ai/
- **Devanagari OCR:** https://tesseract-ocr.github.io/

---

## ✅ Summary

**To get started with training:**

1. **Collect images:** 100+ Sanskrit manuscript images
2. **Organize:** Place in `data/raw/` directory
3. **Configure:** Edit `config.yaml` if needed
4. **Train:** Run `python train.py --train_dir data/raw`
5. **Monitor:** Check progress in `logs/`
6. **Evaluate:** Test on validation images

**Remember:**
- The system works with clean images (synthetic degradation)
- More data = better results
- Start small, scale up
- Monitor training metrics

---

**Need help?** Check the troubleshooting section or review the training script documentation.

**Ready to train?** See `QUICKSTART.md` for training commands.

🕉️ Happy training! 📜

