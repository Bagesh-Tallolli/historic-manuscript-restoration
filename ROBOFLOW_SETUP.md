# 🚀 Roboflow Sanskrit OCR Dataset - Quick Setup Guide

**Dataset:** https://universe.roboflow.com/sanskritocr/yoyoyo-mptyx/browse

---

## 📋 What is this Dataset?

The Roboflow Sanskrit OCR dataset contains images of Sanskrit text in Devanagari script, perfect for training document restoration and OCR models.

**Key Features:**
- ✅ Sanskrit manuscript images
- ✅ Devanagari script
- ✅ Pre-split into train/valid/test sets
- ✅ Annotations included
- ✅ Free for academic/research use

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Install Roboflow Package
```bash
cd /home/bagesh/EL-project
source venv/bin/activate
pip install roboflow
```

### Step 2: Get Your API Key

1. **Sign up/Login:** https://app.roboflow.com/
2. **Navigate to:** Settings → Roboflow API
3. **Copy** your API key

💡 **Tip:** It looks like: `aBcDeFgHiJkLmNoPqRsTuVwXyZ123456`

### Step 3: Download Dataset

**Option A: Using the Download Script (Recommended)**
```bash
# See instructions first
python download_roboflow_dataset.py --instructions

# Download with your API key
python download_roboflow_dataset.py --api-key YOUR_API_KEY
```

**Option B: Manual Python Code**
```python
from roboflow import Roboflow

# Initialize with your API key
rf = Roboflow(api_key="YOUR_API_KEY")

# Access the Sanskrit OCR project
project = rf.workspace("sanskritocr").project("yoyoyo-mptyx")

# Download version 1 in folder format
dataset = project.version(1).download("folder")
```

### Step 4: Verify Download
```bash
# Check training images
ls data/raw/train/ | head -10

# Count images
echo "Training images: $(ls data/raw/train/*.jpg data/raw/train/*.png 2>/dev/null | wc -l)"
echo "Validation images: $(ls data/raw/val/*.jpg data/raw/val/*.png 2>/dev/null | wc -l)"
echo "Test images: $(ls data/raw/test/*.jpg data/raw/test/*.png 2>/dev/null | wc -l)"
```

---

## 🎓 Start Training

Once downloaded, you're ready to train!

### Basic Training Command
```bash
python train.py \
    --train_dir data/raw/train \
    --val_dir data/raw/val \
    --epochs 100 \
    --batch_size 16
```

### Advanced Training with Monitoring
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

---

## 📊 Dataset Information

### Expected Size
- **Training set:** ~70-80% of images
- **Validation set:** ~10-15% of images
- **Test set:** ~10-15% of images

### Image Characteristics
- **Format:** JPG/PNG
- **Content:** Sanskrit text in Devanagari
- **Resolution:** Varies (will be resized during training)
- **Annotations:** Included (can be used for OCR training)

---

## 🔧 Troubleshooting

### Issue: "roboflow module not found"
```bash
pip install roboflow
```

### Issue: "Invalid API key"
**Check:**
1. API key is copied correctly (no extra spaces)
2. You're logged into Roboflow
3. You have access to the dataset

**Get a new key:**
https://app.roboflow.com/settings/api

### Issue: "Download failed"
**Solutions:**
1. Check internet connection
2. Try downloading to a different directory
3. Use manual Python code (see Option B above)

### Issue: "No images in data/raw/"
**Check download location:**
```bash
find /home/bagesh/EL-project -name "*.jpg" -o -name "*.png" | grep -v __pycache__
```

**Re-organize manually:**
```bash
# If images are in data/datasets/roboflow_sanskrit/
cp -r data/datasets/roboflow_sanskrit/train/* data/raw/train/
cp -r data/datasets/roboflow_sanskrit/valid/* data/raw/val/
cp -r data/datasets/roboflow_sanskrit/test/* data/raw/test/
```

---

## 📚 Dataset Formats

Roboflow supports multiple export formats:

| Format | Use Case |
|--------|----------|
| `folder` | Raw images (for restoration) ✅ **Use this** |
| `coco` | Object detection |
| `yolov5` | YOLO training |
| `voc` | Pascal VOC format |
| `tfrecord` | TensorFlow |

For document restoration, use **`folder`** format (default in our script).

---

## 🎯 Next Steps After Download

1. **Verify dataset:**
   ```bash
   python -c "
   from pathlib import Path
   train_imgs = list(Path('data/raw/train').glob('*.jpg')) + list(Path('data/raw/train').glob('*.png'))
   val_imgs = list(Path('data/raw/val').glob('*.jpg')) + list(Path('data/raw/val').glob('*.png'))
   print(f'✅ Training: {len(train_imgs)} images')
   print(f'✅ Validation: {len(val_imgs)} images')
   "
   ```

2. **Test loading:**
   ```bash
   python test_setup.py
   ```

3. **Start training:**
   ```bash
   python train.py --train_dir data/raw/train --val_dir data/raw/val
   ```

4. **Monitor progress:**
   - Check `logs/` directory
   - Use TensorBoard or W&B

---

## 🔗 Useful Links

- **Dataset Page:** https://universe.roboflow.com/sanskritocr/yoyoyo-mptyx/browse
- **Roboflow Docs:** https://docs.roboflow.com/
- **API Reference:** https://docs.roboflow.com/api-reference/authentication
- **Support:** https://help.roboflow.com/

---

## ✅ Success Checklist

After setup, you should have:

- [ ] Roboflow package installed
- [ ] API key obtained
- [ ] Dataset downloaded
- [ ] Images in `data/raw/train/`
- [ ] Images in `data/raw/val/`
- [ ] Images in `data/raw/test/`
- [ ] Verified image count
- [ ] Ready to train

---

## 💡 Tips

1. **Free Account Limits:** Roboflow free tier has generous limits, but check their pricing page for details
2. **Multiple Downloads:** You can re-download without using extra credits
3. **Different Formats:** Try different export formats for different tasks
4. **Version Control:** Dataset version is tracked (currently using v1)
5. **Annotations:** The dataset includes annotations - useful for OCR training later

---

## 📞 Need Help?

- Check **DATASET_REQUIREMENTS.md** for general dataset info
- See **GETTING_STARTED.md** for project setup
- Review **QUICKSTART.md** for training commands
- Check **RUN_REPORT.md** for example results

---

**Ready to train your Sanskrit manuscript restoration model!** 🕉️📜


