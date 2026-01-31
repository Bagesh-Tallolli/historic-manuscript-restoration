# 🚀 Kaggle Training Notebook Guide

## 📋 Complete Setup Instructions for Training on Kaggle

This guide will help you train your Historic Manuscript Restoration model on Kaggle with **fully automated dataset download** from Roboflow.

---

## ✅ What's Automated

- ✅ **Dataset Download**: Automatically downloads from Roboflow using hardcoded API keys
- ✅ **Directory Structure**: Creates all necessary folders automatically
- ✅ **Synthetic Degradation**: Applies realistic degradations to clean images
- ✅ **Training Pipeline**: Complete end-to-end training with checkpointing
- ✅ **Metrics Tracking**: PSNR, SSIM, LPIPS calculated automatically
- ✅ **Model Saving**: Best model saved based on validation PSNR

---

## 🎯 Quick Start (3 Steps)

### Step 1: Upload Notebook to Kaggle

1. Go to [Kaggle](https://www.kaggle.com)
2. Click **"Create"** → **"New Notebook"**
3. Click **"File"** → **"Import Notebook"**
4. Upload: `kaggle_training_notebook.ipynb`

### Step 2: Enable GPU

1. Click **"Accelerator"** on the right sidebar
2. Select **"GPU T4 x2"** (or any available GPU)
3. Click **"Save"**

### Step 3: Run All Cells

1. Click **"Run All"** or press `Ctrl+/` (Windows) or `Cmd+/` (Mac)
2. Wait for training to complete (~2-3 hours)
3. Download trained model from outputs

---

## 📦 What's Included in the Notebook

### Cell Structure

1. **📋 Overview & Instructions** - Introduction and setup guide
2. **🔧 Install Dependencies** - Installs required packages
3. **📚 Import Libraries** - Imports all necessary modules
4. **📥 Download Dataset** - **Automatically downloads from Roboflow**
5. **🏗️ Model Architecture** - Vision Transformer implementation
6. **📉 Loss Functions** - Combined L1 + Perceptual loss
7. **🎨 Dataset Class** - Synthetic degradation pipeline
8. **📊 Metrics** - PSNR, SSIM, LPIPS evaluation
9. **🚂 Trainer** - Complete training loop
10. **⚙️ Configuration** - Hyperparameters
11. **📂 Create Datasets** - DataLoader setup
12. **🤖 Create Model** - Model initialization
13. **🚀 Start Training** - Main training execution
14. **📊 Visualize Results** - Sample predictions
15. **💾 Download Model** - Download trained checkpoints

---

## 🔑 Roboflow Configuration (Pre-configured)

The notebook is **hardcoded** with your Roboflow credentials:

```python
ROBOFLOW_API_KEY = "EBJvHlgSWLyW1Ir6ctkH"
ROBOFLOW_WORKSPACE = "neeew"
ROBOFLOW_PROJECT = "yoyoyo-mptyx-ijqfp"
ROBOFLOW_VERSION = 1
```

**You don't need to change anything!** The dataset will download automatically.

---

## 📁 Automatic Directory Structure

The notebook creates this structure automatically:

```
/kaggle/working/
├── dataset/                      # Auto-downloaded from Roboflow
│   ├── train/                    # Training images
│   ├── valid/                    # Validation images (if available)
│   └── test/                     # Test images (if available)
├── checkpoints/                  # Auto-created during training
│   ├── best_psnr.pth            # Best model (highest PSNR)
│   ├── final.pth                # Final epoch model
│   ├── epoch_10.pth             # Checkpoint at epoch 10
│   ├── epoch_20.pth             # Checkpoint at epoch 20
│   └── ...
└── results_visualization.png     # Sample results
```

---

## ⚙️ Training Configuration

Default settings (can be modified in Cell 9):

```python
IMG_SIZE = 256          # Image resolution
BATCH_SIZE = 16         # Batch size
NUM_EPOCHS = 100        # Training epochs
MODEL_SIZE = 'base'     # Model size: tiny/small/base/large
NUM_WORKERS = 2         # Data loading workers
```

### Model Sizes

| Size  | Parameters | Memory | Speed | Quality |
|-------|-----------|--------|-------|---------|
| tiny  | ~11M      | ~2GB   | Fast  | Good    |
| small | ~22M      | ~4GB   | Medium| Better  |
| base  | ~86M      | ~8GB   | Slow  | Best    |
| large | ~307M     | ~16GB  | Slower| Best    |

**Recommendation**: Use `'base'` for best quality on Kaggle's free GPU.

---

## 🎨 Synthetic Degradation Pipeline

The notebook automatically applies realistic degradations:

1. **Gaussian Noise** (σ=0.01-0.08)
2. **Blur** (kernel=3,5,7)
3. **Contrast Reduction** (α=0.6-0.9)
4. **Salt & Pepper Noise** (0.1-1%)
5. **Yellow Tint** (aging effect)
6. **Stains** (circular degradations)

These create realistic training data from clean images!

---

## 📊 Metrics Tracked

- **Loss**: Combined L1 + Perceptual loss
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)

---

## 💾 Downloading Trained Models

After training completes:

1. The last cell will show download links
2. Click to download:
   - `best_psnr.pth` - Use this for inference
   - `final.pth` - Final epoch checkpoint
   - `results_visualization.png` - Sample results

Or manually download from Kaggle's output panel:
- Right sidebar → "Output" → Download files

---

## 🐛 Troubleshooting

### Issue: Dataset download fails

**Solution 1**: Check Roboflow API key
```python
# Verify in Cell 3 that the API key is correct
ROBOFLOW_API_KEY = "EBJvHlgSWLyW1Ir6ctkH"
```

**Solution 2**: Manual dataset upload
1. Download dataset from Roboflow website
2. Upload as Kaggle dataset
3. Modify Cell 3 to use local path:
```python
TRAIN_DIR = "/kaggle/input/your-dataset/train"
VAL_DIR = "/kaggle/input/your-dataset/valid"
```

### Issue: Out of memory

**Solution**: Reduce batch size in Cell 9
```python
BATCH_SIZE = 8  # Or even 4
```

### Issue: Training too slow

**Solution**: Use smaller model in Cell 9
```python
MODEL_SIZE = 'small'  # Instead of 'base'
```

### Issue: No GPU available

**Solution**: 
1. Check Kaggle quotas (30h GPU/week free)
2. Use CPU (very slow):
   - Training will work but take 10-20x longer

---

## 📈 Expected Training Time

On Kaggle GPU (Tesla T4):

| Configuration | Time/Epoch | Total (100 epochs) |
|--------------|------------|-------------------|
| Tiny + BS16  | ~1 min     | ~1.5 hours        |
| Small + BS16 | ~2 min     | ~3 hours          |
| Base + BS16  | ~3 min     | ~5 hours          |
| Large + BS8  | ~6 min     | ~10 hours         |

---

## 🎯 Expected Performance

After 100 epochs on manuscript dataset:

- **PSNR**: 28-35 dB (higher is better)
- **SSIM**: 0.85-0.95 (closer to 1 is better)
- **LPIPS**: 0.05-0.15 (lower is better)

---

## 🔄 Using the Trained Model Locally

After downloading `best_psnr.pth`:

```python
import torch
from models.vit_restorer import ViTRestorer

# Create model (same config as training)
model = ViTRestorer(img_size=256, embed_dim=768, depth=12, num_heads=12)

# Load trained weights
checkpoint = torch.load('best_psnr.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    restored = model(degraded_image)
```

---

## 🎓 Advanced: Customizing Training

### Change Dataset Source

Modify Cell 3 to use different sources:

```python
# Option 1: Kaggle Dataset
from kaggle import api
api.dataset_download_files('username/dataset-name', path='/kaggle/working/dataset', unzip=True)

# Option 2: Google Drive
import gdown
gdown.download('https://drive.google.com/uc?id=FILE_ID', 'dataset.zip')

# Option 3: Direct URL
!wget -O dataset.zip "https://example.com/dataset.zip"
!unzip dataset.zip -d /kaggle/working/dataset
```

### Modify Training Loop

Add custom callbacks, learning rate schedules, or augmentations in Cell 8 (Trainer class).

### Change Loss Function

Modify Cell 5 to use different losses:
- MSE Loss
- Perceptual Loss only
- GAN Loss (requires discriminator)
- Custom weighted combination

---

## 📚 Additional Resources

- **Roboflow Project**: https://app.roboflow.com/neeew/yoyoyo-mptyx-ijqfp
- **Vision Transformer Paper**: https://arxiv.org/abs/2010.11929
- **Image Restoration Survey**: https://arxiv.org/abs/2003.03808

---

## ✅ Checklist Before Running

- [ ] Uploaded notebook to Kaggle
- [ ] Enabled GPU accelerator
- [ ] Verified Roboflow credentials in Cell 3
- [ ] (Optional) Adjusted hyperparameters in Cell 9
- [ ] Ready to click "Run All"

---

## 🎉 Success!

If everything works, you'll see:

```
✅ TRAINING COMPLETED SUCCESSFULLY!
📁 Checkpoints saved to: /kaggle/working/checkpoints/
🏆 Best model: /kaggle/working/checkpoints/best_psnr.pth
📊 Best validation PSNR: 32.45
```

Download your model and start restoring manuscripts! 🏛️

---

## 🆘 Need Help?

If you encounter issues:

1. Check the troubleshooting section above
2. Review Kaggle notebook output for error messages
3. Verify GPU is enabled
4. Ensure dataset downloaded successfully (check Cell 3 output)

---

**Happy Training! 🚀**

