# KAGGLE TRAINING GUIDE
## Historic Manuscript Restoration with ViT

This guide explains how to train your manuscript restoration model on Kaggle.

---

## 📋 Quick Start (3 Steps)

### Step 1: Setup Kaggle Notebook
1. Go to https://www.kaggle.com
2. Create a **New Notebook**
3. Enable **GPU**: Settings → Accelerator → GPU T4 (or P100)

### Step 2: Install Dependencies
```python
# Run this in the first cell
!pip install einops lpips -q
```

### Step 3: Copy Training Code
Copy the entire content from `kaggle_training_notebook.py` into a new cell.

---

## 📁 Dataset Structure

Your Kaggle dataset should have this structure:
```
/kaggle/input/your-dataset/
    ├── train/          # Training images (clean manuscripts)
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── val/            # Validation images (optional)
    │   ├── img1.jpg
    │   └── ...
    └── test/           # Test images (optional)
```

**Note:** You only need CLEAN images. The code automatically creates degraded versions for training!

---

## ⚙️ Configuration

In the `main()` function, modify these settings:

```python
# Dataset paths
TRAIN_DIR = '/kaggle/input/your-dataset/train'  # ← Change this
VAL_DIR = '/kaggle/input/your-dataset/val'      # ← Change this or set to None

# Training settings
IMG_SIZE = 256          # Image size (256 recommended)
BATCH_SIZE = 16         # Batch size (reduce if out of memory)
NUM_EPOCHS = 100        # Number of training epochs
MODEL_SIZE = 'base'     # 'tiny', 'small', 'base', or 'large'
NUM_WORKERS = 2         # Data loading workers
```

---

## 🎯 Model Sizes

Choose based on your GPU and dataset:

| Model  | Parameters | GPU Memory | Speed    | Quality |
|--------|-----------|------------|----------|---------|
| tiny   | ~5M       | ~4 GB      | Fast     | Good    |
| small  | ~22M      | ~6 GB      | Medium   | Better  |
| base   | ~86M      | ~10 GB     | Slower   | Best    |
| large  | ~307M     | ~16 GB     | Slowest  | Premium |

**Recommendation:** Start with `'small'` or `'base'` for Kaggle's free GPU.

---

## 📊 What Happens During Training

1. **Synthetic Degradation**: Clean images are automatically degraded with:
   - Gaussian noise
   - Blur (aging effect)
   - Reduced contrast (fading)
   - Salt & pepper noise
   - Yellow tint (aging)
   - Random stains/spots

2. **Training Process**:
   - Model learns to restore degraded → clean
   - Progress bars show loss per epoch
   - Metrics calculated: PSNR, SSIM, LPIPS
   - Best model saved automatically

3. **Output Files** (saved to `/kaggle/working/checkpoints/`):
   - `best_psnr.pth` - Best performing model
   - `epoch_N.pth` - Periodic checkpoints
   - `final.pth` - Final trained model

---

## 💾 Downloading Trained Model

After training completes:

### Option 1: From Kaggle UI
1. Click on **Output** tab (right side)
2. Navigate to `checkpoints/`
3. Download `best_psnr.pth`

### Option 2: From Code
```python
# Add this cell after training
from IPython.display import FileLink
FileLink('/kaggle/working/checkpoints/best_psnr.pth')
```

---

## 🔍 Monitoring Training

The training will show:
```
Epoch 1/100 [Train]: 100%|██████| 50/50 [01:23<00:00]
Epoch 1/100 [Val]:   100%|██████| 10/10 [00:15<00:00]

Epoch 1/100
  Train: loss: 0.1234 | psnr: 25.67 | ssim: 0.8512 | lpips: 0.1234
  Val:   loss: 0.1156 | psnr: 26.45 | ssim: 0.8678 | lpips: 0.1123
  ✓ New best PSNR: 26.45
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory
**Error:** `CUDA out of memory`

**Solutions:**
- Reduce `BATCH_SIZE` (try 8 or 4)
- Use smaller model (`'small'` or `'tiny'`)
- Reduce `IMG_SIZE` (try 128)

### Issue 2: No GPU Available
**Error:** `Using device: cpu`

**Solution:**
- Go to Settings → Accelerator → Select GPU

### Issue 3: Dataset Not Found
**Error:** `Found 0 images in train mode`

**Solution:**
- Check `TRAIN_DIR` path is correct
- Verify images are in supported formats (jpg, png, tif, bmp)
- Make sure dataset is added to notebook (Add Data → Search → Add)

### Issue 4: Training Too Slow
**Solutions:**
- Use smaller model size
- Reduce number of epochs
- Increase `BATCH_SIZE` if you have memory
- Use Kaggle's P100 GPU (if available)

---

## 📈 Expected Training Times (Kaggle GPU)

| Model | Epochs | Images | Time    |
|-------|--------|--------|---------|
| tiny  | 100    | 1000   | ~2 hrs  |
| small | 100    | 1000   | ~4 hrs  |
| base  | 100    | 1000   | ~8 hrs  |
| large | 100    | 1000   | ~16 hrs |

**Note:** Kaggle notebooks have a 12-hour time limit. Plan accordingly!

---

## 🎓 Using the Trained Model

After downloading `best_psnr.pth`, use it for inference:

```python
# Load model
model = create_model(model_size='base', img_size=256)
checkpoint = torch.load('best_psnr.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Restore image
with torch.no_grad():
    degraded_img = load_your_image()  # Your image loading code
    restored_img = model(degraded_img)
```

---

## 📚 Additional Resources

- **Full Project Code**: Check the main repository
- **Dataset Examples**: Look in `data/raw/` for sample images
- **Inference Script**: Use `inference.py` with trained model
- **Streamlit App**: Run `app.py` for web interface

---

## ✅ Training Checklist

Before you start:
- [ ] Kaggle account created
- [ ] GPU enabled in notebook settings
- [ ] Dataset uploaded to Kaggle
- [ ] Dependencies installed (`einops`, `lpips`)
- [ ] Training code copied to notebook
- [ ] Paths updated in configuration
- [ ] Model size selected

---

## 💡 Tips for Best Results

1. **More Data = Better Results**: Try to have at least 500-1000 images
2. **Diverse Data**: Include various manuscript types, ages, conditions
3. **Validation Set**: Use 10-20% of data for validation to monitor overfitting
4. **Experiment**: Try different model sizes and hyperparameters
5. **Save Checkpoints**: They're saved automatically every 5 epochs
6. **Monitor Metrics**: PSNR > 25 and SSIM > 0.8 are good targets

---

## 🆘 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Review the "Common Issues" section above
3. Verify all paths and settings
4. Make sure GPU is enabled
5. Check Kaggle's output logs for details

---

**Good luck with your training! 🚀**

