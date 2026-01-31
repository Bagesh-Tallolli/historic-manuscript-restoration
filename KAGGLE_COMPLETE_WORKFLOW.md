# 🚀 KAGGLE TRAINING - COMPLETE WORKFLOW

## ✨ What's New: Dynamic Dataset Download!

Your Kaggle training notebook now supports **automatic dataset downloads** from multiple sources!

---

## 📋 Complete 3-Step Process

### STEP 1: Setup Kaggle Notebook (2 minutes)

```python
# Cell 1: Install dependencies
!pip install einops lpips roboflow gdown kaggle -q
```

### STEP 2: Copy Training Code (1 minute)

Copy the entire `kaggle_training_notebook.py` content into Cell 2.

Get it from:
- GitHub: https://github.com/Bagesh-Tallolli/historic-manuscript-restoration
- Local file: `/home/bagesh/EL-project/kaggle_training_notebook.py`

**✨ GOOD NEWS:** The notebook is **already configured** with your Roboflow dataset!
- API Key: Pre-filled ✅
- Workspace: `neeew` ✅
- Project: `yoyoyo-mptyx-ijqfp` ✅
- Just run it! No configuration needed!

### STEP 3: Run Training (1 click!)

Simply run:
```python
# Cell 3
if __name__ == "__main__":
    main()
```

**That's it!** The dataset will download automatically and training will start!

---

## 🎯 Choose Your Dataset Method

### Method 1: Roboflow (BEST FOR MANUSCRIPTS) 🏆

**Perfect for:** Professional manuscript datasets, labeled data

**Setup:**
```python
USE_ROBOFLOW = True
ROBOFLOW_CONFIG = {
    'api_key': 'your_api_key_here',  # From https://app.roboflow.com
    'workspace': 'your-workspace',
    'project': 'manuscript-dataset',
    'version': 1
}
```

**Get API Key:**
1. Go to https://app.roboflow.com
2. Sign up (free)
3. Profile → Roboflow API → Copy key

---

### Method 2: Kaggle Dataset

**Perfect for:** Public datasets, community datasets

**Setup:**
```python
USE_KAGGLE_DATASET = True
KAGGLE_DATASET_NAME = 'username/dataset-name'
```

**Get Dataset Name:**
1. Browse https://www.kaggle.com/datasets
2. Find a dataset
3. Copy from URL: `kaggle.com/datasets/USERNAME/DATASET-NAME`

**One-time API Setup:**
```python
# Run once in a cell:
!mkdir -p /root/.kaggle
# Upload your kaggle.json, then:
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
```

---

### Method 3: Google Drive

**Perfect for:** Your own datasets, private data

**Setup:**
```python
USE_GOOGLE_DRIVE = True
GOOGLE_DRIVE_FILE_ID = '1a2b3c4d5e6f7g8h9i0j'
```

**Get File ID:**
1. Upload dataset.zip to Google Drive
2. Right-click → Share → "Anyone with link"
3. Copy link: `drive.google.com/file/d/FILE_ID_HERE/view`
4. Extract the `FILE_ID_HERE` part

---

### Method 4: Direct URL

**Perfect for:** Public archives, direct downloads

**Setup:**
```python
USE_URL_DOWNLOAD = True
DATASET_URL = 'https://example.com/dataset.zip'
```

---

### Method 5: Manual Upload

**Perfect for:** When you prefer UI, verified datasets

**Setup:**
1. In Kaggle: Click "Add Data" → Upload your dataset
2. In code:
```python
USE_KAGGLE_INPUT = True
TRAIN_DIR = '/kaggle/input/your-dataset-name/train'
```

---

### Method 6: Sample Dataset

**Perfect for:** Testing, learning the system

**Setup:**
```python
USE_SAMPLE_DATASET = True
```

Creates 50 synthetic images automatically. No real dataset needed!

---

## 📊 Full Configuration Example

Here's what's already configured in your notebook:

```python
def main():
    # ========== DATASET METHOD ==========
    
    # ✅ ALREADY CONFIGURED - Your Roboflow manuscript dataset
    USE_ROBOFLOW = True
    ROBOFLOW_CONFIG = {
        'api_key': 'EBJvHlgSWLyW1Ir6ctkH',
        'workspace': 'neeew',
        'project': 'yoyoyo-mptyx-ijqfp',
        'version': 1
    }
    # Project: https://app.roboflow.com/neeew/yoyoyo-mptyx-ijqfp
    
    # All others are disabled
    USE_KAGGLE_DATASET = False
    USE_URL_DOWNLOAD = False
    USE_GOOGLE_DRIVE = False
    USE_KAGGLE_INPUT = False
    USE_SAMPLE_DATASET = False
    
    # ========== TRAINING SETTINGS (You can modify these) ==========
    IMG_SIZE = 256
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    MODEL_SIZE = 'base'
    NUM_WORKERS = 2
    
    # ... rest of the code runs automatically ...
```

**Note:** You can change the training settings (batch size, epochs, model size) if needed,
but the dataset configuration is ready to go!

---

## ⚙️ Training Configuration Options

### Model Sizes

```python
MODEL_SIZE = 'tiny'   # Fast, 5M params, ~4GB GPU
MODEL_SIZE = 'small'  # Balanced, 22M params, ~6GB GPU
MODEL_SIZE = 'base'   # Best quality, 86M params, ~10GB GPU ✓ RECOMMENDED
MODEL_SIZE = 'large'  # Premium, 307M params, ~16GB GPU
```

### Memory Management

If you get "CUDA out of memory":
```python
BATCH_SIZE = 8        # Reduce from 16
MODEL_SIZE = 'small'  # Use smaller model
IMG_SIZE = 128        # Reduce image size (not recommended)
```

### Quick Training

For fast testing:
```python
NUM_EPOCHS = 10       # Quick test
MODEL_SIZE = 'tiny'   # Fastest
USE_SAMPLE_DATASET = True  # Test data
```

---

## 📥 What Happens During Training

```
============================================================
DOWNLOADING DATASET FROM ROBOFLOW
============================================================
📥 Downloading dataset from Roboflow...
✓ Dataset downloaded to: /kaggle/working/dataset

============================================================
TRAINING CONFIGURATION
============================================================
Device: cuda
Train directory: /kaggle/working/dataset/train
Image size: 256
Batch size: 16
Epochs: 100
Model size: base

============================================================
LOADING DATASETS
============================================================
Found 1000 images in train mode
Found 200 images in val mode

============================================================
CREATING MODEL
============================================================
Model architecture: ViT-BASE
Total parameters: 86,342,400
Trainable parameters: 86,342,400

============================================================
STARTING TRAINING
============================================================
Epoch 1/100 [Train]: 100%|██████| 62/62 [01:45<00:00]
Epoch 1/100 [Val]:   100%|██████| 12/12 [00:18<00:00]

Epoch 1/100
  Train: loss: 0.1234 | psnr: 25.67 | ssim: 0.8512
  Val:   loss: 0.1156 | psnr: 26.45 | ssim: 0.8678
  ✓ New best PSNR: 26.45

... (continues for all epochs) ...

============================================================
✅ TRAINING COMPLETED SUCCESSFULLY!
============================================================
📁 Checkpoints saved to: /kaggle/working/checkpoints/
🏆 Best model: /kaggle/working/checkpoints/best_psnr.pth
============================================================
```

---

## 💾 Download Your Trained Model

### Method 1: Automatic Link
The script automatically displays a download link at the end!

### Method 2: Manual Download
1. Click "Output" tab (right side of Kaggle)
2. Navigate to `checkpoints/`
3. Download `best_psnr.pth`

### Method 3: Add to Code
```python
# Add this cell after training
from IPython.display import FileLink
display(FileLink('/kaggle/working/checkpoints/best_psnr.pth'))
```

---

## 📚 Available Documentation

1. **kaggle_training_notebook.py** - Complete training script
2. **KAGGLE_TRAINING_GUIDE.md** - Detailed guide
3. **KAGGLE_QUICK_START.md** - Quick reference
4. **KAGGLE_DATASET_DOWNLOAD_EXAMPLES.md** - All dataset methods explained
5. **This file** - Complete workflow overview

---

## ✅ Pre-Flight Checklist

Before running:
- [ ] Kaggle notebook created
- [ ] GPU enabled (Settings → Accelerator → GPU)
- [ ] Internet enabled (Settings → Internet → ON)
- [ ] Dependencies installed (`einops`, `lpips`, etc.)
- [ ] Training code copied to notebook
- [ ] ONE dataset method configured (USE_* = True)
- [ ] All other methods disabled (USE_* = False)
- [ ] Required credentials/paths filled in

---

## 🎓 After Training: Use Your Model

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(model_size='base', img_size=256)
checkpoint = torch.load('best_psnr.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Restore an image
img = Image.open('degraded_manuscript.jpg')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    restored = model(input_tensor)

# Save result
output_img = transforms.ToPILImage()(restored.squeeze(0).cpu())
output_img.save('restored_manuscript.jpg')
print("✓ Image restored successfully!")
```

---

## 🚨 Common Issues & Solutions

### Issue: "Found 0 images"
**Solution:** 
- Verify dataset downloaded correctly
- Check folder structure has `train/` with images
- List contents: `!ls /kaggle/working/dataset/train/`

### Issue: "CUDA out of memory"
**Solution:**
```python
BATCH_SIZE = 8        # Reduce batch size
MODEL_SIZE = 'small'  # Use smaller model
```

### Issue: "Invalid API key" (Roboflow)
**Solution:**
- Copy entire API key (no spaces)
- Get fresh key from Roboflow dashboard
- Check quotation marks are correct

### Issue: "Session timeout"
**Solution:**
- Reduce NUM_EPOCHS (Kaggle has 12-hour limit)
- Use smaller model
- Train in multiple sessions

---

## 📊 Expected Training Times

| Setup | Images | Epochs | Model | Time |
|-------|--------|--------|-------|------|
| Quick test | 50 | 10 | tiny | ~15 min |
| Small dataset | 500 | 50 | small | ~2 hrs |
| Medium dataset | 1000 | 100 | base | ~8 hrs |
| Large dataset | 2000 | 100 | base | ~15 hrs |

**Note:** Times for Kaggle's free GPU (T4)

---

## 🎯 Recommended Workflow

### First Time Users:
1. Use `USE_SAMPLE_DATASET = True`
2. Set `NUM_EPOCHS = 10`
3. Use `MODEL_SIZE = 'tiny'`
4. Verify everything works (~15 minutes)

### For Real Training:
1. Choose your dataset method
2. Use `MODEL_SIZE = 'base'`
3. Set `NUM_EPOCHS = 100`
4. Let it train (several hours)

### For Production:
1. Use Roboflow or curated dataset
2. Split train/val properly
3. Use `MODEL_SIZE = 'base'` or `'large'`
4. Train for full 100 epochs
5. Monitor PSNR/SSIM metrics

---

## 💡 Pro Tips

1. **Start Small:** Test with sample data first
2. **Monitor Metrics:** PSNR > 25 and SSIM > 0.8 are good
3. **Use Validation:** Include val set if possible
4. **Save Checkpoints:** They save every 5 epochs automatically
5. **Check GPU:** Make sure "Using device: cuda" appears
6. **Enable Internet:** Required for downloading datasets
7. **Time Management:** Plan for Kaggle's 12-hour limit

---

## 🌟 What Makes This Better?

### Old Way:
1. Manually upload dataset to Kaggle
2. Wait for upload
3. Add dataset via UI
4. Configure paths
5. Hope it works

### New Way:
1. Set `USE_ROBOFLOW = True`
2. Add API key
3. Run notebook
4. Done! ✨

**No manual uploads, no path confusion, no waiting!**

---

## 🚀 Ready to Start!

You now have everything you need:
- ✅ Complete training script with built-in dataset downloaders
- ✅ 6 different ways to get your dataset
- ✅ Comprehensive documentation
- ✅ Example configurations
- ✅ Troubleshooting guide

### Quick Start Command:
```
1. Create Kaggle notebook
2. Enable GPU + Internet
3. !pip install einops lpips roboflow gdown kaggle -q
4. Copy kaggle_training_notebook.py content
5. Run main() - That's it! (Already configured with your Roboflow dataset)
6. Wait for training to complete
7. Download trained model (best_psnr.pth)
```

**No configuration needed!** Your Roboflow credentials are already hardcoded.

**Let's train some models! 🎉**

---

## 📞 Need More Help?

Check these files in order:
1. **KAGGLE_QUICK_START.md** - Quick reference
2. **KAGGLE_DATASET_DOWNLOAD_EXAMPLES.md** - Dataset setup examples
3. **KAGGLE_TRAINING_GUIDE.md** - Full detailed guide
4. **This file** - Complete workflow

All files are in your GitHub repository:
https://github.com/Bagesh-Tallolli/historic-manuscript-restoration

**Happy Training! 🚀**

