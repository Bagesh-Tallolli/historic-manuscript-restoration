# KAGGLE TRAINING - QUICK REFERENCE

## 🎯 What You Need for Kaggle Training

### 1. Single Python File
**File:** `kaggle_training_notebook.py`
- Contains EVERYTHING needed for training
- Includes: Model, Dataset, Training Loop, Metrics, **Dataset Downloaders**
- ✅ **Pre-configured with your Roboflow dataset!**
- No external files required!

### 2. Your Dataset (Already Set Up!)
- ✅ **Roboflow dataset pre-configured**
- API Key: Already included
- Workspace: `neeew`
- Project: `yoyoyo-mptyx-ijqfp`
- **Just run the notebook - no setup needed!**

### 3. Kaggle Setup
- Enable GPU (Settings → Accelerator → GPU)
- Enable Internet (Settings → Internet → ON)
- Install packages: `einops`, `lpips`, `roboflow`

---

## 📝 Step-by-Step Instructions (SIMPLIFIED!)

### STEP 1: Create Kaggle Notebook
```
1. Go to kaggle.com
2. Click "Create" → "New Notebook"
3. Settings → Accelerator → GPU T4
4. Settings → Internet → ON (IMPORTANT!)
```

### STEP 2: Install Dependencies
```python
# Cell 1: Install packages
!pip install einops lpips roboflow gdown kaggle -q
```

### STEP 3: Copy Training Code
```python
# Cell 2: Copy ENTIRE content from kaggle_training_notebook.py
# Get it from: https://github.com/Bagesh-Tallolli/historic-manuscript-restoration
```

### STEP 4: Run Training (That's It!)
```python
# Cell 3: Run the training
if __name__ == "__main__":
    main()
```

**✨ NO CONFIGURATION NEEDED!** The notebook is already set up with your Roboflow dataset.
The dataset will download automatically when you run it!

---

## ⚙️ Configuration Options

### Essential Settings (in main() function)
```python
TRAIN_DIR = '/kaggle/input/your-dataset/train'  # Your dataset path
IMG_SIZE = 256                                   # Image size (keep at 256)
BATCH_SIZE = 16                                  # Reduce if out of memory
NUM_EPOCHS = 100                                 # Training epochs
MODEL_SIZE = 'base'                             # Model complexity
```

### Model Size Options
```
'tiny'  → Fast, lower quality   (Use if: limited time/data)
'small' → Balanced              (Use if: moderate dataset)
'base'  → Best quality          (Use if: good dataset, 1000+ images) ✓ RECOMMENDED
'large' → Highest quality       (Use if: large dataset, powerful GPU)
```

---

## 📊 What to Expect

### Training Output
```
Found 1000 images in train mode
Creating base model...
Total parameters: 86,342,400
Starting training for 100 epochs
Device: cuda

Epoch 1/100 [Train]: 100%|████████| 62/62 [01:45<00:00, loss: 0.1234]
Epoch 1/100 [Val]:   100%|████████| 16/16 [00:18<00:00, loss: 0.1156]

Epoch 1/100
  Train: loss: 0.1234 | psnr: 25.67 | ssim: 0.8512
  Val:   loss: 0.1156 | psnr: 26.45 | ssim: 0.8678
  ✓ New best PSNR: 26.45

... (continues for 100 epochs)

✓ Training complete!
Checkpoints saved to: /kaggle/working/checkpoints/
```

### Output Files
```
/kaggle/working/checkpoints/
├── best_psnr.pth      ← Best model (download this!)
├── epoch_5.pth        ← Checkpoint at epoch 5
├── epoch_10.pth       ← Checkpoint at epoch 10
├── ...
└── final.pth          ← Final model
```

---

## 📥 Download Your Trained Model

### Method 1: Kaggle UI
```
1. Wait for training to complete
2. Click "Output" tab (right side)
3. Open "checkpoints" folder
4. Click on "best_psnr.pth"
5. Click download icon
```

### Method 2: Add Download Link in Code
```python
# Add this cell after training
from IPython.display import FileLink
display(FileLink('/kaggle/working/checkpoints/best_psnr.pth'))
```

---

## 🐛 Troubleshooting

### Problem: "CUDA out of memory"
**Solution:**
```python
BATCH_SIZE = 8        # Reduce from 16 to 8
# OR
MODEL_SIZE = 'small'  # Use smaller model
```

### Problem: "Found 0 images"
**Solution:**
- Verify dataset path: `/kaggle/input/YOUR-DATASET-NAME/train`
- Check images are .jpg, .png, .tif, or .bmp
- Make sure dataset is added to notebook

### Problem: Training is too slow
**Solution:**
- Use 'small' or 'tiny' model
- Reduce NUM_EPOCHS to 50
- Make sure GPU is enabled (not CPU)

### Problem: Session timeout (>12 hours)
**Solution:**
- Reduce NUM_EPOCHS
- Use smaller model
- Train in stages (save checkpoint, resume later)

---

## 📈 Training Time Estimates

| Setup | Time (approx) |
|-------|---------------|
| 500 images, 'small' model, 50 epochs | ~2 hours |
| 1000 images, 'base' model, 100 epochs | ~8 hours |
| 2000 images, 'large' model, 100 epochs | ~15 hours |

**Note:** Kaggle has 12-hour limit for free tier!

---

## ✅ Pre-Flight Checklist

Before clicking "Run All":

- [ ] GPU is enabled (check top-right corner shows "GPU T4" or "GPU P100")
- [ ] Dependencies installed (einops, lpips)
- [ ] Dataset added to notebook
- [ ] TRAIN_DIR path is correct
- [ ] Notebook has internet access ON
- [ ] You have enough time (check estimates above)

---

## 🎓 After Training

### Using Your Model for Inference

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

# Load and restore an image
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
```

---

## 📚 Files You Need

### For Training on Kaggle:
1. `kaggle_training_notebook.py` (copy contents to Kaggle)
2. Your dataset (uploaded to Kaggle Datasets)

### For Reference:
1. `KAGGLE_TRAINING_GUIDE.md` (detailed guide)
2. This file (quick reference)

---

## 💡 Pro Tips

1. **Start Small**: Test with 50-100 images first to ensure everything works
2. **Monitor Progress**: Check metrics every few epochs
3. **Save Often**: Checkpoints save automatically every 5 epochs
4. **Good PSNR**: Aim for PSNR > 25 and SSIM > 0.8
5. **Validation**: If possible, use 10-20% of data for validation
6. **GPU Check**: Verify "Using device: cuda" appears in output

---

## 🚀 Ready to Train?

You now have everything you need! Here's the minimal workflow:

```
1. Open Kaggle → New Notebook → Enable GPU
2. !pip install einops lpips -q
3. Copy kaggle_training_notebook.py content
4. Update TRAIN_DIR path
5. Run the notebook
6. Wait for training (grab coffee ☕)
7. Download best_psnr.pth
8. Use for inference!
```

**Good luck! 🎉**

---

## 📞 Quick Reference Commands

### Install Dependencies
```bash
!pip install einops lpips -q
```

### Check GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### List Dataset Files
```bash
!ls -la /kaggle/input/
!ls -la /kaggle/input/your-dataset/train/ | head -20
```

### Check Training Progress
```bash
!ls -lh /kaggle/working/checkpoints/
```

### Download Model
```python
from IPython.display import FileLink
FileLink('/kaggle/working/checkpoints/best_psnr.pth')
```

---

**That's it! You're ready to train on Kaggle! 🎯**

