# 🎯 Kaggle Training - Step-by-Step Visual Guide

## 📸 Complete Screenshot Guide for Kaggle Setup

---

## STEP 1: Create New Kaggle Notebook

### 1.1 Go to Kaggle
```
Open browser → Navigate to: https://www.kaggle.com
```

### 1.2 Create Notebook
```
Click "Create" (top right) → Select "New Notebook"
```

### 1.3 Import Your Notebook
```
File → Import Notebook
Browse → Select: kaggle_training_notebook.ipynb
Click "Upload"
```

**✅ Expected Result**: Notebook opens with all cells visible

---

## STEP 2: Configure GPU Settings

### 2.1 Open Settings Panel
```
Look at right sidebar → Find "Settings" section
```

### 2.2 Enable GPU
```
Settings → Accelerator → Select "GPU T4 x2"
(or any available GPU option)
```

### 2.3 Save Settings
```
Settings will auto-save
✓ You should see "GPU" badge at top of notebook
```

**✅ Expected Result**: GPU indicator shows enabled

---

## STEP 3: Verify Configuration (Optional)

### 3.1 Check GPU Cell (Cell 2)
```
Run Cell 2 to verify:

✓ PyTorch version: 2.x.x
✓ CUDA available: True
✓ GPU: Tesla T4 (or similar)
```

### 3.2 Check Roboflow Config (Cell 3)
```
Verify hardcoded values:

ROBOFLOW_API_KEY = "EBJvHlgSWLyW1Ir6ctkH"
ROBOFLOW_WORKSPACE = "neeew"
ROBOFLOW_PROJECT = "yoyoyo-mptyx-ijqfp"
ROBOFLOW_VERSION = 1

✓ These are ALREADY SET - no changes needed!
```

**✅ Expected Result**: All values match above

---

## STEP 4: Run Training

### 4.1 Run All Cells
```
Option A: Click "Run All" at top
Option B: Press Ctrl+/ (Windows) or Cmd+/ (Mac)
```

### 4.2 Monitor Progress

#### Cell 3: Dataset Download (1-5 minutes)
```
Expected output:
======================================================================
📥 DOWNLOADING DATASET FROM ROBOFLOW
======================================================================
Workspace: neeew
Project: yoyoyo-mptyx-ijqfp
Version: 1
Download location: /kaggle/working/dataset

[Progress bars...]

✅ Dataset downloaded successfully!
📁 Location: /kaggle/working/dataset
  ✓ train: XXX files
  ✓ valid: XXX files
```

#### Cell 10: Dataset Loading (30 seconds)
```
Expected output:
======================================================================
LOADING DATASETS
======================================================================
Found XXX images in train mode
Found XXX images in val mode

✓ Datasets loaded successfully
  Train batches: XX
  Train images: XXX
```

#### Cell 11: Model Creation (10 seconds)
```
Expected output:
======================================================================
CREATING MODEL
======================================================================
Architecture: ViT-BASE
Total parameters: 86,415,363
Trainable parameters: 86,415,363
Model size: ~329.54 MB (FP32)
```

#### Cell 12: Training Loop (2-5 hours)
```
Expected output (per epoch):

Epoch 1 [Train]: 100%|██████████| 50/50 [02:45<00:00]
Epoch 1 [Val]:   100%|██████████| 10/10 [00:30<00:00]

Epoch 1/100
  Train: loss: 0.1234 | psnr: 25.67 | ssim: 0.82 | lpips: 0.15
  Val:   loss: 0.1123 | psnr: 26.45 | ssim: 0.84 | lpips: 0.13
  ✓ New best PSNR: 26.45

Epoch 2 [Train]: 100%|██████████| 50/50 [02:45<00:00]
...
```

**Progress Indicators to Watch:**
- ✅ Loss decreasing (0.15 → 0.10 → 0.08...)
- ✅ PSNR increasing (25 → 28 → 31...)
- ✅ SSIM increasing (0.80 → 0.85 → 0.90...)
- ✅ "New best PSNR" messages

---

## STEP 5: Monitor Training Progress

### 5.1 Time Estimates
```
Typical training timeline (100 epochs on GPU T4):

Epoch 1-10:   ~30 minutes  (model learning basic features)
Epoch 11-50:  ~2 hours     (model improving quality)
Epoch 51-100: ~2.5 hours   (fine-tuning)

Total: ~5 hours
```

### 5.2 Early Stopping Criteria
```
You can stop early if:

✓ Validation PSNR > 30 dB (good quality)
✓ Validation PSNR > 32 dB (excellent quality)
✓ Validation PSNR plateaus for 20+ epochs
```

### 5.3 Checkpoint Saving
```
Automatic checkpoints saved at:

Epoch 10:  epoch_10.pth
Epoch 20:  epoch_20.pth
...
Epoch 100: epoch_100.pth + final.pth

Best model: best_psnr.pth (automatically updated)
```

---

## STEP 6: Training Complete

### 6.1 Success Message
```
======================================================================
✅ TRAINING COMPLETED SUCCESSFULLY!
======================================================================
📁 Checkpoints saved to: /kaggle/working/checkpoints/
🏆 Best model: /kaggle/working/checkpoints/best_psnr.pth
📊 Best validation PSNR: 32.45
======================================================================
```

### 6.2 Visualization (Cell 13)
```
Sample results will be displayed showing:

[ Degraded ] [ Restored ] [ Ground Truth ]
[ Degraded ] [ Restored ] [ Ground Truth ]
[ Degraded ] [ Restored ] [ Ground Truth ]
[ Degraded ] [ Restored ] [ Ground Truth ]

✓ Results saved to: /kaggle/working/results_visualization.png
```

---

## STEP 7: Download Trained Model

### 7.1 Using Download Links (Cell 14)
```
Cell 14 output will show:

📥 Download your trained models:

Best model (highest PSNR):
[Download: best_psnr.pth]

Final model:
[Download: final.pth]

Visualization:
[Download: results_visualization.png]

→ Click the blue download links
```

### 7.2 Manual Download (Alternative)
```
Right sidebar → Click "Output" tab
You'll see folder structure:

checkpoints/
  ├── best_psnr.pth ← DOWNLOAD THIS
  ├── final.pth
  ├── epoch_10.pth
  └── ...

→ Click ⋮ menu next to file → Download
```

### 7.3 Download All Files
```
Right sidebar → Output tab → Top right "Download All"
→ Downloads zip file with all checkpoints
```

---

## STEP 8: Verify Downloaded Files

### 8.1 Check File Sizes
```
Expected file sizes:

best_psnr.pth:     ~330 MB (base model)
final.pth:         ~330 MB
results_visualization.png: ~500 KB

✓ If files are much smaller, download may have failed
```

### 8.2 Test Loading Model Locally
```python
import torch

# Should load without errors
checkpoint = torch.load('best_psnr.pth', map_location='cpu')

# Check contents
print(checkpoint.keys())
# Expected: ['epoch', 'model_state_dict', 'optimizer_state_dict', ...]

# Check epoch
print(f"Trained for {checkpoint['epoch']} epochs")
print(f"Best PSNR: {checkpoint['best_val_psnr']:.2f}")
```

---

## 📊 Understanding Training Output

### Normal Training Pattern
```
Epoch 1:   Loss=0.15, PSNR=25.3, SSIM=0.81
Epoch 10:  Loss=0.12, PSNR=27.5, SSIM=0.84  ✓ Improving
Epoch 20:  Loss=0.10, PSNR=29.2, SSIM=0.87  ✓ Improving
Epoch 50:  Loss=0.08, PSNR=31.5, SSIM=0.91  ✓ Improving
Epoch 100: Loss=0.07, PSNR=32.8, SSIM=0.93  ✓ Excellent!
```

### Warning Signs
```
❌ Loss increasing after epoch 10
   → Check learning rate or batch size

❌ PSNR stuck at < 20 dB
   → Model not learning, check dataset

❌ Validation much worse than training
   → Overfitting, reduce model size or add regularization

❌ All metrics = 0 or NaN
   → Error in data loading or model, restart notebook
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Dataset Download Fails
```
Error: "Failed to download from Roboflow"

Solutions:
1. Check Cell 3 output for specific error
2. Verify API key is correct
3. Check Roboflow project is public or you have access
4. Try running Cell 3 alone to isolate issue
```

### Issue 2: Out of Memory (OOM)
```
Error: "CUDA out of memory"

Solutions (try in order):
1. Cell 9: BATCH_SIZE = 8 (instead of 16)
2. Cell 9: MODEL_SIZE = 'small' (instead of 'base')
3. Cell 9: IMG_SIZE = 128 (instead of 256)
4. Restart notebook and try again
```

### Issue 3: No GPU Available
```
Warning: "GPU quota exceeded"

Solutions:
1. Check Kaggle quota: Settings → Account → GPU Quota
   (Free tier: 30 hours/week)
2. Wait for quota to reset (Saturdays)
3. Use CPU (very slow but works)
   Cell 2 will show: CUDA available: False
```

### Issue 4: Training Very Slow
```
Problem: Training taking 10+ hours

Solutions:
1. Verify GPU is enabled (check Cell 2 output)
2. Cell 9: Reduce NUM_EPOCHS to 50
3. Cell 9: Use MODEL_SIZE = 'small'
4. Check if GPU is being used (nvidia-smi in code cell)
```

### Issue 5: Download Links Not Working
```
Problem: Cell 14 doesn't show download links

Solutions:
1. Use manual download (Right sidebar → Output)
2. Add new cell at end:
   from IPython.display import FileLink
   display(FileLink('/kaggle/working/checkpoints/best_psnr.pth'))
3. Download entire output folder
```

---

## ✅ Completion Checklist

After training completes, verify:

- [ ] Training finished without errors
- [ ] Cell 12 shows "✅ TRAINING COMPLETED"
- [ ] Best PSNR > 28 dB (good quality)
- [ ] Downloaded `best_psnr.pth` (~330 MB)
- [ ] Downloaded `results_visualization.png`
- [ ] Tested loading checkpoint locally
- [ ] Ready to use model for inference!

---

## 🎯 Next Steps After Training

### 1. Use Model for Inference
```python
# See inference.py in your project
python inference.py --model checkpoints/best_psnr.pth \
                    --input degraded_manuscript.jpg \
                    --output restored_manuscript.jpg
```

### 2. Fine-tune on Your Own Data
```python
# Upload your dataset to Kaggle
# Modify Cell 3 to point to your dataset
# Run training again
```

### 3. Try Different Configurations
```python
# Experiment with:
- Different model sizes
- Different degradation types
- Different training epochs
- Different image resolutions
```

---

## 📞 Support Resources

| Issue Type | Resource |
|------------|----------|
| Kaggle Platform | https://www.kaggle.com/discussions |
| Roboflow Dataset | https://docs.roboflow.com |
| PyTorch Errors | https://pytorch.org/docs |
| General Help | KAGGLE_NOTEBOOK_GUIDE.md |

---

## 🎉 Congratulations!

You've successfully:
✅ Set up Kaggle training environment
✅ Configured automatic dataset download
✅ Trained a Vision Transformer model
✅ Downloaded your trained model
✅ Ready for manuscript restoration!

**Your model is now ready to restore historic manuscripts! 🏛️**

---

*Last Updated: November 2025*
*For the latest version, check: KAGGLE_NOTEBOOK_GUIDE.md*

