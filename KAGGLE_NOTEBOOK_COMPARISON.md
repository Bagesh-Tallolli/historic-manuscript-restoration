# 📊 Kaggle Notebook Comparison

## Files Available

### 1. `kaggle_training_notebook.ipynb` (OLD - 5.3 KB)
❌ **Missing Features:**
- No skip connections in model
- Basic degradation only (4 techniques)
- No perceptual loss (L1 only)
- No rotation augmentation
- Incomplete training loop

### 2. `kaggle_training_complete.ipynb` (NEW - 37 KB) ✅ RECOMMENDED
✅ **Complete Features:**
- ✅ Skip connections for better detail preservation
- ✅ Enhanced degradation (6 techniques including JPEG artifacts)
- ✅ Perceptual loss (LPIPS) + L1 loss
- ✅ Full data augmentation (flip + rotation)
- ✅ Complete training loop with validation
- ✅ Automatic model saving (best + final + periodic)
- ✅ Sample visualization
- ✅ Training history tracking
- ✅ Comprehensive documentation

## 🔑 Key Differences

### Paired Training Implementation

**Both notebooks implement paired training correctly!**

```python
# How pairing works:
def __getitem__(self, idx):
    img = load_image(path)                    # Load original
    
    clean = normalize(img)                     # Target (what we want)
    degraded = self._degrade(img.copy())       # Input (what model receives)
    
    return {'degraded': degraded, 'clean': clean}
```

### Model Architecture

**OLD (No Skip Connections):**
```python
class ViTRestorer(nn.Module):
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        # Reconstruct image
        return x  # ❌ Details may be lost
```

**NEW (With Skip Connections):**
```python
class ViTRestorer(nn.Module):
    def forward(self, x):
        input_img = x  # Save input
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        # Reconstruct + fuse with input
        if self.use_skip:
            x = torch.cat([x, input_img], dim=1)
            x = self.skip_fusion(x)  # ✅ Preserves fine details
        return x
```

### Degradation Techniques

**OLD (4 techniques):**
- Gaussian noise
- Gaussian blur
- Contrast reduction
- Aging tint

**NEW (6 techniques):**
- Gaussian noise (wider range)
- Gaussian blur
- Contrast/brightness reduction
- Aging tint (more realistic)
- Salt & pepper noise (stains/spots)
- JPEG compression artifacts ✅ NEW!

### Loss Function

**OLD:**
- L1 loss only

**NEW:**
- L1 loss (pixel accuracy)
- LPIPS perceptual loss (visual quality) ✅ NEW!

## 📥 What to Upload to Kaggle

**Use: `kaggle_training_complete.ipynb`** (37 KB)

This has everything you need:
1. Automatic dataset download (Roboflow credentials hardcoded)
2. Complete model with skip connections
3. Enhanced synthetic degradation
4. Perceptual loss for better quality
5. Full training loop with validation
6. Automatic model saving
7. Sample visualizations
8. Ready-to-download models

## 🎯 Training Process Explained

```
For each image in dataset:
┌─────────────────────────────────────────────────┐
│ 1. Load clean image from Roboflow              │
│    → This is your TARGET (ground truth)        │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│ 2. Apply synthetic degradation                  │
│    → Noise, blur, fading, stains, etc.         │
│    → This creates the INPUT (degraded version)  │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│ 3. Training step:                               │
│    degraded → Model → restored                  │
│    Loss = compare(restored, clean)              │
│    Update model to reduce loss                  │
└─────────────────────────────────────────────────┘
```

## 📁 File Structure Created Automatically

The notebook creates this structure on Kaggle:

```
/kaggle/working/
├── dataset/                    # Downloaded from Roboflow
│   ├── train/                  # Training images
│   └── valid/                  # Validation images
├── checkpoints/                # Model checkpoints
│   ├── best_psnr.pth          # Best model (recommended)
│   ├── final.pth              # Final epoch
│   ├── epoch_20.pth           # Periodic saves
│   ├── epoch_40.pth
│   └── ...
├── test_results/              # Sample visualizations
│   ├── sample_1_comparison.jpg
│   ├── sample_2_comparison.jpg
│   └── ...
├── restoration_examples.png   # Visual summary
└── training_history.json      # Training metrics
```

## 🚀 After Training on Kaggle

### Download These Files:
1. **best_psnr.pth** (recommended) - Best model based on validation PSNR
2. **final.pth** - Final epoch model
3. **desti.pth** - Complete checkpoint with optimizer state
4. **restoration_examples.png** - Visual results

### Place in Your Project:
```bash
mkdir -p /home/bagesh/EL-project/checkpoints/kaggle
cp ~/Downloads/best_psnr.pth /home/bagesh/EL-project/checkpoints/kaggle/
cp ~/Downloads/final.pth /home/bagesh/EL-project/checkpoints/kaggle/
cp ~/Downloads/desti.pth /home/bagesh/EL-project/checkpoints/kaggle/
```

### Test Locally:
```bash
cd /home/bagesh/EL-project
source activate_venv.sh

# Test on single image
python main.py \
    --image_path data/raw/test/test_0001.jpg \
    --restoration_model checkpoints/kaggle/best_psnr.pth

# Run web UI
streamlit run app.py
```

---

✅ **Recommendation**: Delete old `kaggle_training_notebook.ipynb` and use `kaggle_training_complete.ipynb` for Kaggle training!

