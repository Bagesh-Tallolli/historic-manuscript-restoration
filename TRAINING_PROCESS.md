# 📖 Training Process for Sanskrit Manuscript Restoration

## Overview

For your dataset of **593 Sanskrit manuscript images** (415 train, 119 val, 59 test), we use a **Vision Transformer (ViT) based image restoration** approach with synthetic degradation.

---

## 🎯 Training Strategy

### The Core Concept: Self-Supervised Learning

Since your dataset contains clean manuscript images (not degraded ones), we use a **synthetic degradation** approach:

```
Clean Image (from dataset)
    ↓
Apply Synthetic Degradation (noise, blur, contrast reduction)
    ↓
Degraded Image (input to model)
    ↓
ViT Restoration Model
    ↓
Restored Image (output)
    ↓
Compare with Original Clean Image (loss calculation)
```

This allows us to train on clean images by artificially degrading them, then learning to restore them back to the original.

---

## 📊 Complete Training Pipeline

### Phase 1: Data Preparation

**1. Image Loading**
- Read images from `data/raw/train/` and `data/raw/val/`
- Original size: Variable (e.g., 1463×2316, 978×1411)
- Format: RGB JPG

**2. Preprocessing**
```python
# For each image:
1. Resize to training size (default: 256×256 or 512×512)
2. Normalize to [0, 1] range
3. Convert to tensor
```

**3. Synthetic Degradation** (Applied on-the-fly during training)
```python
Degradation types applied randomly:
- Gaussian noise (σ = 0.01-0.05)
- Gaussian blur (kernel size 3-7)
- JPEG compression artifacts (quality 50-90)
- Contrast reduction (factor 0.7-0.9)
- Brightness variation (±0.1)
```

**4. Data Augmentation** (Optional, helps prevent overfitting)
```python
- Random horizontal flip (50% probability)
- Random vertical flip (50% probability)
- Random rotation (±15 degrees)
- Random crop and resize
```

---

### Phase 2: Model Architecture

**Vision Transformer (ViT) for Image Restoration**

```
Input Image (degraded)
    ↓
Patch Embedding (split into 16×16 patches)
    ↓
Positional Encoding (add position information)
    ↓
Transformer Encoder (12 layers)
    ├── Multi-Head Self-Attention (learns global dependencies)
    ├── Feed-Forward Network (processes features)
    └── Layer Normalization + Residual Connections
    ↓
Decoder (reconstruct image)
    ├── Upsampling layers
    ├── Convolution layers
    └── Skip connections from encoder
    ↓
Output Image (restored)
```

**Model Specifications:**
- **Patch size:** 16×16 pixels
- **Embedding dimension:** 768 (base model)
- **Transformer layers:** 12
- **Attention heads:** 12
- **Parameters:** ~86M (base model)

---

### Phase 3: Training Process

**1. Initialization**
```python
- Model: ViT Restorer (randomly initialized or pretrained)
- Optimizer: AdamW (adaptive learning rate)
- Learning rate: 1e-4 (initial)
- Weight decay: 0.01 (regularization)
- Batch size: 16 (adjustable based on GPU memory)
```

**2. Loss Function (Combined Loss)**
```python
Total Loss = α × L1_Loss + β × Perceptual_Loss + γ × SSIM_Loss

Where:
- L1 Loss: Pixel-wise absolute difference
  L1 = mean(|restored - clean|)
  
- Perceptual Loss: Feature-based similarity (using VGG network)
  Perceptual = mean(|VGG(restored) - VGG(clean)|)
  
- SSIM Loss: Structural similarity
  SSIM = 1 - SSIM(restored, clean)
  
Default weights: α=1.0, β=0.1, γ=0.1
```

**3. Training Loop (Per Epoch)**

```python
For each batch in training data:
    1. Load clean images
    2. Apply synthetic degradation → get degraded images
    3. Forward pass: degraded → model → restored
    4. Calculate combined loss: compare restored with clean
    5. Backward pass: compute gradients
    6. Gradient clipping (prevent exploding gradients)
    7. Optimizer step: update model weights
    8. Learning rate scheduler step
    9. Calculate metrics (PSNR, SSIM) every 10 batches
    10. Update progress bar
```

**4. Validation Loop (After Each Epoch)**

```python
For each batch in validation data:
    1. Load clean images
    2. Apply synthetic degradation
    3. Forward pass (no gradient computation)
    4. Calculate loss and metrics
    5. Track best performance
```

**5. Learning Rate Schedule**

```python
Cosine Annealing Schedule:
- Start: 1e-4
- Min: 1e-6
- Gradually decreases in cosine curve
- Helps model converge smoothly
```

---

### Phase 4: Monitoring & Evaluation

**Metrics Tracked:**

1. **Loss Metrics**
   - Training loss (per batch, per epoch)
   - Validation loss (per epoch)

2. **Quality Metrics**
   - **PSNR** (Peak Signal-to-Noise Ratio)
     - Higher is better
     - Target: >25 dB (good), >30 dB (excellent)
   
   - **SSIM** (Structural Similarity Index)
     - Range: 0 to 1
     - Higher is better
     - Target: >0.85 (good), >0.90 (excellent)
   
   - **MSE** (Mean Squared Error)
     - Lower is better

3. **Training Metrics**
   - Learning rate (per step)
   - Gradient norm (for stability monitoring)
   - Time per epoch
   - GPU memory usage

---

### Phase 5: Checkpointing & Model Saving

**Automatic Saving:**

1. **Best model by PSNR** → `checkpoints/best_psnr.pth`
2. **Best model by loss** → `checkpoints/best_loss.pth`
3. **Periodic checkpoints** → `checkpoints/epoch_N.pth` (every 5 epochs)
4. **Final model** → `checkpoints/final.pth`
5. **Training history** → `checkpoints/training_history.json`

**Checkpoint Contents:**
```python
{
    'epoch': current_epoch,
    'model_state_dict': model weights,
    'optimizer_state_dict': optimizer state,
    'scheduler_state_dict': LR scheduler state,
    'best_val_loss': best validation loss seen,
    'best_val_psnr': best PSNR achieved,
    'train_history': all training metrics,
    'val_history': all validation metrics
}
```

---

## 🔄 Complete Training Workflow

### Step-by-Step Process

```
START
  ↓
[1] Load Dataset
  - Read 415 training images
  - Read 119 validation images
  - Create data loaders (batch size 16)
  ↓
[2] Initialize Model
  - Create ViT Restorer
  - Move to GPU/CPU
  - Count parameters (~86M)
  ↓
[3] Setup Training
  - Create optimizer (AdamW)
  - Create loss function (Combined Loss)
  - Create LR scheduler (Cosine Annealing)
  - Create metrics trackers
  ↓
[4] Training Loop (100 epochs)
  │
  ├─> For Epoch 1 to 100:
  │     │
  │     ├─> Training Phase:
  │     │     - Process 26 batches (415 images / 16 per batch)
  │     │     - For each batch:
  │     │         1. Apply synthetic degradation
  │     │         2. Forward pass through model
  │     │         3. Calculate loss
  │     │         4. Backpropagate
  │     │         5. Update weights
  │     │         6. Track metrics
  │     │     - Time: ~2-4 minutes per epoch (CPU)
  │     │
  │     ├─> Validation Phase:
  │     │     - Process 8 batches (119 images / 16 per batch)
  │     │     - No gradient computation (faster)
  │     │     - Calculate metrics (PSNR, SSIM, Loss)
  │     │     - Time: ~30-60 seconds per epoch (CPU)
  │     │
  │     ├─> Checkpoint Saving:
  │     │     - If best PSNR → save best_psnr.pth
  │     │     - If best loss → save best_loss.pth
  │     │     - Every 5 epochs → save epoch_N.pth
  │     │
  │     └─> Logging:
  │           - Print epoch summary
  │           - Log to TensorBoard/Wandb
  │           - Save training history
  │
  ↓
[5] Training Complete
  - Save final model
  - Generate training plots
  - Export metrics
  ↓
END
```

---

## 📊 Expected Training Timeline

### For Your Dataset (593 images, 415 train, 119 val)

**On CPU:**
- **Per epoch:** 2-4 minutes (train) + 0.5-1 minute (val) = ~3-5 minutes
- **50 epochs:** ~2.5-4 hours
- **100 epochs:** ~5-8 hours
- **200 epochs:** ~10-16 hours

**On GPU (if available):**
- **Per epoch:** 20-40 seconds (train) + 10-20 seconds (val) = ~30-60 seconds
- **50 epochs:** ~25-50 minutes
- **100 epochs:** ~50-100 minutes
- **200 epochs:** ~1.5-3 hours

---

## 🎯 Training Configuration Options

### Quick Test (Verify Everything Works)
```bash
python3 train.py \
    --train_dir data/raw/train \
    --val_dir data/raw/val \
    --epochs 5 \
    --batch_size 8 \
    --img_size 256
```
**Time:** ~15-20 minutes (CPU)  
**Purpose:** Ensure no errors, see initial results

### Standard Training (Good Results)
```bash
python3 train.py \
    --train_dir data/raw/train \
    --val_dir data/raw/val \
    --epochs 100 \
    --batch_size 16 \
    --img_size 256 \
    --lr 1e-4
```
**Time:** ~5-8 hours (CPU), ~1.5 hours (GPU)  
**Purpose:** Production-quality model

### Advanced Training (Best Results)
```bash
python3 train.py \
    --train_dir data/raw/train \
    --val_dir data/raw/val \
    --epochs 200 \
    --batch_size 8 \
    --img_size 512 \
    --lr 1e-4 \
    --use_wandb
```
**Time:** ~16-24 hours (CPU), ~3-4 hours (GPU)  
**Purpose:** Highest quality restoration

---

## 📈 What Happens During Training

### Epoch 1-10: Initial Learning
- Loss decreases rapidly
- Model learns basic structure
- PSNR: 15-20 dB
- SSIM: 0.6-0.7

### Epoch 10-50: Primary Learning
- Model learns fine details
- Loss decreases steadily
- PSNR: 20-25 dB
- SSIM: 0.75-0.85

### Epoch 50-100: Fine-Tuning
- Refinement of restoration quality
- Slower improvement
- PSNR: 25-28 dB
- SSIM: 0.85-0.90

### Epoch 100-200: Convergence
- Model converges to optimal weights
- Minimal improvement
- PSNR: 28-30 dB
- SSIM: 0.90-0.92

---

## 🔍 Key Technical Details

### 1. Why Vision Transformer?
- **Global context:** ViT processes entire image through self-attention
- **Long-range dependencies:** Better for document structure
- **Patch-based:** Efficient for high-resolution images
- **State-of-art:** Better than CNNs for restoration tasks

### 2. Why Synthetic Degradation?
- **No paired data needed:** Don't need degraded-clean pairs
- **Flexible:** Can simulate various degradation types
- **Realistic:** Mimics real document degradation
- **Data augmentation:** Increases effective dataset size

### 3. Why Combined Loss?
- **L1 Loss:** Ensures pixel-level accuracy
- **Perceptual Loss:** Preserves visual quality
- **SSIM Loss:** Maintains structural similarity
- **Together:** Best of all worlds

### 4. Why AdamW Optimizer?
- **Adaptive learning rates:** Different rates for different parameters
- **Weight decay:** Prevents overfitting
- **Momentum:** Smooth convergence
- **Proven:** Best for transformer models

---

## 🎛️ Hyperparameter Details

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Image Size** | 256×256 | Balance between quality and speed |
| **Batch Size** | 16 | Fit in GPU memory, stable gradients |
| **Learning Rate** | 1e-4 | Not too fast, not too slow |
| **Weight Decay** | 0.01 | Regularization to prevent overfitting |
| **Gradient Clip** | 1.0 | Prevent exploding gradients |
| **Warmup Epochs** | 0 | Start training immediately |
| **LR Schedule** | Cosine | Smooth decay for better convergence |

---

## 📊 Output After Training

### 1. Model Checkpoints
```
checkpoints/
├── best_psnr.pth         # Best performing model
├── best_loss.pth         # Lowest loss model
├── epoch_5.pth           # Periodic saves
├── epoch_10.pth
├── ...
├── final.pth             # Final epoch model
└── training_history.json # All metrics
```

### 2. Training Metrics
```json
{
  "train": [
    {"epoch": 1, "loss": 0.245, "psnr": 18.3, "ssim": 0.65},
    {"epoch": 2, "loss": 0.198, "psnr": 21.2, "ssim": 0.72},
    ...
  ],
  "val": [
    {"epoch": 1, "loss": 0.256, "psnr": 17.8, "ssim": 0.63},
    {"epoch": 2, "loss": 0.205, "psnr": 20.5, "ssim": 0.70},
    ...
  ]
}
```

### 3. Visualizations (if using TensorBoard/Wandb)
- Loss curves (train vs val)
- PSNR progression
- SSIM progression
- Learning rate schedule
- Sample restoration examples

---

## 🚀 Start Training Now

**Recommended command for your dataset:**

```bash
cd /home/bagesh/EL-project
source venv/bin/activate

# Start with a test run
python3 train.py \
    --train_dir data/raw/train \
    --val_dir data/raw/val \
    --epochs 10 \
    --batch_size 16 \
    --img_size 256

# If test run succeeds, run full training
python3 train.py \
    --train_dir data/raw/train \
    --val_dir data/raw/val \
    --epochs 100 \
    --batch_size 16 \
    --img_size 256 \
    --checkpoint_dir models/checkpoints
```

---

## 🎯 Success Criteria

**Your model is ready when:**
- ✅ Validation PSNR > 25 dB
- ✅ Validation SSIM > 0.85
- ✅ Validation loss stops decreasing
- ✅ Visual quality of restored images is good
- ✅ No overfitting (train/val losses similar)

---

This training process is specifically designed for your 593 Sanskrit manuscript images and will produce a high-quality restoration model! 🕉️📜

