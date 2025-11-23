# 🎯 Training Flow - Quick Summary

## The 6-Step Training Process

This is what happens **every single time** a batch is processed during training:

---

### 1️⃣ Clean Image (from dataset)
```
Source: data/raw/train/train_0077.jpg
Size: 1463×2316 → Resized to 256×256
Format: RGB image
```
**This is your original manuscript image**

---

### 2️⃣ Apply Synthetic Degradation
```
Add: Noise + Blur + Fading + Stains
Simulates: Aged manuscript damage
Result: Degraded version of the image
```
**This creates a "damaged" version artificially**

---

### 3️⃣ Degraded Image (input to model)
```
Format: Tensor [16, 3, 256, 256]
Values: Normalized to [0, 1]
```
**This degraded image goes into the model as INPUT**

---

### 4️⃣ ViT Restoration Model
```
Process:
  • Split into 16×16 patches (256 patches total)
  • Transformer processes all patches
  • Learns to restore the image
  • Reconstructs clean version

Model: 86,433,813 parameters
```
**The AI model tries to restore the damaged image**

---

### 5️⃣ Restored Image (output)
```
Format: Tensor [16, 3, 256, 256]
Result: Model's prediction of clean image
```
**This is what the model THINKS the clean image should look like**

---

### 6️⃣ Compare with Original Clean Image (loss)
```
Calculate:
  • How different is restored from original clean?
  • L1 Loss = pixel differences
  • PSNR = quality metric
  • SSIM = structural similarity

Then:
  • Calculate gradients
  • Update model weights
  • Model learns to do better next time
```
**Model learns by comparing its output to the original clean image**

---

## 🔄 The Complete Loop

```
FOR each epoch (1 to 100):
    FOR each batch (1 to 26):
        
        1. Load 16 clean images
           ↓
        2. Degrade them synthetically
           ↓
        3. Feed degraded to model
           ↓
        4. Model restores them
           ↓
        5. Get restored output
           ↓
        6. Compare to original clean
           ↓
        7. Calculate loss
           ↓
        8. Update model weights
           ↓
        REPEAT
```

**This happens 2,600 times (26 batches × 100 epochs)**

---

## 🎓 Key Insight

The beauty of this approach:

1. You only have **clean images** (no degraded-clean pairs needed)
2. We **synthetically degrade** them during training
3. Model learns: `degraded → clean` restoration
4. At inference: Works on **real degraded manuscripts**

---

## 📊 Training Numbers

| What | How Many |
|------|----------|
| Clean images | 415 |
| Degradations per image | Random (different each epoch) |
| Batches per epoch | 26 |
| Epochs | 100 |
| Total training iterations | 2,600 |
| Model parameters updated | 86,433,813 |
| Time (CPU) | ~5-8 hours |

---

## 🖼️ See It In Action

Run this to visualize the flow:
```bash
source venv/bin/activate
python3 visualize_training_flow.py
```

This creates 4 images showing:
1. Original clean image
2. Degraded version
3. Model's restoration
4. Comparison of all three

Location: `output/training_flow/`

---

## 🚀 Start Training

```bash
source venv/bin/activate
python3 train.py \
    --train_dir data/raw/train \
    --val_dir data/raw/val \
    --epochs 100 \
    --batch_size 16
```

---

## 📚 More Details

- **TRAINING_PROCESS.md** - Complete detailed explanation
- **TRAINING_FLOW_DIAGRAM.md** - Visual diagrams
- **show_training_flow.sh** - Quick reference script

---

## ✅ Summary

**Question:** How does the model learn without degraded images?

**Answer:** 
1. Take clean image
2. Degrade it artificially (noise, blur, etc.)
3. Train model to restore it back to clean
4. Repeat thousands of times
5. Model learns the restoration pattern
6. Apply to real degraded manuscripts!

**That's the magic! 🪄**

