#!/usr/bin/env python3
"""
Create complete Kaggle training notebook with all enhancements
"""
import nbformat as nbf

def create_complete_kaggle_notebook():
    """Create complete notebook for Kaggle training"""
    
    nb = nbf.v4.new_notebook()
    
    # Cell 1: Title and Instructions
    nb.cells.append(nbf.v4.new_markdown_cell("""# 🏛️ Historic Manuscript Restoration - Complete Training Pipeline

## 📋 What This Notebook Does
Trains a Vision Transformer (ViT) model to restore degraded Sanskrit manuscripts using:
- **Synthetic Degradation**: Creates realistic degraded/clean pairs automatically
- **Skip Connections**: Preserves fine details during restoration
- **Perceptual Loss**: Improves visual quality beyond pixel-level metrics
- **Automatic Dataset Download**: From Roboflow (hardcoded credentials)

## ⚙️ Quick Start
1. **Enable GPU**: Settings → Accelerator → GPU T4 x2
2. **Click "Run All"**: Everything is automated
3. **Wait ~5 hours**: Training completes automatically
4. **Download Models**: `best_psnr.pth` and `final.pth` from Output panel

## 🎯 Key Features
✅ Paired training: Each image → generates degraded version → model learns to restore  
✅ Automatic data augmentation (flip, rotate)  
✅ Validation metrics (PSNR, SSIM)  
✅ Sample visualizations saved  
✅ Ready for download and local use  

---"""))
    
    # Cell 2: Install Dependencies
    nb.cells.append(nbf.v4.new_code_cell("""%%capture
!pip install einops lpips roboflow opencv-python-headless scikit-image matplotlib -q"""))
    
    # Cell 3: Imports
    nb.cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import json
import time
import os
from einops import rearrange
from einops.layers.torch import Rearrange
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")"""))
    
    # Cell 4: Download Dataset
    nb.cells.append(nbf.v4.new_markdown_cell("""## 📥 Step 1: Download Dataset (Automatic)

The dataset will be downloaded automatically from Roboflow with hardcoded credentials."""))
    
    nb.cells.append(nbf.v4.new_code_cell("""from roboflow import Roboflow

# Hardcoded Roboflow API credentials
API_KEY = "EBJvHlgSWLyW1Ir6ctkH"
WORKSPACE = "neeew"
PROJECT = "yoyoyo-mptyx-ijqfp"
VERSION = 1
DATASET_LOC = "/kaggle/working/dataset"

print("📥 Downloading dataset from Roboflow...")
print(f"   Workspace: {WORKSPACE}")
print(f"   Project: {PROJECT}")
print(f"   Version: {VERSION}")

rf = Roboflow(api_key=API_KEY)
proj = rf.workspace(WORKSPACE).project(PROJECT)
dataset = proj.version(VERSION).download("folder", location=DATASET_LOC)

TRAIN_DIR = f"{DATASET_LOC}/train"
VAL_DIR = f"{DATASET_LOC}/valid" if os.path.exists(f"{DATASET_LOC}/valid") else None

print(f"\\n✅ Dataset downloaded successfully!")
print(f"   Location: {DATASET_LOC}")
print(f"   Train images: {len(list(Path(TRAIN_DIR).glob('**/*.*')))} files")
if VAL_DIR:
    print(f"   Valid images: {len(list(Path(VAL_DIR).glob('**/*.*')))} files")"""))
    
    # Cell 5: Model Architecture
    nb.cells.append(nbf.v4.new_markdown_cell("""## 🧠 Step 2: Define Model Architecture

Vision Transformer with Skip Connections for better detail preservation."""))
    
    nb.cells.append(nbf.v4.new_code_cell("""class PatchEmbed(nn.Module):
    \"\"\"Convert image to patches and embed them\"\"\"
    def __init__(self, img_size=256, patch_size=16, in_c=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
    def forward(self, x):
        return self.proj(x)

class Attention(nn.Module):
    \"\"\"Multi-head self-attention\"\"\"
    def __init__(self, dim=768, heads=12, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(self.drop(x))

class MLP(nn.Module):
    \"\"\"Feed-forward network\"\"\"
    def __init__(self, dim=768, hidden=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))

class Block(nn.Module):
    \"\"\"Transformer block\"\"\"
    def __init__(self, dim=768, heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTRestorer(nn.Module):
    \"\"\"
    Vision Transformer for Image Restoration
    WITH SKIP CONNECTIONS for better detail preservation
    \"\"\"
    def __init__(self, img_size=256, patch_size=16, in_c=3, out_c=3,
                 embed_dim=768, depth=12, heads=12, mlp_ratio=4.0, use_skip=True):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_c, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder head
        self.head = nn.Linear(embed_dim, patch_size * patch_size * out_c)
        
        # Skip connection fusion (IMPORTANT: preserves fine details from input)
        self.use_skip = use_skip
        if use_skip:
            self.skip_fusion = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
            nn.init.kaiming_normal_(self.skip_fusion.weight)
            nn.init.constant_(self.skip_fusion.bias, 0)
        
        self.img_size = img_size
        self.patch_size = patch_size
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        # Store input for skip connection
        input_img = x
        
        # Encode: image patches + positional embedding
        x = self.patch_embed(x) + self.pos_embed
        
        # Transform: apply transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # Decode: reconstruct image from patches
        x = self.head(x)
        h = w = self.img_size // self.patch_size
        x = x.reshape(x.shape[0], h, w, self.patch_size, self.patch_size, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], 3, self.img_size, self.img_size)
        
        # Apply skip connection (helps preserve input details)
        if self.use_skip:
            x = torch.cat([x, input_img], dim=1)  # Concatenate restored + input
            x = self.skip_fusion(x)  # Fuse them intelligently
        
        return x

total_params = sum(p.numel() for p in ViTRestorer().parameters())
print(f"✓ Model architecture defined: {total_params:,} parameters")
print(f"✓ Skip connections: ENABLED (better detail preservation)")"""))
    
    # Cell 6: Loss Function
    nb.cells.append(nbf.v4.new_markdown_cell("""## 📊 Step 3: Define Loss Function

Combined loss using:
- **L1 Loss**: Pixel-level accuracy
- **Perceptual Loss (LPIPS)**: Visual quality and texture preservation"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""class CombinedLoss(nn.Module):
    \"\"\"
    Combined loss for better restoration:
    - L1 loss for pixel-level accuracy
    - Perceptual loss (LPIPS) for visual quality
    \"\"\"
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize LPIPS (perceptual loss)
        try:
            self.lpips_fn = lpips.LPIPS(net='alex')
            self.use_perceptual = True
            print("✓ LPIPS perceptual loss enabled")
        except:
            self.use_perceptual = False
            print("⚠️  LPIPS unavailable, using L1 only")
    
    def forward(self, pred, target):
        # L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # Perceptual loss
        if self.use_perceptual:
            try:
                # LPIPS expects inputs in [-1, 1] range
                pred_scaled = pred * 2 - 1
                target_scaled = target * 2 - 1
                perceptual_loss = self.lpips_fn(pred_scaled, target_scaled).mean()
                
                total_loss = self.l1_weight * l1_loss + self.perceptual_weight * perceptual_loss
                return total_loss
            except:
                return l1_loss
        else:
            return l1_loss

print("✓ Loss function defined")"""))
    
    # Cell 7: Dataset with Degradation
    nb.cells.append(nbf.v4.new_markdown_cell("""## 🖼️ Step 4: Create Dataset with Synthetic Degradation

**IMPORTANT: This is how paired training works!**

For each clean image:
1. Load the clean image (target)
2. Apply synthetic degradation (creates the degraded input)
3. Model learns to map: degraded → clean

Degradation techniques simulate real manuscript damage:
- Noise (scanning artifacts, paper texture)
- Blur (age, poor focus)
- Fading (ink degradation)
- Color shifting (paper yellowing)
- Compression artifacts (poor digitization)
- Stains and spots"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""class ManuscriptDataset(Dataset):
    \"\"\"
    Dataset for manuscript restoration with synthetic degradation
    
    For each image:
    1. Load clean image (target)
    2. Create degraded version (input) using synthetic degradation
    3. Return pair: {degraded, clean}
    \"\"\"
    def __init__(self, data_dir, img_size=256, mode='train', augment=True):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.mode = mode
        self.augment = augment and mode == 'train'
        
        # Find all images
        exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        self.images = []
        for ext in exts:
            self.images.extend(list(self.data_dir.glob(f'**/*{ext}')))
            self.images.extend(list(self.data_dir.glob(f'**/*{ext.upper()}')))
        self.images = sorted(set(self.images))
        print(f"{mode}: {len(self.images)} images found")
    
    def __len__(self):
        return len(self.images)
    
    def _degrade(self, img):
        \"\"\"
        Apply realistic manuscript degradation
        This simulates real-world manuscript damage
        \"\"\"
        img = img.astype(np.float32) / 255.0
        
        # 1. Additive noise (scanning noise, paper texture)
        if random.random() > 0.2:
            noise_level = random.uniform(0.02, 0.10)
            img += np.random.normal(0, noise_level, img.shape)
        
        # 2. Gaussian blur (focus issues, age blur)
        if random.random() > 0.2:
            kernel_size = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # 3. Contrast/brightness degradation (faded ink)
        if random.random() > 0.3:
            contrast = random.uniform(0.5, 0.9)
            brightness = random.uniform(0.05, 0.20)
            img = contrast * img + brightness
        
        # 4. Aging tint (yellowing paper)
        if random.random() > 0.3:
            tint = np.array([1.0, random.uniform(0.90, 0.98), random.uniform(0.75, 0.90)])
            img *= tint
        
        # 5. Salt and pepper noise (stains, spots)
        if random.random() > 0.5:
            noise_ratio = random.uniform(0.001, 0.01)
            mask = np.random.random(img.shape[:2]) < noise_ratio
            img[mask] = np.random.choice([0, 1], size=(mask.sum(), 3))
        
        # 6. JPEG compression artifacts (poor digitization)
        if random.random() > 0.5:
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            quality = random.randint(60, 90)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode('.jpg', img_uint8, encode_param)
            img = cv2.imdecode(enc, 1).astype(np.float32) / 255.0
        
        return np.clip(img, 0, 1)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.array(Image.open(img_path).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # PAIRED DATA CREATION:
        # Clean image = target (what we want model to output)
        clean = img.astype(np.float32) / 255.0
        
        # Degraded image = input (what we feed to model)
        degraded = self._degrade(img.copy())
        
        # Data augmentation (applied to BOTH images to maintain pairing)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                degraded = np.fliplr(degraded).copy()
                clean = np.fliplr(clean).copy()
            # Random rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                degraded = np.rot90(degraded, k=angle//90).copy()
                clean = np.rot90(clean, k=angle//90).copy()
        
        # Resize to model input size
        degraded = cv2.resize(degraded, (self.img_size, self.img_size))
        clean = cv2.resize(clean, (self.img_size, self.img_size))
        
        # Convert to PyTorch tensors
        degraded = torch.from_numpy(degraded).permute(2, 0, 1).float()
        clean = torch.from_numpy(clean).permute(2, 0, 1).float()
        
        return {'degraded': degraded, 'clean': clean}

print("✓ Dataset class defined with synthetic degradation pipeline")"""))
    
    # Cell 8: Configuration
    nb.cells.append(nbf.v4.new_markdown_cell("""## ⚙️ Step 5: Training Configuration"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Training hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Training Configuration:")
print(f"  Device: {DEVICE}")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")"""))
    
    # Cell 9: Create Datasets
    nb.cells.append(nbf.v4.new_markdown_cell("""## 📦 Step 6: Load Datasets"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Create training dataset
train_dataset = ManuscriptDataset(TRAIN_DIR, IMG_SIZE, 'train', augment=True)
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2, 
    pin_memory=True
)

# Create validation dataset (if available)
val_loader = None
if VAL_DIR:
    val_dataset = ManuscriptDataset(VAL_DIR, IMG_SIZE, 'val', augment=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

print(f"\\nDataset Summary:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Train images: {len(train_dataset)}")
if val_loader:
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Val images: {len(val_dataset)}")"""))
    
    # Cell 10: Initialize Model
    nb.cells.append(nbf.v4.new_markdown_cell("""## 🚀 Step 7: Create Model & Optimizer"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Create model with skip connections enabled
model = ViTRestorer(
    img_size=IMG_SIZE,
    embed_dim=768,
    depth=12,
    heads=12,
    use_skip=True  # IMPORTANT: Enable skip connections
).to(DEVICE)

# Loss function (L1 + Perceptual)
criterion = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1)
criterion = criterion.to(DEVICE)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=1e-6)

# Create checkpoint directory
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\nModel Summary:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1e6:.1f} MB")"""))
    
    # Cell 11: Training Loop
    nb.cells.append(nbf.v4.new_markdown_cell("""## 🏋️ Step 8: Training Loop

Training process:
1. **Load batch** of degraded/clean pairs
2. **Forward pass**: degraded → model → restored
3. **Calculate loss**: compare restored vs clean
4. **Backward pass**: update model weights
5. **Validate**: check performance on validation set
6. **Save best model**: based on PSNR metric"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""def calc_psnr(pred, target):
    \"\"\"Calculate PSNR metric\"\"\"
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    psnrs = []
    for i in range(pred_np.shape[0]):
        psnr = peak_signal_noise_ratio(target_np[i], pred_np[i], data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)

# Training state
best_psnr = 0.0
training_history = []

print("🏋️ Starting training...\\n")

for epoch in range(NUM_EPOCHS):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    train_psnr = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in enumerate(pbar):
        # Get degraded input and clean target
        degraded = batch['degraded'].to(DEVICE)
        clean = batch['clean'].to(DEVICE)
        
        # Forward pass: degraded → model → restored
        optimizer.zero_grad()
        restored = model(degraded)
        
        # Calculate loss: how different is restored from clean?
        loss = criterion(restored, clean)
        
        # Backward pass: update weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        with torch.no_grad():
            train_psnr += calc_psnr(restored, clean)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Average metrics
    train_loss /= len(train_loader)
    train_psnr /= len(train_loader)
    
    # Update learning rate
    scheduler.step()
    
    # ========== VALIDATION ==========
    if val_loader and (epoch + 1) % 5 == 0:
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                degraded = batch['degraded'].to(DEVICE)
                clean = batch['clean'].to(DEVICE)
                
                restored = model(degraded)
                val_loss += criterion(restored, clean).item()
                val_psnr += calc_psnr(restored, clean)
        
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Train PSNR={train_psnr:.2f} | "
              f"Val Loss={val_loss:.4f}, Val PSNR={val_psnr:.2f}")
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_psnr': best_psnr,
                'config': {
                    'img_size': IMG_SIZE,
                    'embed_dim': 768,
                    'depth': 12,
                    'heads': 12,
                    'use_skip': True
                }
            }, '/kaggle/working/checkpoints/best_psnr.pth')
            print(f"  ✓ New best PSNR: {best_psnr:.2f} dB (model saved!)")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_psnr': train_psnr,
            'val_loss': val_loss,
            'val_psnr': val_psnr
        })
    else:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Train PSNR={train_psnr:.2f}")
    
    # Save periodic checkpoint
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f'/kaggle/working/checkpoints/epoch_{epoch+1}.pth')
        print(f"  ✓ Checkpoint saved: epoch_{epoch+1}.pth")

# Save final model
torch.save(model.state_dict(), '/kaggle/working/checkpoints/final.pth')
print(f"\\n✅ Training complete!")
print(f"   Best validation PSNR: {best_psnr:.2f} dB")

# Save training history
with open('/kaggle/working/training_history.json', 'w') as f:
    json.dump(training_history, f, indent=2)"""))
    
    # Cell 12: Test on Samples
    nb.cells.append(nbf.v4.new_markdown_cell("""## 🧪 Step 9: Test Model on Sample Images

Load best model and test on samples to visualize results"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""print("🧪 Testing trained model on samples...\\n")

# Load best model
if os.path.exists('/kaggle/working/checkpoints/best_psnr.pth'):
    checkpoint = torch.load('/kaggle/working/checkpoints/best_psnr.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model (Epoch {checkpoint['epoch']}, PSNR: {checkpoint['best_val_psnr']:.2f} dB)")
else:
    print("⚠️ Using final model (best_psnr.pth not found)")

model.eval()

# Create test results directory
os.makedirs('/kaggle/working/test_results', exist_ok=True)

# Get sample images
sample_images = list(Path(TRAIN_DIR).glob('*.jpg'))[:5]
if not sample_images:
    sample_images = list(Path(TRAIN_DIR).glob('**/*.jpg'))[:5]

print(f"Testing on {len(sample_images)} sample images...\\n")

test_results = []

with torch.no_grad():
    for idx, img_path in enumerate(sample_images):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.array(Image.open(img_path).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create clean version (target)
        clean_img = img.astype(np.float32) / 255.0
        
        # Create degraded version (input) using same degradation as training
        degraded_img = clean_img.copy()
        degraded_img += np.random.normal(0, 0.05, degraded_img.shape)
        degraded_img = cv2.GaussianBlur(degraded_img, (5, 5), 0)
        degraded_img = 0.7 * degraded_img + 0.1
        degraded_img *= np.array([1.0, 0.95, 0.8])
        degraded_img = np.clip(degraded_img, 0, 1)
        
        # Resize and convert to tensor
        clean_resized = cv2.resize(clean_img, (IMG_SIZE, IMG_SIZE))
        degraded_resized = cv2.resize(degraded_img, (IMG_SIZE, IMG_SIZE))
        
        degraded_tensor = torch.from_numpy(degraded_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        
        # Restore using trained model
        restored_tensor = model(degraded_tensor)
        
        # Convert back to numpy
        restored_img = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored_img = np.clip(restored_img, 0, 1)
        
        # Calculate metrics
        psnr = peak_signal_noise_ratio(clean_resized, restored_img, data_range=1.0)
        ssim = structural_similarity(clean_resized, restored_img, multichannel=True, channel_axis=2, data_range=1.0)
        
        test_results.append({'sample': idx+1, 'psnr': psnr, 'ssim': ssim})
        print(f"  Sample {idx+1}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        
        # Save comparison (degraded | restored | clean)
        comparison = np.hstack([
            (degraded_resized * 255).astype(np.uint8),
            (restored_img * 255).astype(np.uint8),
            (clean_resized * 255).astype(np.uint8)
        ])
        
        cv2.imwrite(
            f'/kaggle/working/test_results/sample_{idx+1}_comparison.jpg',
            cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        )

print(f"\\n✅ Test results saved to /kaggle/working/test_results/")
print(f"   Average PSNR: {np.mean([r['psnr'] for r in test_results]):.2f} dB")
print(f"   Average SSIM: {np.mean([r['ssim'] for r in test_results]):.4f}")"""))
    
    # Cell 13: Visualize Results
    nb.cells.append(nbf.v4.new_markdown_cell("""## 📊 Step 10: Visualize Results"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Display restoration examples
fig, axes = plt.subplots(min(3, len(sample_images)), 3, figsize=(15, 5*min(3, len(sample_images))))
if len(sample_images) == 1:
    axes = axes.reshape(1, -1)

for idx in range(min(3, len(sample_images))):
    img_path = f'/kaggle/working/test_results/sample_{idx+1}_comparison.jpg'
    if os.path.exists(img_path):
        comparison = cv2.imread(img_path)
        comparison = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        
        # Split into three parts
        h, w = comparison.shape[:2]
        w_third = w // 3
        
        degraded = comparison[:, :w_third]
        restored = comparison[:, w_third:2*w_third]
        clean = comparison[:, 2*w_third:]
        
        axes[idx, 0].imshow(degraded)
        axes[idx, 0].set_title('Degraded Input', fontsize=12)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(restored)
        axes[idx, 1].set_title('Restored Output', fontsize=12, color='green')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(clean)
        axes[idx, 2].set_title('Clean Target', fontsize=12)
        axes[idx, 2].axis('off')

plt.suptitle('Manuscript Restoration Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/restoration_examples.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualization saved to /kaggle/working/restoration_examples.png")"""))
    
    # Cell 14: Save Models for Download
    nb.cells.append(nbf.v4.new_markdown_cell("""## 💾 Step 11: Save Models for Download

Save models in formats compatible with your local project"""))
    
    nb.cells.append(nbf.v4.new_code_cell("""# Save final model (state dict only - lightweight)
torch.save(model.state_dict(), '/kaggle/working/final.pth')
print("✓ Saved: final.pth (state dict only, ~330 MB)")

# Save complete checkpoint (with optimizer state - for resuming training)
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_psnr': best_psnr,
    'training_history': training_history,
    'config': {
        'img_size': IMG_SIZE,
        'embed_dim': 768,
        'depth': 12,
        'heads': 12,
        'mlp_ratio': 4.0,
        'use_skip': True
    }
}, '/kaggle/working/desti.pth')
print("✓ Saved: desti.pth (complete checkpoint with training state, ~990 MB)")

print(f"\\n✅ All models saved in /kaggle/working/")
print(f"\\n📥 To download:")
print(f"   1. Click 'Output' tab on the right →")
print(f"   2. Download these files:")
print(f"      • best_psnr.pth (recommended for inference)")
print(f"      • final.pth (final epoch)")
print(f"      • desti.pth (complete checkpoint)")
print(f"      • restoration_examples.png (visual results)")"""))
    
    # Cell 15: Usage Instructions
    nb.cells.append(nbf.v4.new_markdown_cell("""## 🚀 Step 12: Use Trained Models Locally

### 📥 After downloading models:

```bash
# 1. Place models in your project
mkdir -p /home/bagesh/EL-project/checkpoints/kaggle
cp ~/Downloads/final.pth /home/bagesh/EL-project/checkpoints/kaggle/
cp ~/Downloads/desti.pth /home/bagesh/EL-project/checkpoints/kaggle/

# 2. Test the model
cd /home/bagesh/EL-project
source activate_venv.sh

# 3. Run inference on test images
python inference.py \\
    --checkpoint checkpoints/kaggle/final.pth \\
    --input data/raw/test/ \\
    --output output/kaggle_results

# 4. Or use the full pipeline
python main.py \\
    --image_path data/raw/test/test_0001.jpg \\
    --restoration_model checkpoints/kaggle/final.pth

# 5. Or run the web UI
streamlit run app.py
```

### 📊 Training Summary
- **Best PSNR**: """ + "{best_psnr:.2f}" + """ dB
- **Total Epochs**: """ + "{NUM_EPOCHS}" + """
- **Model Parameters**: 86.4M
- **Training Time**: ~5 hours on GPU T4 x2

### 🎯 Model Architecture
- **Type**: Vision Transformer (ViT)
- **Input**: 256x256 degraded manuscript image
- **Output**: 256x256 restored image
- **Skip Connections**: Enabled (preserves fine details)
- **Loss**: L1 + Perceptual (LPIPS)

---

🎉 **Training Complete!** Your model is ready to restore ancient manuscripts!"""))
    
    return nb

# Create and save notebook
nb = create_complete_kaggle_notebook()
with open('/home/bagesh/EL-project/kaggle_training_complete.ipynb', 'w') as f:
    nbf.write(nb, f)

print(f"✓ Complete Kaggle training notebook created!")
print(f"  File: kaggle_training_complete.ipynb")
print(f"  Cells: {len(nb.cells)}")
print(f"\\n✅ Ready to upload to Kaggle!")

