# ============================================================================
# KAGGLE TRAINING NOTEBOOK - Complete Restoration Pipeline
# ============================================================================
# Upload this to Kaggle as a notebook
# Enable GPU: Settings → GPU T4 x2
# Run All Cells
# ============================================================================

# CELL 1: Install Dependencies
# %%
get_ipython().run_cell_magic('capture', '', '!pip install einops lpips roboflow opencv-python-headless scikit-image matplotlib -q')

# CELL 2: Imports
# %%
import torch
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

# CELL 3: Download Dataset
# %%
from roboflow import Roboflow

API_KEY = "EBJvHlgSWLyW1Ir6ctkH"
WORKSPACE = "neeew"
PROJECT = "yoyoyo-mptyx-ijqfp"
VERSION = 1
DATASET_LOC = "/kaggle/working/dataset"

print("📥 Downloading dataset from Roboflow...")
rf = Roboflow(api_key=API_KEY)
proj = rf.workspace(WORKSPACE).project(PROJECT)
dataset = proj.version(VERSION).download("folder", location=DATASET_LOC)

TRAIN_DIR = f"{DATASET_LOC}/train"
VAL_DIR = f"{DATASET_LOC}/valid" if os.path.exists(f"{DATASET_LOC}/valid") else None

print(f"✅ Dataset downloaded to: {DATASET_LOC}")
print(f"Train: {len(list(Path(TRAIN_DIR).glob('**/*.*')))} files")
if VAL_DIR:
    print(f"Valid: {len(list(Path(VAL_DIR).glob('**/*.*')))} files")

# CELL 4: Model Architecture with Skip Connections
# %%
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_c=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
    def forward(self, x):
        return self.proj(x)

class Attention(nn.Module):
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
    def __init__(self, dim=768, hidden=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))

class Block(nn.Module):
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
    """
    Vision Transformer for Image Restoration
    With skip connections for better detail preservation
    """
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

        # Skip connection fusion (preserves fine details from input)
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

        # Encode: patches + positional embedding
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

        # Apply skip connection (helps preserve input details and textures)
        if self.use_skip:
            x = torch.cat([x, input_img], dim=1)  # Concatenate restored + input
            x = self.skip_fusion(x)  # Fuse them intelligently

        return x

print("✓ Model architecture defined (with skip connections for better restoration)")

# CELL 5: Enhanced Loss Function
# %%
class CombinedLoss(nn.Module):
    """
    Combined loss for better restoration:
    - L1 loss for pixel-level accuracy
    - Perceptual loss for visual quality
    """
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

        # Perceptual loss using LPIPS
        try:
            self.lpips_fn = lpips.LPIPS(net='vgg').eval()
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
            self.use_perceptual = True
            print("✓ Using L1 + Perceptual loss")
        except:
            self.use_perceptual = False
            print("⚠️ Perceptual loss not available, using L1 only")

    def forward(self, pred, target, device='cuda'):
        # L1 loss
        l1_loss = F.l1_loss(pred, target)

        # Perceptual loss
        if self.use_perceptual and self.perceptual_weight > 0:
            try:
                self.lpips_fn = self.lpips_fn.to(device)
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

print("✓ Loss function defined")

# CELL 6: Dataset with Better Degradation
# %%
class ManuscriptDataset(Dataset):
    def __init__(self, data_dir, img_size=256, mode='train', augment=True):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.mode = mode
        self.augment = augment and mode == 'train'

        exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        self.images = []
        for ext in exts:
            self.images.extend(list(self.data_dir.glob(f'**/*{ext}')))
            self.images.extend(list(self.data_dir.glob(f'**/*{ext.upper()}')))
        self.images = sorted(set(self.images))
        print(f"{mode}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def _degrade(self, img):
        """Apply realistic manuscript degradation"""
        img = img.astype(np.float32) / 255.0

        # Additive noise (simulates scanning noise, paper texture)
        if random.random() > 0.2:
            noise_level = random.uniform(0.02, 0.10)
            img += np.random.normal(0, noise_level, img.shape)

        # Gaussian blur (simulates focus issues, age blur)
        if random.random() > 0.2:
            kernel_size = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # Contrast/brightness degradation (faded ink)
        if random.random() > 0.3:
            contrast = random.uniform(0.5, 0.9)
            brightness = random.uniform(0.05, 0.20)
            img = contrast * img + brightness

        # Aging tint (yellowing paper)
        if random.random() > 0.3:
            tint = np.array([1.0, random.uniform(0.90, 0.98), random.uniform(0.75, 0.90)])
            img *= tint

        # JPEG compression artifacts (simulates poor digitization)
        if random.random() > 0.5:
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            quality = random.randint(60, 90)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode('.jpg', img_uint8, encode_param)
            img = cv2.imdecode(enc, 1).astype(np.float32) / 255.0

        return np.clip(img, 0, 1)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.array(Image.open(img_path).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Clean target
        clean = img.astype(np.float32) / 255.0

        # Degraded input
        degraded = self._degrade(img.copy())

        # Data augmentation
        if self.augment:
            if random.random() > 0.5:
                degraded = np.fliplr(degraded).copy()
                clean = np.fliplr(clean).copy()
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                degraded = np.rot90(degraded, k=angle//90).copy()
                clean = np.rot90(clean, k=angle//90).copy()

        # Resize
        degraded = cv2.resize(degraded, (self.img_size, self.img_size))
        clean = cv2.resize(clean, (self.img_size, self.img_size))

        # To tensor
        degraded = torch.from_numpy(degraded).permute(2, 0, 1).float()
        clean = torch.from_numpy(clean).permute(2, 0, 1).float()

        return {'degraded': degraded, 'clean': clean}

print("✓ Enhanced dataset with realistic degradation")

# CELL 7: Configuration
# %%
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

# CELL 8: Create Datasets
# %%
train_dataset = ManuscriptDataset(TRAIN_DIR, IMG_SIZE, 'train', True)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

val_loader = None
if VAL_DIR:
    val_dataset = ManuscriptDataset(VAL_DIR, IMG_SIZE, 'val', False)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}")
if val_loader:
    print(f"Val batches: {len(val_loader)}")

# CELL 9: Create Model, Loss, Optimizer
# %%
model = ViTRestorer(
    img_size=IMG_SIZE,
    embed_dim=768,
    depth=12,
    heads=12,
    use_skip=True  # Enable skip connections!
).to(DEVICE)

criterion = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1)
criterion = criterion.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=1e-6)

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/test_results', exist_ok=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model: {total_params:,} parameters")
print(f"✓ Skip connections: ENABLED (for better detail preservation)")

# Continue in next message...

