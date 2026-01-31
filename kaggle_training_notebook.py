"""
KAGGLE TRAINING SCRIPT FOR HISTORIC MANUSCRIPT RESTORATION
===========================================================

This is a standalone script that contains all necessary code to train the
ViT-based manuscript restoration model on Kaggle.

INSTRUCTIONS FOR KAGGLE:
1. Create a new Kaggle Notebook
2. Copy this entire script into a cell
3. Upload your dataset or use Kaggle datasets
4. Enable GPU in Kaggle (Settings > Accelerator > GPU)
5. Run the notebook

Dataset Structure Expected:
/kaggle/input/your-dataset/
    ├── train/  (your training images)
    ├── val/    (optional validation images)
    └── test/   (optional test images)
"""

# ============================================================================
# SECTION 1: INSTALL DEPENDENCIES
# ============================================================================

# Run this in a Kaggle cell:

!pip install einops lpips roboflow kaggle gdown -q


# ============================================================================
# SECTION 2: IMPORTS
# ============================================================================

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
import zipfile
import urllib.request
from einops import rearrange
from einops.layers.torch import Rearrange
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ============================================================================
# SECTION 2A: DATASET DOWNLOAD UTILITIES
# ============================================================================

class DatasetDownloader:
    """Download datasets from various sources"""

    @staticmethod
    def download_from_roboflow(api_key, workspace, project, version, location="/kaggle/working/dataset"):
        """
        Download dataset from Roboflow

        Args:
            api_key: Your Roboflow API key
            workspace: Roboflow workspace name
            project: Project name
            version: Dataset version
            location: Where to save the dataset
        """
        try:
            from roboflow import Roboflow

            print("📥 Downloading dataset from Roboflow...")
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace).project(project)
            dataset = project.version(version).download("folder", location=location)

            print(f"✓ Dataset downloaded to: {location}")
            return location
        except Exception as e:
            print(f"❌ Error downloading from Roboflow: {e}")
            return None

    @staticmethod
    def download_from_kaggle_dataset(dataset_name, location="/kaggle/working/dataset"):
        """
        Download dataset from Kaggle Datasets

        Args:
            dataset_name: Kaggle dataset name (e.g., 'username/dataset-name')
            location: Where to save the dataset
        """
        try:
            import kaggle

            print(f"📥 Downloading dataset from Kaggle: {dataset_name}")
            os.makedirs(location, exist_ok=True)

            # Download using Kaggle API
            kaggle.api.dataset_download_files(dataset_name, path=location, unzip=True)

            print(f"✓ Dataset downloaded to: {location}")
            return location
        except Exception as e:
            print(f"❌ Error downloading from Kaggle: {e}")
            print("Make sure you have kaggle.json in ~/.kaggle/ or /root/.kaggle/")
            return None

    @staticmethod
    def download_from_url(url, location="/kaggle/working/dataset"):
        """
        Download dataset from direct URL (zip file)

        Args:
            url: Direct URL to zip file
            location: Where to save the dataset
        """
        try:
            print(f"📥 Downloading dataset from URL...")
            os.makedirs(location, exist_ok=True)

            zip_path = f"{location}/dataset.zip"

            # Download with progress bar
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\rDownloading: {percent}%", end='')

            urllib.request.urlretrieve(url, zip_path, reporthook)
            print("\n")

            # Extract
            print("📦 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(location)

            # Remove zip file
            os.remove(zip_path)

            print(f"✓ Dataset downloaded and extracted to: {location}")
            return location
        except Exception as e:
            print(f"❌ Error downloading from URL: {e}")
            return None

    @staticmethod
    def download_from_google_drive(file_id, location="/kaggle/working/dataset"):
        """
        Download dataset from Google Drive

        Args:
            file_id: Google Drive file ID
            location: Where to save the dataset
        """
        try:
            import gdown

            print(f"📥 Downloading dataset from Google Drive...")
            os.makedirs(location, exist_ok=True)

            zip_path = f"{location}/dataset.zip"
            url = f"https://drive.google.com/uc?id={file_id}"

            gdown.download(url, zip_path, quiet=False)

            # Extract
            print("📦 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(location)

            # Remove zip file
            os.remove(zip_path)

            print(f"✓ Dataset downloaded and extracted to: {location}")
            return location
        except Exception as e:
            print(f"❌ Error downloading from Google Drive: {e}")
            return None

    @staticmethod
    def use_sample_dataset(location="/kaggle/working/dataset"):
        """
        Create a small sample dataset for testing

        Args:
            location: Where to create the dataset
        """
        print("🎨 Creating sample dataset for testing...")

        train_dir = Path(location) / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        # Create 50 random sample images
        for i in range(50):
            # Create a random "manuscript-like" image
            img = np.random.randint(200, 255, (512, 512, 3), dtype=np.uint8)

            # Add some text-like patterns
            for _ in range(20):
                x = random.randint(50, 450)
                y = random.randint(50, 450)
                w = random.randint(10, 100)
                h = random.randint(5, 20)
                color = random.randint(0, 100)
                cv2.rectangle(img, (x, y), (x+w, y+h), (color, color, color), -1)

            # Save
            img_path = train_dir / f"sample_{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)

        print(f"✓ Created {50} sample images in: {location}")
        return str(location)

# ============================================================================
# SECTION 3: MODEL ARCHITECTURE
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them"""
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

    def forward(self, x):
        return self.proj(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        assert embed_dim % num_heads == 0
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchReconstruction(nn.Module):
    """Reconstruct image from patches"""
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, out_channels=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_side = img_size // patch_size
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.rearrange = Rearrange(
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.num_patches_side,
            w=self.num_patches_side,
            p1=patch_size,
            p2=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.rearrange(x)
        return x


class ViTRestorer(nn.Module):
    """Vision Transformer for Image Restoration"""
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        use_skip_connections=True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_skip_connections = use_skip_connections

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Patch reconstruction
        self.patch_recon = PatchReconstruction(img_size, patch_size, embed_dim, out_channels)

        # Skip connection fusion
        if use_skip_connections:
            self.skip_fusion = nn.Conv2d(in_channels + out_channels, out_channels, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_img = x
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.patch_recon(x)

        if self.use_skip_connections:
            x = torch.cat([x, input_img], dim=1)
            x = self.skip_fusion(x)

        return x


# ============================================================================
# SECTION 4: LOSS FUNCTIONS
# ============================================================================

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features

        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return F.mse_loss(pred_features, target_features)


class CombinedLoss(nn.Module):
    """Combined L1 + Perceptual Loss"""
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        return self.l1_weight * l1 + self.perceptual_weight * perceptual


# ============================================================================
# SECTION 5: DATASET LOADER
# ============================================================================

class ManuscriptDataset(Dataset):
    """Dataset for manuscript images with synthetic degradation"""

    def __init__(self, data_dir, img_size=256, mode='train', augment=True):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.mode = mode
        self.augment = augment and mode == 'train'
        self.image_paths = self._find_images()
        print(f"Found {len(self.image_paths)} images in {mode} mode")

    def _find_images(self):
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(self.data_dir.glob(f'**/*{ext}'))
            image_paths.extend(self.data_dir.glob(f'**/*{ext.upper()}'))
        return sorted(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        clean_img = self._load_image(img_path)
        degraded_img = self._degrade_image(clean_img.copy())

        if self.augment:
            degraded_img, clean_img = self._augment(degraded_img, clean_img)

        degraded_img = cv2.resize(degraded_img, (self.img_size, self.img_size))
        clean_img = cv2.resize(clean_img, (self.img_size, self.img_size))

        degraded_tensor = self._to_tensor(degraded_img)
        clean_tensor = self._to_tensor(clean_img)

        return {
            'degraded': degraded_tensor,
            'clean': clean_tensor,
            'path': str(img_path)
        }

    def _load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            img = np.array(Image.open(path).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _degrade_image(self, img):
        """Synthetically degrade image"""
        img = img.astype(np.float32) / 255.0

        # Gaussian noise
        if random.random() > 0.3:
            noise_sigma = random.uniform(0.01, 0.08)
            noise = np.random.normal(0, noise_sigma, img.shape)
            img = img + noise

        # Blur
        if random.random() > 0.3:
            kernel_size = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # Reduce contrast
        if random.random() > 0.4:
            alpha = random.uniform(0.6, 0.9)
            beta = random.uniform(0.05, 0.15)
            img = alpha * img + beta

        # Salt and pepper noise
        if random.random() > 0.5:
            noise_ratio = random.uniform(0.001, 0.01)
            mask = np.random.random(img.shape[:2]) < noise_ratio
            img[mask] = np.random.choice([0, 1], size=(mask.sum(), 3))

        # Yellow tint (aging)
        if random.random() > 0.4:
            yellow_tint = np.array([1.0, 0.95, 0.8])
            img = img * yellow_tint

        # Stains
        if random.random() > 0.6:
            num_stains = random.randint(1, 5)
            for _ in range(num_stains):
                center_x = random.randint(0, img.shape[1])
                center_y = random.randint(0, img.shape[0])
                radius = random.randint(10, 50)
                intensity = random.uniform(0.3, 0.7)
                y, x = np.ogrid[:img.shape[0], :img.shape[1]]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                img[mask] = img[mask] * intensity

        return np.clip(img, 0, 1)

    def _augment(self, degraded, clean):
        """Apply augmentation to both images"""
        # Random flip
        if random.random() > 0.5:
            degraded = np.fliplr(degraded).copy()
            clean = np.fliplr(clean).copy()

        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            degraded = np.rot90(degraded, k=angle//90).copy()
            clean = np.rot90(clean, k=angle//90).copy()

        return degraded, clean

    def _to_tensor(self, img):
        """Convert numpy image to tensor"""
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img


# ============================================================================
# SECTION 6: METRICS
# ============================================================================

class ImageMetrics:
    """Calculate image quality metrics"""

    def __init__(self, device='cuda'):
        self.device = device
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        except:
            self.lpips_fn = None

    def calculate_all(self, pred, target):
        """Calculate all metrics"""
        metrics = {}

        # Convert to numpy for PSNR and SSIM
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # PSNR
        psnr_values = []
        for i in range(pred_np.shape[0]):
            psnr = peak_signal_noise_ratio(target_np[i], pred_np[i], data_range=1.0)
            psnr_values.append(psnr)
        metrics['psnr'] = np.mean(psnr_values)

        # SSIM
        ssim_values = []
        for i in range(pred_np.shape[0]):
            ssim = structural_similarity(
                target_np[i].transpose(1, 2, 0),
                pred_np[i].transpose(1, 2, 0),
                data_range=1.0,
                channel_axis=2
            )
            ssim_values.append(ssim)
        metrics['ssim'] = np.mean(ssim_values)

        # LPIPS
        if self.lpips_fn is not None:
            with torch.no_grad():
                lpips_val = self.lpips_fn(pred, target)
            metrics['lpips'] = lpips_val.mean().item()

        return metrics


class RunningMetrics:
    """Track running average of metrics"""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value
            self.counts[key] += 1

    def get_averages(self):
        return {k: v / self.counts[k] for k, v in self.metrics.items()}


# ============================================================================
# SECTION 7: TRAINER
# ============================================================================

class Trainer:
    """Training manager"""

    def __init__(self, model, train_loader, val_loader=None, device='cuda',
                 checkpoint_dir='/kaggle/working/checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100, eta_min=1e-6
        )

        # Metrics
        self.metrics = ImageMetrics(device=device)

        # Training state
        self.current_epoch = 0
        self.best_val_psnr = 0.0
        self.train_history = []
        self.val_history = []

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        running_metrics = RunningMetrics()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            degraded = batch['degraded'].to(self.device)
            clean = batch['clean'].to(self.device)

            self.optimizer.zero_grad()
            restored = self.model(degraded)
            loss = self.criterion(restored, clean)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Calculate metrics periodically
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    metrics = self.metrics.calculate_all(restored, clean)
            else:
                metrics = {}

            metrics['loss'] = loss.item()
            running_metrics.update(metrics)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return running_metrics.get_averages()

    def validate(self):
        """Validate the model"""
        if self.val_loader is None:
            return None

        self.model.eval()
        running_metrics = RunningMetrics()

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")

        with torch.no_grad():
            for batch in pbar:
                degraded = batch['degraded'].to(self.device)
                clean = batch['clean'].to(self.device)

                restored = self.model(degraded)
                loss = self.criterion(restored, clean)

                metrics = self.metrics.calculate_all(restored, clean)
                metrics['loss'] = loss.item()
                running_metrics.update(metrics)
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return running_metrics.get_averages()

    def train(self, num_epochs, save_every=5):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)

                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train: {self._format_metrics(train_metrics)}")
                print(f"  Val:   {self._format_metrics(val_metrics)}")

                val_psnr = val_metrics.get('psnr', 0)
                if val_psnr > self.best_val_psnr:
                    self.best_val_psnr = val_psnr
                    self.save_checkpoint('best_psnr.pth')
                    print(f"  ✓ New best PSNR: {val_psnr:.4f}")

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pth')

        self.save_checkpoint('final.pth')
        print("\n✓ Training complete!")

    def save_checkpoint(self, filename):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_psnr': self.best_val_psnr,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        torch.save(checkpoint, checkpoint_path)

    def _format_metrics(self, metrics):
        return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])


# ============================================================================
# SECTION 8: MAIN TRAINING FUNCTION
# ============================================================================

def create_model(model_size='base', img_size=256):
    """Create model"""
    configs = {
        'tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
    }
    config = configs.get(model_size, configs['base'])
    return ViTRestorer(img_size=img_size, **config)


def main():
    """
    MAIN TRAINING FUNCTION - MODIFY THESE SETTINGS FOR YOUR KAGGLE SETUP
    """

    # ========== DATASET CONFIGURATION ==========
    # Choose ONE of the following methods to get your dataset:

    # METHOD 1: Download from Roboflow (RECOMMENDED for manuscript datasets)
    # ✅ PRECONFIGURED WITH YOUR ROBOFLOW DATASET
    USE_ROBOFLOW = True  # ← Already set up and ready to use!
    ROBOFLOW_CONFIG = {
        'api_key': 'EBJvHlgSWLyW1Ir6ctkH',  # Your Roboflow API key
        'workspace': 'neeew',                # Your workspace
        'project': 'yoyoyo-mptyx-ijqfp',    # Your manuscript project
        'version': 1                         # Dataset version
    }
    # Project URL: https://app.roboflow.com/neeew/yoyoyo-mptyx-ijqfp

    # METHOD 2: Download from Kaggle Dataset
    USE_KAGGLE_DATASET = False
    KAGGLE_DATASET_NAME = 'username/dataset-name'  # e.g., 'user123/sanskrit-manuscripts'

    # METHOD 3: Download from URL (direct link to zip file)
    USE_URL_DOWNLOAD = False
    DATASET_URL = 'https://example.com/dataset.zip'

    # METHOD 4: Download from Google Drive
    USE_GOOGLE_DRIVE = False
    GOOGLE_DRIVE_FILE_ID = 'your-file-id'  # From drive link

    # METHOD 5: Use pre-uploaded Kaggle input dataset
    USE_KAGGLE_INPUT = False  # Disabled (using Roboflow instead)
    TRAIN_DIR = '/kaggle/input/your-dataset/train'
    VAL_DIR = None

    # METHOD 6: Use sample dataset (for testing only)
    USE_SAMPLE_DATASET = False

    # ========== DOWNLOAD DATASET (if needed) ==========
    dataset_location = None

    if USE_ROBOFLOW:
        print("=" * 60)
        print("DOWNLOADING DATASET FROM ROBOFLOW")
        print("=" * 60)
        dataset_location = DatasetDownloader.download_from_roboflow(
            api_key=ROBOFLOW_CONFIG['api_key'],
            workspace=ROBOFLOW_CONFIG['workspace'],
            project=ROBOFLOW_CONFIG['project'],
            version=ROBOFLOW_CONFIG['version']
        )
        if dataset_location:
            TRAIN_DIR = f"{dataset_location}/train"
            VAL_DIR = f"{dataset_location}/valid" if os.path.exists(f"{dataset_location}/valid") else None

    elif USE_KAGGLE_DATASET:
        print("=" * 60)
        print("DOWNLOADING DATASET FROM KAGGLE")
        print("=" * 60)
        dataset_location = DatasetDownloader.download_from_kaggle_dataset(
            dataset_name=KAGGLE_DATASET_NAME
        )
        if dataset_location:
            TRAIN_DIR = f"{dataset_location}/train"
            VAL_DIR = f"{dataset_location}/val" if os.path.exists(f"{dataset_location}/val") else None

    elif USE_URL_DOWNLOAD:
        print("=" * 60)
        print("DOWNLOADING DATASET FROM URL")
        print("=" * 60)
        dataset_location = DatasetDownloader.download_from_url(url=DATASET_URL)
        if dataset_location:
            TRAIN_DIR = f"{dataset_location}/train"
            VAL_DIR = f"{dataset_location}/val" if os.path.exists(f"{dataset_location}/val") else None

    elif USE_GOOGLE_DRIVE:
        print("=" * 60)
        print("DOWNLOADING DATASET FROM GOOGLE DRIVE")
        print("=" * 60)
        dataset_location = DatasetDownloader.download_from_google_drive(
            file_id=GOOGLE_DRIVE_FILE_ID
        )
        if dataset_location:
            TRAIN_DIR = f"{dataset_location}/train"
            VAL_DIR = f"{dataset_location}/val" if os.path.exists(f"{dataset_location}/val") else None

    elif USE_SAMPLE_DATASET:
        print("=" * 60)
        print("CREATING SAMPLE DATASET FOR TESTING")
        print("=" * 60)
        dataset_location = DatasetDownloader.use_sample_dataset()
        TRAIN_DIR = f"{dataset_location}/train"
        VAL_DIR = None

    # ========== TRAINING CONFIGURATION ==========
    IMG_SIZE = 256
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    MODEL_SIZE = 'base'  # 'tiny', 'small', 'base', or 'large'
    NUM_WORKERS = 2

    # ========== SETUP ==========
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Val directory: {VAL_DIR}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Model size: {MODEL_SIZE}")

    # Create datasets
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)
    train_dataset = ManuscriptDataset(TRAIN_DIR, img_size=IMG_SIZE, mode='train', augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)

    val_loader = None
    if VAL_DIR and Path(VAL_DIR).exists():
        val_dataset = ManuscriptDataset(VAL_DIR, img_size=IMG_SIZE, mode='val', augment=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    print(f"Model architecture: ViT-{MODEL_SIZE.upper()}")
    model = create_model(model_size=MODEL_SIZE, img_size=IMG_SIZE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device=device)

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    trainer.train(num_epochs=NUM_EPOCHS, save_every=5)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 Checkpoints saved to: /kaggle/working/checkpoints/")
    print(f"🏆 Best model: /kaggle/working/checkpoints/best_psnr.pth")
    print("="*60)

    # Create download link
    try:
        from IPython.display import FileLink
        print("\n📥 Download your trained model:")
        display(FileLink('/kaggle/working/checkpoints/best_psnr.pth'))
    except:
        pass


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    main()

