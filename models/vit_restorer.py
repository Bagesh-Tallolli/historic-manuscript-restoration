"""
Vision Transformer (ViT) based Image Restoration Model for Sanskrit Manuscripts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


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

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

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
    """
    Vision Transformer for Image Restoration
    Designed specifically for Sanskrit manuscript restoration
    """
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
        use_skip_connections=True,
        use_simple_head=False  # For backward compatibility with old checkpoints
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_skip_connections = use_skip_connections
        self.use_simple_head = use_simple_head

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

        # Decoder: support both old and new formats
        if use_simple_head:
            # Old format: simple linear head (for Kaggle checkpoints)
            self.head = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        else:
            # New format: patch reconstruction with rearrange
            self.patch_recon = PatchReconstruction(img_size, patch_size, embed_dim, out_channels)

        # Skip connection fusion
        if use_skip_connections:
            self.skip_fusion = nn.Conv2d(in_channels + out_channels, out_channels, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize other layers
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
        # Store input for skip connection
        input_img = x

        # Patch embedding
        x = self.patch_embed(x)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Normalize
        x = self.norm(x)

        # Reconstruct image (support both old and new decoders)
        if self.use_simple_head:
            # Old format: simple linear head
            x = self.head(x)
            # Reshape patches back to image
            h = w = self.img_size // self.patch_size
            x = x.reshape(x.shape[0], h, w, self.patch_size, self.patch_size, 3)
            x = x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], 3, self.img_size, self.img_size)
        else:
            # New format: patch reconstruction
            x = self.patch_recon(x)

        # Skip connection
        if self.use_skip_connections:
            x = torch.cat([x, input_img], dim=1)
            x = self.skip_fusion(x)

        return x

    def get_attention_maps(self, x):
        """Extract attention maps for visualization"""
        attention_maps = []

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            # Get attention from each block
            B, N, C = x.shape
            qkv = block.attn.qkv(block.norm1(x))
            qkv = qkv.reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach().cpu())

            # Continue forward pass
            x = block(x)

        return attention_maps


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

        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return F.mse_loss(pred_features, target_features)


class CombinedLoss(nn.Module):
    """Combined L1 + Perceptual Loss for restoration"""
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


def create_vit_restorer(model_size='base', img_size=256, **kwargs):
    """
    Factory function to create ViT restorer models

    Args:
        model_size: 'tiny', 'small', 'base', or 'large'
        img_size: Input image size
        **kwargs: Additional arguments for ViTRestorer
    """
    configs = {
        'tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
    }

    config = configs.get(model_size, configs['base'])
    config.update(kwargs)

    return ViTRestorer(img_size=img_size, **config)


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_vit_restorer('base', img_size=256).to(device)

    # Test input
    x = torch.randn(2, 3, 256, 256).to(device)

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

