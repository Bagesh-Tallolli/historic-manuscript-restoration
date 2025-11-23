"""
Training script for ViT-based image restoration model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import json

from models.vit_restorer import create_vit_restorer, CombinedLoss
from utils.dataset_loader import create_dataloaders
from utils.metrics import ImageMetrics, RunningMetrics


class Trainer:
    """Training manager for ViT restoration model"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        criterion=None,
        optimizer=None,
        device='cuda',
        checkpoint_dir='checkpoints',
        use_wandb=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Loss function
        self.criterion = criterion or CombinedLoss()

        # Optimizer
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100,
            eta_min=1e-6
        )

        # Metrics
        self.metrics = ImageMetrics(device=device)

        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("wandb not available, disabling")
                self.use_wandb = False

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0
        self.train_history = []
        self.val_history = []

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_metrics = RunningMetrics()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            degraded = batch['degraded'].to(self.device)
            clean = batch['clean'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            restored = self.model(degraded)

            # Calculate loss
            loss = self.criterion(restored, clean)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Calculate metrics (on a subset to save time)
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    metrics = self.metrics.calculate_all(restored, clean)
            else:
                metrics = {}

            # Update running metrics
            metrics['loss'] = loss.item()
            running_metrics.update(metrics)

            # Update progress bar
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

                # Forward pass
                restored = self.model(degraded)

                # Calculate loss
                loss = self.criterion(restored, clean)

                # Calculate metrics
                metrics = self.metrics.calculate_all(restored, clean)
                metrics['loss'] = loss.item()

                # Update running metrics
                running_metrics.update(metrics)

                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return running_metrics.get_averages()

    def train(self, num_epochs, save_every=5):
        """
        Train the model

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)

                # Print metrics
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train: {self._format_metrics(train_metrics)}")
                print(f"  Val:   {self._format_metrics(val_metrics)}")

                # Log to wandb
                if self.use_wandb:
                    self.wandb.log({
                        'epoch': epoch + 1,
                        **{f'train/{k}': v for k, v in train_metrics.items()},
                        **{f'val/{k}': v for k, v in val_metrics.items()},
                    })

                # Save best model
                val_psnr = val_metrics.get('psnr', 0)
                if val_psnr > self.best_val_psnr:
                    self.best_val_psnr = val_psnr
                    self.save_checkpoint('best_psnr.pth')
                    print(f"  ✓ New best PSNR: {val_psnr:.4f}")

                val_loss = val_metrics.get('loss', float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_loss.pth')
                    print(f"  ✓ New best loss: {val_loss:.4f}")
            else:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Train: {self._format_metrics(train_metrics)}")

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pth')

        # Save final model
        self.save_checkpoint('final.pth')
        print("\n✓ Training complete!")

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_psnr': self.best_val_psnr,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }

        torch.save(checkpoint, checkpoint_path)

        # Also save training history as JSON
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history,
            }, f, indent=2)

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_psnr = checkpoint['best_val_psnr']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _format_metrics(self, metrics):
        """Format metrics for printing"""
        return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])


def main():
    parser = argparse.ArgumentParser(description='Train ViT Image Restoration Model')

    # Data arguments
    parser.add_argument('--train_dir', type=str, default='data/raw',
                       help='Directory with training images')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Directory with validation images')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size for training')

    # Model arguments
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Model size')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for ViT')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Initialize wandb
    if args.use_wandb:
        import wandb
        wandb.init(project='sanskrit-manuscript-restoration', config=vars(args))

    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        synthetic_degradation=True
    )

    # Create model
    print(f"\nCreating {args.model_size} ViT model...")
    model = create_vit_restorer(
        model_size=args.model_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(num_epochs=args.epochs, save_every=5)


if __name__ == "__main__":
    main()

