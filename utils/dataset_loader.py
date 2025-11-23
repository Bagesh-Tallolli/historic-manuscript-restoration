"""
Dataset loader for Sanskrit manuscript image restoration
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import random


class ManuscriptDataset(Dataset):
    """
    Dataset for Sanskrit manuscript images
    Loads pairs of degraded and clean images for training
    """

    def __init__(
        self,
        data_dir,
        img_size=256,
        mode='train',
        augment=True,
        synthetic_degradation=True
    ):
        """
        Args:
            data_dir: Directory containing images
            img_size: Size to resize images to
            mode: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
            synthetic_degradation: If True, create degraded images synthetically
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.mode = mode
        self.augment = augment and mode == 'train'
        self.synthetic_degradation = synthetic_degradation

        # Find all images
        self.image_paths = self._find_images()

        print(f"Found {len(self.image_paths)} images in {mode} mode")

    def _find_images(self):
        """Find all image files in the directory"""
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        image_paths = []

        for ext in extensions:
            image_paths.extend(self.data_dir.glob(f'**/*{ext}'))
            image_paths.extend(self.data_dir.glob(f'**/*{ext.upper()}'))

        return sorted(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        clean_img = self._load_image(img_path)

        if self.synthetic_degradation:
            # Create degraded version synthetically
            degraded_img = self._degrade_image(clean_img.copy())
        else:
            # TODO: Load paired degraded image if available
            degraded_img = clean_img.copy()

        # Apply augmentation
        if self.augment:
            degraded_img, clean_img = self._augment(degraded_img, clean_img)

        # Resize
        degraded_img = cv2.resize(degraded_img, (self.img_size, self.img_size))
        clean_img = cv2.resize(clean_img, (self.img_size, self.img_size))

        # Convert to tensor
        degraded_tensor = self._to_tensor(degraded_img)
        clean_tensor = self._to_tensor(clean_img)

        return {
            'degraded': degraded_tensor,
            'clean': clean_tensor,
            'path': str(img_path)
        }

    def _load_image(self, path):
        """Load image from path"""
        img = cv2.imread(str(path))
        if img is None:
            img = np.array(Image.open(path).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _degrade_image(self, img):
        """
        Synthetically degrade an image to simulate manuscript damage
        Applies: noise, blur, fading, stains, etc.
        """
        img = img.astype(np.float32) / 255.0

        # Random combination of degradations

        # 1. Add Gaussian noise
        if random.random() > 0.3:
            noise_sigma = random.uniform(0.01, 0.08)
            noise = np.random.normal(0, noise_sigma, img.shape)
            img = img + noise

        # 2. Apply blur (simulating fading/aging)
        if random.random() > 0.3:
            kernel_size = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # 3. Reduce contrast (fading)
        if random.random() > 0.4:
            alpha = random.uniform(0.6, 0.9)  # contrast
            beta = random.uniform(0.05, 0.15)  # brightness
            img = alpha * img + beta

        # 4. Add salt and pepper noise
        if random.random() > 0.5:
            noise_ratio = random.uniform(0.001, 0.01)
            mask = np.random.random(img.shape[:2]) < noise_ratio
            img[mask] = np.random.choice([0, 1], size=(mask.sum(), 3))

        # 5. Color shifting (aging effect)
        if random.random() > 0.4:
            # Yellowish tint
            yellow_tint = np.array([1.0, 0.95, 0.8])
            img = img * yellow_tint

        # 6. Random stains/spots
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

        # Clip values
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        return img

    def _augment(self, degraded, clean):
        """Apply data augmentation (same transform to both images)"""
        # Random horizontal flip
        if random.random() > 0.5:
            degraded = cv2.flip(degraded, 1)
            clean = cv2.flip(clean, 1)

        # Random vertical flip
        if random.random() > 0.5:
            degraded = cv2.flip(degraded, 0)
            clean = cv2.flip(clean, 0)

        # Random rotation (90, 180, 270)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            degraded = np.rot90(degraded, k)
            clean = np.rot90(clean, k)

        # Random crop (then resize back)
        if random.random() > 0.3:
            h, w = degraded.shape[:2]
            crop_size = int(min(h, w) * random.uniform(0.8, 1.0))
            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)

            degraded = degraded[y:y+crop_size, x:x+crop_size]
            clean = clean[y:y+crop_size, x:x+crop_size]

        return degraded, clean

    def _to_tensor(self, img):
        """Convert numpy image to PyTorch tensor"""
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img)


def create_dataloaders(
    train_dir,
    val_dir=None,
    batch_size=16,
    img_size=256,
    num_workers=4,
    synthetic_degradation=True
):
    """
    Create train and validation dataloaders

    Args:
        train_dir: Directory with training images
        val_dir: Directory with validation images (optional)
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers for data loading
        synthetic_degradation: Whether to create synthetic degradations

    Returns:
        train_loader, val_loader (val_loader is None if val_dir not provided)
    """
    # Training dataset
    train_dataset = ManuscriptDataset(
        train_dir,
        img_size=img_size,
        mode='train',
        augment=True,
        synthetic_degradation=synthetic_degradation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Validation dataset
    val_loader = None
    if val_dir is not None:
        val_dataset = ManuscriptDataset(
            val_dir,
            img_size=img_size,
            mode='val',
            augment=False,
            synthetic_degradation=synthetic_degradation
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing ManuscriptDataset...")

    # Create a dummy dataset directory for testing
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"Warning: {data_dir} does not exist")
        print("Please create this directory and add some images")
    else:
        dataset = ManuscriptDataset(data_dir, img_size=256, mode='train')

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample data:")
            print(f"  Degraded shape: {sample['degraded'].shape}")
            print(f"  Clean shape: {sample['clean'].shape}")
            print(f"  Image path: {sample['path']}")

            # Test dataloader
            train_loader, _ = create_dataloaders(data_dir, batch_size=4)
            batch = next(iter(train_loader))
            print(f"\nBatch shapes:")
            print(f"  Degraded: {batch['degraded'].shape}")
            print(f"  Clean: {batch['clean'].shape}")
        else:
            print("No images found in dataset")

