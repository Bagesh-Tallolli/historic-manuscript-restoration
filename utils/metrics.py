"""
Metrics for evaluating image restoration quality
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


class ImageMetrics:
    """Calculate various image quality metrics"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Initialize LPIPS (Learned Perceptual Image Patch Similarity)
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        except:
            print("Warning: LPIPS not available")
            self.lpips_fn = None

    def psnr(self, pred, target):
        """
        Calculate Peak Signal-to-Noise Ratio

        Args:
            pred: Predicted image (numpy array or tensor)
            target: Target image (numpy array or tensor)

        Returns:
            PSNR value in dB
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        return peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)

    def ssim(self, pred, target):
        """
        Calculate Structural Similarity Index

        Args:
            pred: Predicted image (numpy array or tensor)
            target: Target image (numpy array or tensor)

        Returns:
            SSIM value (0 to 1)
        """
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        # Handle batch dimension
        if pred_np.ndim == 4:
            ssim_values = []
            for i in range(pred_np.shape[0]):
                ssim_val = structural_similarity(
                    target_np[i].transpose(1, 2, 0),
                    pred_np[i].transpose(1, 2, 0),
                    data_range=1.0,
                    channel_axis=2,
                    multichannel=True
                )
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:
            return structural_similarity(
                target_np.transpose(1, 2, 0),
                pred_np.transpose(1, 2, 0),
                data_range=1.0,
                channel_axis=2,
                multichannel=True
            )

    def lpips_score(self, pred, target):
        """
        Calculate LPIPS perceptual similarity

        Args:
            pred: Predicted image (tensor)
            target: Target image (tensor)

        Returns:
            LPIPS score (lower is better)
        """
        if self.lpips_fn is None:
            return None

        pred_tensor = self._to_tensor(pred)
        target_tensor = self._to_tensor(target)

        with torch.no_grad():
            lpips_val = self.lpips_fn(pred_tensor, target_tensor)

        return lpips_val.mean().item()

    def mse(self, pred, target):
        """Calculate Mean Squared Error"""
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        return np.mean((pred_np - target_np) ** 2)

    def mae(self, pred, target):
        """Calculate Mean Absolute Error"""
        pred_np = self._to_numpy(pred)
        target_np = self._to_numpy(target)

        return np.mean(np.abs(pred_np - target_np))

    def calculate_all(self, pred, target):
        """
        Calculate all metrics

        Returns:
            Dictionary with all metric values
        """
        metrics = {
            'psnr': self.psnr(pred, target),
            'ssim': self.ssim(pred, target),
            'mse': self.mse(pred, target),
            'mae': self.mae(pred, target),
        }

        lpips_val = self.lpips_score(pred, target)
        if lpips_val is not None:
            metrics['lpips'] = lpips_val

        return metrics

    def _to_numpy(self, img):
        """Convert image to numpy array"""
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # Ensure values are in [0, 1]
        img = np.clip(img, 0, 1)

        return img

    def _to_tensor(self, img):
        """Convert image to tensor"""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()

        if img.device != self.device:
            img = img.to(self.device)

        # Ensure 4D tensor (batch, channels, height, width)
        if img.ndim == 3:
            img = img.unsqueeze(0)

        # Normalize to [-1, 1] for LPIPS
        img = img * 2 - 1

        return img


class RunningMetrics:
    """Track running averages of metrics during training"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
        self.counts = {}

    def update(self, metric_dict):
        """
        Update metrics with new values

        Args:
            metric_dict: Dictionary of metric names and values
        """
        for name, value in metric_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0

            self.metrics[name] += value
            self.counts[name] += 1

    def get_averages(self):
        """Get average values of all metrics"""
        averages = {}
        for name in self.metrics:
            if self.counts[name] > 0:
                averages[name] = self.metrics[name] / self.counts[name]
            else:
                averages[name] = 0.0

        return averages

    def get_summary(self):
        """Get formatted summary string"""
        averages = self.get_averages()
        summary = " | ".join([f"{name}: {value:.4f}" for name, value in averages.items()])
        return summary


def calculate_batch_metrics(pred, target, device='cuda'):
    """
    Convenient function to calculate metrics for a batch

    Args:
        pred: Predicted images (tensor or numpy)
        target: Target images (tensor or numpy)
        device: Device to use for computation

    Returns:
        Dictionary with average metrics
    """
    metrics_calc = ImageMetrics(device=device)
    return metrics_calc.calculate_all(pred, target)


if __name__ == "__main__":
    # Test metrics
    print("Testing image metrics...")

    # Create dummy images
    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)

    # Calculate metrics
    metrics = ImageMetrics()
    results = metrics.calculate_all(pred, target)

    print("\nMetric Results:")
    for name, value in results.items():
        print(f"  {name.upper()}: {value:.4f}")

    # Test running metrics
    print("\nTesting running metrics...")
    running = RunningMetrics()

    for i in range(5):
        batch_metrics = {
            'loss': np.random.rand(),
            'psnr': 20 + np.random.rand() * 10,
            'ssim': 0.5 + np.random.rand() * 0.5
        }
        running.update(batch_metrics)

    print("\nRunning averages:")
    print(running.get_summary())

