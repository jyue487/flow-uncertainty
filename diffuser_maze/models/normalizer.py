"""
Normalizer wrapper for GPU-friendly normalization.

Copy from notebook Cell 18 - The TorchGaussianNormalizer class.

This class wraps cleandiffuser.utils.GaussianNormalizer to provide
GPU-friendly tensor-based normalization operations.

Key features:
- Stores mean and std as torch tensors on device
- Provides normalize() and unnormalize() methods
- Compatible with autograd for gradient computation

Dependencies:
- torch
- cleandiffuser.utils.GaussianNormalizer

Usage:
    from diffuser_maze.models.normalizer import TorchGaussianNormalizer

    # Create from GaussianNormalizer
    normalizer = TorchGaussianNormalizer(original_normalizer, device)

    # Normalize data
    normalized = normalizer.normalize(data)

    # Unnormalize
    original = normalizer.unnormalize(normalized)
"""

import torch
from cleandiffuser.utils import GaussianNormalizer
import numpy as np

class TorchGaussianNormalizer:
    def __init__(self, data, device='cpu'):
        """
        GPU-friendly wrapper for GaussianNormalizer.
        Args:
            data: np.ndarray of shape [B, H, D] or [N, D]
            device: device to store normalization parameters
        """
        self.device = device

        # Convert to numpy if needed
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data

        # Reshape if 3D for GaussianNormalizer
        if data_np.ndim == 3:
            B, H, D = data_np.shape
            data_flat = data_np.reshape(-1, D)
        else:
            data_flat = data_np

        # Create GaussianNormalizer
        self.normalizer = GaussianNormalizer(data_flat)

        # Convert mean and std to torch tensors on device
        self.mean = torch.tensor(self.normalizer.mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(self.normalizer.std, dtype=torch.float32, device=device)

        print(f"GaussianNormalizer initialized on {device}:")
        print(f"  Input shape: {data_np.shape}")
        print(f"  Mean values: {self.normalizer.mean}")
        print(f"  Std values: {self.normalizer.std}")

    def normalize(self, x):
        """
        Normalize data to zero mean and unit variance.
        Works with both numpy arrays and torch tensors.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if x.device != self.device:
            x = x.to(self.device)

        # Gaussian normalization
        return (x - self.mean) / self.std

    def unnormalize(self, x_norm):
        """
        Unnormalize data from standard normal back to original range.
        """
        if isinstance(x_norm, np.ndarray):
            x_norm = torch.tensor(x_norm, dtype=torch.float32, device=self.device)

        if x_norm.device != self.device:
            x_norm = x_norm.to(self.device)

        # Reverse Gaussian normalization
        return x_norm * self.std + self.mean