"""
Marginal Gaussian Dataset Generators for Uncertainty Quantification

This module provides synthetic datasets with known ground-truth covariance structures
for evaluating precision head predictions:

1. **2D Isotropic Marginal Dataset**:
   - Independent Gaussian noise per timestep (no temporal correlation)
   - Isotropic uncertainty: S_t = σ_t² I
   - Mid-peak variance schedule

2. **1D Y-Only Noise Dataset**:
   - Deterministic x: x_t = t (+ tiny noise for numerical stability)
   - Gaussian noise only in y: y_t ~ N(μ_y(t), σ_t²)
   - Easier validation: compare predicted σ_y vs ground-truth directly

Dependencies:
- numpy
- matplotlib
- dataclasses (for config objects)

Usage:
    from diffuser_maze.data.marginal_datasets import (
        sample_1d_gaussian_dataset,
        OneDGaussianDataset,
        plot_1d_dataset_overview
    )

    # Configure 1D dataset
    config = OneDGaussianDataset(
        n_samples=2500,
        T=10,
        y_function='sine',
        sigma_min=0.02,
        sigma_max=0.15
    )

    # Generate
    X, mu, S_true, meta = sample_1d_gaussian_dataset(**config.__dict__)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Literal
import math


# ==============================================================================
# Utility: Chi-squared quantile for 2D
# ==============================================================================

def chi2_ppf_df2(alpha: float) -> float:
    """Chi-squared quantile for df=2 (2D Gaussian)."""
    return -2.0 * math.log(max(1e-16, 1.0 - float(alpha)))


# ==============================================================================
# Variance Schedule (shared by 2D and 1D datasets)
# ==============================================================================

def marginal_variance_schedule(
    T: int,
    sigma_min: float = 0.02,
    sigma_max: float = 0.15,
    schedule: Literal['cosine', 'gaussian', 'quadratic'] = 'cosine'
) -> np.ndarray:
    """
    Generate per-timestep variance schedule with mid-peak profile.

    Args:
        T: Number of timesteps
        sigma_min: Minimum std (at endpoints)
        sigma_max: Maximum std (at middle)
        schedule: 'cosine', 'gaussian', or 'quadratic'

    Returns:
        sigma_t: np.ndarray [T] with σ_t values
    """
    t_norm = np.linspace(0, 1, T)  # normalized time [0, 1]

    if schedule == 'cosine':
        # σ_t = σ_min + (σ_max - σ_min) * sin²(πt)
        profile = np.sin(np.pi * t_norm) ** 2
    elif schedule == 'gaussian':
        # Gaussian bump centered at t=0.5
        profile = np.exp(-((t_norm - 0.5) ** 2) / (2 * 0.15 ** 2))
    elif schedule == 'quadratic':
        # Parabola peaking at t=0.5
        profile = 1.0 - 4 * (t_norm - 0.5) ** 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    sigma_t = sigma_min + (sigma_max - sigma_min) * profile
    return sigma_t


# ==============================================================================
# 2D Isotropic Marginal Gaussian Dataset
# ==============================================================================

@dataclass
class MarginalGaussianDataset:
    """Configuration for 2D isotropic marginal Gaussian dataset."""
    n_samples: int = 2500
    T: int = 10
    x_start: np.ndarray = None
    x_T_star: np.ndarray = None
    sigma_min: float = 0.02
    sigma_max: float = 0.15
    schedule: Literal['cosine', 'gaussian', 'quadratic'] = 'cosine'
    isotropic: bool = True
    endpoint_sigma: float = 0.01  # Small variance at endpoint (avoids whitening issues)
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.x_start is None:
            self.x_start = np.array([1.0, 1.0])
        if self.x_T_star is None:
            self.x_T_star = np.array([5.0, 1.0])


def build_ground_truth_covariances(
    sigma_schedule: np.ndarray,
    isotropic: bool = True,
    endpoint_sigma: float = 0.01
) -> np.ndarray:
    """
    Build ground-truth covariance matrices for each timestep.

    Args:
        sigma_schedule: np.ndarray [T] with σ_t values
        isotropic: If True, S_t = σ_t² I; else anisotropic with aspect ratio
        endpoint_sigma: Small std for endpoint (avoids whitening issues)

    Returns:
        S_true: np.ndarray [T, 2, 2] covariance matrices
    """
    T = len(sigma_schedule)
    S_true = np.zeros((T, 2, 2), dtype=np.float64)

    for t in range(T - 1):  # All except last timestep
        sigma = sigma_schedule[t]
        if isotropic:
            S_true[t] = (sigma ** 2) * np.eye(2, dtype=np.float64)
        else:
            # Anisotropic: aspect ratio 2:1
            S_true[t] = np.diag([sigma ** 2, (0.5 * sigma) ** 2]).astype(np.float64)

    # Last timestep: small variance (avoids whitening scale issues)
    S_true[T - 1] = (endpoint_sigma ** 2) * np.eye(2, dtype=np.float64)

    return S_true


def sample_marginal_gaussian_dataset(
    n_samples: int,
    T: int,
    x_start: np.ndarray = np.array([0.0, 0.0]),
    x_T_star: np.ndarray = np.array([5.0, 0.0]),
    sigma_min: float = 0.02,
    sigma_max: float = 0.15,
    schedule: Literal['cosine', 'gaussian', 'quadratic'] = 'cosine',
    isotropic: bool = True,
    endpoint_sigma: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Sample trajectories from marginal-only Gaussian distribution (2D isotropic).

    Each trajectory:
      - Starts near x_start with low variance
      - Has highest variance in the middle
      - Ends near x_T_star with small variance (endpoint_sigma)
      - Independent noise per timestep (no temporal correlation)

    Args:
        n_samples: Number of trajectories to sample
        T: Number of timesteps per trajectory
        x_start: Starting point (mean at t=0)
        x_T_star: Endpoint mean (small variance around this)
        sigma_min: Minimum std (at endpoints)
        sigma_max: Maximum std (at middle)
        schedule: Variance schedule type
        isotropic: Use circular (True) or elliptical (False) uncertainty
        endpoint_sigma: Small std for endpoint (avoids whitening issues)
        seed: Random seed

    Returns:
        X: np.ndarray [n_samples, T, 2] - sampled trajectories
        mu: np.ndarray [T, 2] - mean path (straight line)
        S_true: np.ndarray [T, 2, 2] - ground-truth covariances
        meta: dict - generation metadata
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Generate variance schedule
    sigma_schedule = marginal_variance_schedule(T, sigma_min, sigma_max, schedule)

    # 2. Build ground-truth covariances
    S_true = build_ground_truth_covariances(sigma_schedule, isotropic, endpoint_sigma)

    # 3. Compute mean path (straight line from x_start to x_T_star)
    t_norm = np.linspace(0, 1, T)
    mu = x_start[np.newaxis, :] + t_norm[:, np.newaxis] * (x_T_star - x_start)[np.newaxis, :]

    # 4. Sample trajectories
    X = np.zeros((n_samples, T, 2), dtype=np.float64)

    for i in range(n_samples):
        for t in range(T):
            # Sample from N(μ_t, S_t)
            noise = np.random.multivariate_normal(np.zeros(2), S_true[t])
            X[i, t, :] = mu[t] + noise

    # 5. Metadata
    meta = {
        'n_samples': n_samples,
        'T': T,
        'x_start': x_start.copy(),
        'x_T_star': x_T_star.copy(),
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'schedule': schedule,
        'isotropic': isotropic,
        'endpoint_sigma': endpoint_sigma,
        'seed': seed,
        'dataset_type': '2D_isotropic_marginal'
    }

    return X, mu, S_true, meta


# ==============================================================================
# 1D Y-Only Noise Dataset
# ==============================================================================

@dataclass
class OneDGaussianDataset:
    """Configuration for 1D Y-only noise dataset (x deterministic)."""
    n_samples: int = 2500
    T: int = 10
    y_function: Literal['constant', 'sine', 'linear', 'parabola'] = 'sine'
    y_amplitude: float = 2.0
    y_offset: float = 1.0
    sigma_min: float = 0.02
    sigma_max: float = 0.15
    schedule: Literal['cosine', 'gaussian', 'quadratic'] = 'cosine'
    x_variance_eps: float = 1e-6  # Tiny variance for x (numerical stability)
    endpoint_sigma: float = 0.01
    seed: Optional[int] = 42


def build_1d_mean_path(
    T: int,
    function_type: Literal['constant', 'sine', 'linear', 'parabola'] = 'sine',
    amplitude: float = 2.0,
    offset: float = 1.0
) -> np.ndarray:
    """
    Generate deterministic x and mean y path.

    Args:
        T: Number of timesteps
        function_type: Shape of y path
            - 'constant': y = offset (no variation, simplest case)
            - 'sine': y = offset + amplitude * sin(2πt)
            - 'linear': y = offset + amplitude * t
            - 'parabola': y = offset + amplitude * (1 - 4(t-0.5)²)
        amplitude: Amplitude of y variation (ignored for 'constant')
        offset: Offset of y from zero

    Returns:
        mu: np.ndarray [T, 2] with mu[:, 0] = [0, 1, ..., T-1] and mu[:, 1] = y path
    """
    mu = np.zeros((T, 2), dtype=np.float64)

    # x is deterministic: 0, 1, 2, ..., T-1
    mu[:, 0] = np.arange(T, dtype=np.float64)

    # y follows specified function
    t_norm = np.linspace(0, 1, T)

    if function_type == 'constant':
        # Simplest case: y = constant (just the offset)
        mu[:, 1] = offset
    elif function_type == 'sine':
        mu[:, 1] = offset + amplitude * np.sin(2 * np.pi * t_norm)
    elif function_type == 'linear':
        mu[:, 1] = offset + amplitude * t_norm
    elif function_type == 'parabola':
        mu[:, 1] = offset + amplitude * (1.0 - 4 * (t_norm - 0.5) ** 2)
    else:
        raise ValueError(f"Unknown function_type: {function_type}")

    return mu


def build_1d_covariances(
    sigma_schedule: np.ndarray,
    x_eps: float = 1e-6
) -> np.ndarray:
    """
    Build diagonal covariances for 1D case: S_t = diag([x_eps², σ_t²]).

    Args:
        sigma_schedule: [T] variance schedule for y
        x_eps: Tiny variance for x (numerical stability)

    Returns:
        S_true: [T, 2, 2] diagonal covariances
    """
    T = len(sigma_schedule)
    S_true = np.zeros((T, 2, 2), dtype=np.float64)

    for t in range(T):
        S_true[t] = np.diag([x_eps ** 2, sigma_schedule[t] ** 2])

    return S_true


def sample_1d_gaussian_dataset(
    n_samples: int,
    T: int,
    y_function: Literal['constant', 'sine', 'linear', 'parabola'] = 'sine',
    y_amplitude: float = 2.0,
    y_offset: float = 1.0,
    sigma_min: float = 0.02,
    sigma_max: float = 0.15,
    schedule: Literal['cosine', 'gaussian', 'quadratic'] = 'cosine',
    x_variance_eps: float = 1e-6,
    endpoint_sigma: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Sample trajectories with uncertainty ONLY in y direction.

    Each trajectory:
      - x is deterministic: x_t = t (+ tiny noise for numerical stability)
      - y has Gaussian noise: y_t ~ N(μ_y(t), σ_t²)
      - Easier validation: σ_y is directly comparable to ground truth

    Args:
        n_samples: Number of trajectories
        T: Number of timesteps
        y_function: Mean path shape for y ('constant', 'sine', 'linear', 'parabola')
            - 'constant': y = y_offset (simplest, just a horizontal line)
        y_amplitude: Amplitude of y variation (ignored for 'constant')
        y_offset: Offset of y from zero (for 'constant', this is the y value)
        sigma_min: Min std for y
        sigma_max: Max std for y
        schedule: Variance schedule type
        x_variance_eps: Tiny variance for x (numerical stability, ~1e-6)
        endpoint_sigma: Small std for y at endpoint
        seed: Random seed

    Returns:
        X: [n_samples, T, 2] trajectories
        mu: [T, 2] mean path
        S_true: [T, 2, 2] ground-truth covariances (diagonal)
        meta: dict with metadata
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Generate mean path
    mu = build_1d_mean_path(T, y_function, y_amplitude, y_offset)

    # 2. Generate variance schedule (for y only)
    sigma_schedule = marginal_variance_schedule(T, sigma_min, sigma_max, schedule)

    # Override last timestep with endpoint_sigma
    sigma_schedule[-1] = endpoint_sigma

    # 3. Build covariances
    S_true = build_1d_covariances(sigma_schedule, x_variance_eps)

    # 4. Sample trajectories
    X = np.zeros((n_samples, T, 2), dtype=np.float64)

    for i in range(n_samples):
        for t in range(T):
            # Sample from N(μ_t, S_t)
            noise = np.random.multivariate_normal(np.zeros(2), S_true[t])
            X[i, t, :] = mu[t] + noise

    # 5. Metadata
    meta = {
        'n_samples': n_samples,
        'T': T,
        'y_function': y_function,
        'y_amplitude': y_amplitude,
        'y_offset': y_offset,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'schedule': schedule,
        'x_variance_eps': x_variance_eps,
        'endpoint_sigma': endpoint_sigma,
        'seed': seed,
        'dataset_type': '1D_y_only_noise'
    }

    return X, mu, S_true, meta


# ==============================================================================
# Visualization Functions for 1D Dataset
# ==============================================================================

def plot_1d_dataset_overview(
    X: np.ndarray,
    mu: np.ndarray,
    S_true: np.ndarray,
    alpha: float = 0.9,
    n_show: int = 30,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5)
):
    """
    Visualize 1D Y-only noise dataset.

    Args:
        X: [n_samples, T, 2] trajectories
        mu: [T, 2] mean path
        S_true: [T, 2, 2] ground-truth covariances
        alpha: Coverage level for error bars
        n_show: Number of trajectories to display
        title: Plot title
        figsize: Figure size
    """
    T = X.shape[1]
    n_samples = X.shape[0]
    n_show = min(n_show, n_samples)

    # Extract y uncertainties
    sigma_y = np.sqrt(S_true[:, 1, 1])

    # Z-score for alpha coverage (1D Gaussian)
    from scipy import stats
    z = stats.norm.ppf((1 + alpha) / 2)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Trajectories in (x, y) space
    ax = axes[0]
    for i in range(n_show):
        ax.plot(X[i, :, 0], X[i, :, 1], '-', alpha=0.3, lw=0.8, color='C0')

    # Mean path
    ax.plot(mu[:, 0], mu[:, 1], 'k-', lw=2.5, label='Mean path', zorder=10)

    # Uncertainty band
    ax.fill_between(
        mu[:, 0],
        mu[:, 1] - z * sigma_y,
        mu[:, 1] + z * sigma_y,
        alpha=0.3,
        color='orange',
        label=f'{int(100*alpha)}% band (y only)'
    )

    ax.set_xlabel('x (deterministic)', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title('Trajectories with Y-only Uncertainty', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Right: Y variance over time
    ax = axes[1]
    t_grid = np.arange(T)
    ax.plot(t_grid, sigma_y, 'o-', lw=2, ms=6, color='C1', label='σ_y(t)')
    ax.axhline(sigma_y.min(), ls='--', color='gray', lw=1, alpha=0.5)
    ax.axhline(sigma_y.max(), ls='--', color='gray', lw=1, alpha=0.5)
    ax.set_xlabel('Timestep t', fontsize=11)
    ax.set_ylabel('Standard deviation σ_y', fontsize=11)
    ax.set_title('Y-Uncertainty Schedule', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.show()


def plot_1d_variance_comparison(
    sigma_gt: np.ndarray,
    sigma_pred: np.ndarray,
    timesteps: Optional[np.ndarray] = None,
    title: str = "Predicted vs Ground-Truth σ_y(t)",
    figsize: Tuple[float, float] = (10, 4)
):
    """
    Compare ground-truth and predicted y-uncertainty.

    Args:
        sigma_gt: [T] ground-truth σ_y values
        sigma_pred: [T] predicted σ_y values
        timesteps: Optional timestep array
        title: Plot title
        figsize: Figure size
    """
    T = len(sigma_gt)
    if timesteps is None:
        timesteps = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Overlay
    ax = axes[0]
    ax.plot(timesteps, sigma_gt, 'o-', lw=2, ms=6, label='Ground Truth', color='C2')
    ax.plot(timesteps, sigma_pred, 's-', lw=2, ms=5, label='Predicted', color='C3', alpha=0.7)
    ax.set_xlabel('Timestep t', fontsize=11)
    ax.set_ylabel('σ_y', fontsize=11)
    ax.set_title('Comparison', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Right: Relative error
    ax = axes[1]
    rel_error = np.abs(sigma_pred - sigma_gt) / (sigma_gt + 1e-12)
    ax.plot(timesteps, 100 * rel_error, 'o-', lw=2, ms=5, color='C1')
    ax.axhline(0, ls='--', color='gray', lw=1)
    ax.set_xlabel('Timestep t', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title('Prediction Error', fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# Evaluation Functions for 1D Dataset
# ==============================================================================

def extract_y_std_from_covariance(S: np.ndarray) -> float:
    """
    Extract σ_y from 2x2 covariance matrix.

    Args:
        S: [2, 2] covariance matrix

    Returns:
        σ_y: sqrt(S[1, 1])
    """
    return np.sqrt(np.maximum(S[1, 1], 1e-12))


def evaluate_1d_precision_accuracy(
    S_pred: np.ndarray,
    S_true: np.ndarray
) -> dict:
    """
    Evaluate 1D precision head accuracy.

    Args:
        S_pred: [T, 2, 2] predicted covariances
        S_true: [T, 2, 2] ground-truth covariances

    Returns:
        metrics: dict with MSE, relative error, correlation
    """
    T = S_pred.shape[0]

    # Extract σ_y for each timestep
    sigma_y_pred = np.array([extract_y_std_from_covariance(S_pred[t]) for t in range(T)])
    sigma_y_true = np.array([extract_y_std_from_covariance(S_true[t]) for t in range(T)])

    # Compute metrics
    mse = np.mean((sigma_y_pred - sigma_y_true) ** 2)
    mae = np.mean(np.abs(sigma_y_pred - sigma_y_true))
    rel_error = np.mean(np.abs(sigma_y_pred - sigma_y_true) / (sigma_y_true + 1e-12))
    correlation = np.corrcoef(sigma_y_pred, sigma_y_true)[0, 1]

    return {
        'mse': mse,
        'mae': mae,
        'relative_error': rel_error,
        'correlation': correlation,
        'sigma_y_pred': sigma_y_pred,
        'sigma_y_true': sigma_y_true
    }


def plot_precision_comparison_1d(
    S_pred: np.ndarray,
    S_true: np.ndarray,
    title: str = "Precision Head: 1D Y-Uncertainty Prediction"
):
    """
    Visualize precision head prediction quality for 1D dataset.

    Args:
        S_pred: [T, 2, 2] predicted covariances
        S_true: [T, 2, 2] ground-truth covariances
        title: Plot title
    """
    metrics = evaluate_1d_precision_accuracy(S_pred, S_true)

    plot_1d_variance_comparison(
        metrics['sigma_y_true'],
        metrics['sigma_y_pred'],
        title=f"{title}\nMSE={metrics['mse']:.4e}, Rel.Err={100*metrics['relative_error']:.2f}%, Corr={metrics['correlation']:.3f}"
    )
