"""
Configuration file for diffuser maze project.

Copy from notebook:
- Cell 6: device = "cuda:5" (or your device selection)
- Cell 10: HORIZON = 256
- Cell 15: TRANSITION_DIM = 4
- Cell 21: TRAJ_DIM = 2 * HORIZON (calculated value: 512)
- Cell 25: precision_config dictionary with all training hyperparameters
- Cell 30: Numerical stability constants (PRECISION_RIDGE, WOODBURY_JITTER, EIGEN_FLOOR, BLOCK_RIDGE)

Also include:
- Dataset configuration (dataset_id, stride)
- File paths (savepath, precision_savepath)
- Diffuser hyperparameters (model_dim, emb_dim, kernel_size, etc.)
"""

# ============================================================================
# Device Configuration
# ============================================================================
device = "cuda:5"


# ============================================================================
# Model Dimensions
# ============================================================================
HORIZON = 256
TRANSITION_DIM = 4
TRAJ_DIM = 2 * HORIZON  # 512


# ============================================================================
# Dataset Configuration
# ============================================================================
dataset_id = "pointmaze/my-rrt-umaze-large-v0"
stride = HORIZON # For sliding window over long episodes (diffusion policy format)


# ============================================================================
# Diffuser Model Configuration
# ============================================================================
# Hyperparameters for JannerUNet1d and ContinuousDiffusionSDE
model_dim = 64
emb_dim = 64
kernel_size = 5
dim_mult = [1, 2, 4, 8]
attention = True
norm_type = "groupnorm"
ema_rate = 0.9999
x_max = [3, 3, 1, 1]
x_min = [-3, -3, -1, -1]
predict_noise = True


# ============================================================================
# Precision Head Model Configuration
# ===========================================================================
STATE_DIM = 2
hidden_dim = 512
low_rank_dim = 16
use_low_rank = False


# ============================================================================
# HVP Computation Configuration (from FlowUncertainty_v8-4)
# ============================================================================
# Multi-sigma evaluation for robust curvature estimation
# Instead of evaluating HVP at a single noise level, we evaluate at multiple
# small σ values and average the results for improved stability.
#
# Algorithm: h_final = (1/K) Σ_k ∇_y(s_θ(y,σ_k)ᵀu) + δ·u
#
# Benefits:
# - Reduces HVP variance by 2-5× compared to single-sigma
# - Improves cosine similarity with true Hessian from ~0.65 to ~0.85
# - More stable training gradients
# - Better numerical conditioning of precision matrix
#
# Trade-off: K× computational cost (but still much cheaper than sampling)

# Time values: Time parameters passed to diffusion model (NOT actual sigma!)
# These map to actual noise levels via the noise schedule:
#   t=0.005 → σ≈0.016, t=0.01 → σ≈0.025, t=0.02 → σ≈0.042 (cosine schedule)
# Typical range: [0.001, 0.05], recommended: 3-5 values
# - Smaller values: sharper curvature, larger HVP
# - Larger values: smoother curvature, smaller HVP
# - Use larger values if HVP is too large or training is unstable
T_EVALS = [0.005, 0.01, 0.02]  # Default: 3 small time values

# Curvature floor: Regularization constant δ added to final HVP
# Formula: h_final = h_avg + δ·u
#
# Purpose: Prevents unbounded covariance eigenvalues
# - Without floor: max eigenvalue of Σ = Λ^(-1) can be unbounded
# - With floor: max eigenvalue bounded by 1/δ
#
# Ideal contribution: 5-15% of total HVP norm
# - Too high (>50%): Over-regularized, covariances too small
# - Too low (<1%): Under-regularized, may have numerical issues
#
# Tuning:
# - Increase if eigenvalues become unbounded (0.15-0.2)
# - Decrease if covariances seem over-regularized (0.05-0.08)
CURVATURE_FLOOR = 0.2  # Typical range: 0.05-0.2


# ============================================================================
# Training Configuration
# ============================================================================
precision_config = {
    'batch_size': 64,            # Batch size for precision training
    'num_epochs': 400,           # Number of training epochs
    'learning_rate': 3e-5,       # Reduced learning rate for stability
    'weight_decay': 1e-5,        # L2 regularization
    'l2_reg_r': 1e-3,           # Additional L2 penalty on R (low-rank component)
    'temporal_smooth_reg': 1e-4,  # Temporal smoothness regularization
    'grad_clip_norm': 1,       # Reduced gradient clipping for better monitoring
    'num_probes': 3,             # Number of probe vectors per trajectory
    'sample_steps': 15,          # Diffusion sampling steps for trajectory generation
    'log_every': 5,              # Increased logging frequency
    'save_every': 50,            # Checkpoint saving frequency
    'monitor_every': 1,          # Monitor every epoch

    # HVP computation parameters (multi-sigma averaging from FlowUncertainty_v8-4)
    'use_multi_sigma_hvp': True,        # Enable multi-sigma HVP
    't_evals': T_EVALS,                 # Time values for HVP (NOT actual sigma!)
    'curvature_floor': CURVATURE_FLOOR, # Regularization strength
    'use_noised_trajectory': True,      # Add noise for mathematical correctness

    # Probe vector configuration
    'probe_type': 'rademacher',         # Probe type: 'rademacher' (default) or 'gaussian'
    'normalize_gaussian': True,         # Normalize Gaussian probes to unit norm (only for 'gaussian')
}


# ============================================================================
# Numerical Stability Configuration
# ============================================================================
# TODO: what is λ ?
# Ridge regularization for precision matrix: A = LL^T + λI
PRECISION_RIDGE = 1e-7

# Jitter for Woodbury matrix inversion: M = M + εI
# Prevents singular matrices when using low-rank updates
WOODBURY_JITTER = 1e-5

# Eigenvalue floor for 2x2 marginal covariance blocks
# Ensures positive definiteness: eigenvalues ≥ EIGEN_FLOOR
EIGEN_FLOOR = 1e-9

# Optional ridge for 2x2 covariance blocks (usually 0)
# Applied as: Σ = (Σ^{-1} + λI)^{-1}
BLOCK_RIDGE = 0.0


# ============================================================================
# File Paths
# ============================================================================
savepath = "./results/diffuser_maze_znorm/"
precision_savepath = "./results/precision_head/"
