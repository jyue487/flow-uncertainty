"""
Covariance computation utilities.

Copy from notebook Cell 31:
1. full_cov_from_factors_np() - Compute full covariance from L and R
2. blocks_from_cov_np() - Extract per-timestep 2×2 blocks
3. unnormalize_block() - Transform block from normalized to raw space

These functions handle:
- Stable covariance computation via Woodbury identity
- Eigenvalue clamping for numerical stability
- Space transformation for unnormalization

Dependencies:
- numpy

Usage:
    from diffuser_maze.evaluation.covariance import full_cov_from_factors_np

    # Compute covariance
    cov_matrix = full_cov_from_factors_np(
        L_t=L_matrix,
        R_t=R_matrix,
        precision_ridge=1e-7,
        woodbury_jitter=1e-5
    )

    # Extract blocks
    blocks = blocks_from_cov_np(cov_matrix, horizon=256)

    # Unnormalize
    block_raw = unnormalize_block(block_norm, t=0, normalizer=normalizer)
"""
import torch
import numpy as np
#
# def full_cov_from_factors_np(L_t, R_t, precision_ridge=1e-7, woodbury_jitter=1e-5):
#     """
#     Compute covariance Σ = (L L^T + λI + R R^T)^(-1) via Woodbury identity.
#
#     Method:
#     1. Compute A = L L^T + λI
#     2. Cholesky decomposition of A
#     3. Compute A^(-1) via Cholesky solve (stable)
#     4. If R exists: Apply Woodbury correction
#
#     Args:
#         L_t: [D, D] numpy array, L matrix
#         R_t: [D, rank] numpy array, R matrix (or None)
#         precision_ridge: Ridge regularization for numerical stability
#         woodbury_jitter: Jitter for Woodbury matrix inversion
#
#     Returns:
#         Sigma: [D, D] numpy float64 covariance matrix
#     """
def full_cov_from_factors_np(L_t: torch.Tensor, R_t, precision_ridge=None, woodbury_jitter=None) -> np.ndarray:
    """
    Compute dense covariance Σ = (L L^T + λI + R R^T)^{-1} via Woodbury identity.
    
    Uses Cholesky decomposition + linear solve instead of direct inversion for numerical stability.
    
    Args:
        L_t: Lower triangular factor tensor [D, D] from precision head
        R_t: Low-rank factor tensor [D, rank] or None
        precision_ridge: Ridge parameter λ for A = LL^T + λI (default: PRECISION_RIDGE)
        woodbury_jitter: Jitter ε for Woodbury matrix (default: WOODBURY_JITTER)
    
    Returns:
        Σ: Covariance matrix as (D, D) numpy float64 array
    
    Mathematical details:
        - A = LL^T + λI (ridge regularized precision from Cholesky factor)
        - If R is None: Σ = A^{-1}
        - If R exists: Σ = A^{-1} - A^{-1}R(I + R^T A^{-1}R)^{-1}R^T A^{-1}  (Woodbury)
    """
    if precision_ridge is None:
        precision_ridge = globals().get('PRECISION_RIDGE', 1e-7)
    if woodbury_jitter is None:
        woodbury_jitter = globals().get('WOODBURY_JITTER', 1e-5)
    
    # Convert to numpy float64 for stable linear algebra
    L = L_t.detach().cpu().numpy().astype(np.float64)
    
    # Compute A = LL^T + λI
    A = L @ L.T
    if precision_ridge > 0.0:
        A = A + float(precision_ridge) * np.eye(L.shape[0], dtype=np.float64)
    
    # Cholesky decomposition of A
    Ld = np.linalg.cholesky(A)
    
    # Compute A^{-1} using Cholesky solve (more stable than direct inversion)
    I = np.eye(L.shape[0], dtype=np.float64)
    Ainv = np.linalg.solve(Ld.T, np.linalg.solve(Ld, I))
    
    # If no low-rank component, return A^{-1}
    if (R_t is None) or (R_t.numel() == 0):
        return Ainv
    
    # Apply Woodbury identity for low-rank update: Σ = A^{-1} - A^{-1}R(I + R^T A^{-1}R)^{-1}R^T A^{-1}
    R = R_t.detach().cpu().numpy().astype(np.float64)
    AinvR = np.linalg.solve(Ld.T, np.linalg.solve(Ld, R))
    RtAinvR = R.T @ AinvR
    
    # Form Woodbury matrix M = I + R^T A^{-1}R with jitter for stability
    M = RtAinvR + np.eye(R.shape[1], dtype=np.float64)
    M = 0.5 * (M + M.T) + float(woodbury_jitter) * np.eye(M.shape[0], dtype=np.float64)
    
    # Solve for Woodbury term
    B = np.linalg.solve(M, np.eye(M.shape[0], dtype=np.float64))
    
    # Apply Woodbury correction
    Sigma = Ainv - AinvR @ B @ AinvR.T
    
    # Symmetrize for numerical stability
    Sigma = 0.5 * (Sigma + Sigma.T)
    
    return Sigma


# def blocks_from_cov_np(Sigma, horizon, block_ridge=0.0, eigen_floor=1e-9):
#     """
#     Extract per-timestep 2×2 covariance blocks from full matrix.
#
#     Processing:
#     1. Extract diagonal 2×2 blocks
#     2. Optional block ridge regularization
#     3. Eigenvalue clamping (≥ eigen_floor)
#
#     Args:
#         Sigma: [2*horizon, 2*horizon] full covariance matrix
#         horizon: Planning horizon (T)
#         block_ridge: Optional ridge for blocks
#         eigen_floor: Minimum eigenvalue
#
#     Returns:
#         blocks: [T, 2, 2] per-timestep covariance blocks
def blocks_from_cov_np(Sigma: np.ndarray, horizon: int) -> np.ndarray:
    """
    Extract 2x2 diagonal blocks [T, 2, 2] from a (D, D) covariance matrix.
    
    Applies eigenvalue clamping to ensure positive definiteness of each block.
    
    Args:
        Sigma: Full covariance matrix [D, D] where D = 2*horizon
        horizon: Number of timesteps
    
    Returns:
        blocks: Array of 2x2 covariance blocks [T, 2, 2]
    """
    out = np.zeros((horizon, 2, 2), dtype=np.float64)
    
    for t in range(horizon):
        i = 2 * t
        S = Sigma[i:i+2, i:i+2]
        
        # Optional block ridge (usually 0, controlled by BLOCK_RIDGE)
        block_ridge = globals().get('BLOCK_RIDGE', 0.0)
        if block_ridge > 0.0:
            Sinv = np.linalg.inv(S)
            S = np.linalg.inv(Sinv + float(block_ridge) * np.eye(2, dtype=np.float64))
        
        # Enforce positive definiteness via eigenvalue clamping
        eigen_floor = globals().get('EIGEN_FLOOR', 1e-9)
        w, V = np.linalg.eigh(0.5 * (S + S.T))
        w = np.clip(w, eigen_floor, None)
        S = (V * w) @ V.T
        
        out[t] = S
    
    return out

# def unnormalize_block(S_normalized, t, normalizer):
#     """
#     Transform covariance block from normalized space to raw space.
#
#     Transformation: Σ_raw = D * Σ_norm * D where D = diag(σ_x, σ_y)
#
#     Args:
#         S_normalized: [2, 2] covariance block in normalized space
#         t: Timestep index
#         normalizer: Normalizer dict with 'state' key
#
#     Returns:
#         S_raw: [2, 2] covariance block in raw space
#     """
def unnormalize_block(S_normalized, t, normalizer):
    """
    Transform 2x2 covariance block from normalized to raw space.
    
    Applies the transformation: Σ_raw = D Σ_norm D
    where D = diag(σ_x, σ_y) is the diagonal scaling matrix.
    
    Args:
        S_normalized: 2x2 covariance in normalized space
        t: Timestep index (not used in current implementation, for compatibility)
        normalizer: Normalizer dict with ["state"] containing std values
    
    Returns:
        S_raw: 2x2 covariance in raw (unnormalized) space
    
    Mathematical details:
        - Normalized space: ỹ = (y - μ) / σ
        - Covariance transformation: Σ_y = D Σ_ỹ D
        - Element-wise: Σ_y[i,j] = Σ_ỹ[i,j] * σ_i * σ_j
    """
    # Get standard deviations [σ_x, σ_y]
    std_values = normalizer["state"].std.cpu().numpy()
    
    # Create diagonal scaling matrix D = diag(σ_x, σ_y)
    D_t = np.diag(std_values)
    
    # Apply transformation: Σ_raw = D Σ_norm D
    return D_t @ S_normalized @ D_t
