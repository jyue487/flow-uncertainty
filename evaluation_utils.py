"""
Distribution-Level Evaluation for Precision Head Tubes

This module implements the evaluation framework described in EVALUATION.tex:
unconditional tube generation and distribution-level calibration assessment
via Coverage@95 (path-wise) and Negative Log-Likelihood (NLL).

Key Functions:
- generate_unconditional_tubes: Sample N raw (uncalibrated) tubes from precision head
- compute_pathwise_coverage_single_tube: All-or-nothing coverage per tube
- compute_nll_single_tube: Factorized NLL over timesteps
- evaluate_distribution_level: Full evaluation returning (Cov_mean, Cov_max, NLL_mean, NLL_min)
- plot_distribution_evaluation: Visualize results
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from typing import Tuple, List, Optional, Dict, Callable


# ==============================================================================
# Utility Functions
# ==============================================================================

def chi2_ppf_df2(alpha: float) -> float:
    """Chi-squared quantile for df=2 (2D Gaussian)."""
    return -2.0 * math.log(max(1e-16, 1.0 - float(alpha)))


def mahalanobis_sq(diff: np.ndarray, S: np.ndarray, floor: float = 1e-12) -> float:
    """
    Compute Mahalanobis distance squared: d^T S^{-1} d.

    Args:
        diff: [2] difference vector
        S: [2, 2] covariance matrix
        floor: Eigenvalue floor for numerical stability

    Returns:
        m2: Mahalanobis distance squared
    """
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    w = np.clip(w, floor, None)
    q = V.T @ diff
    return float((q[0] * q[0] / w[0]) + (q[1] * q[1] / w[1]))


# ==============================================================================
# Precision Head Factor Extraction (from notebook)
# ==============================================================================

def solve_band_cholesky(L: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve (LL^T) X = B for X with triangular solves (banded L)."""
    Z = torch.linalg.solve_triangular(L, B, upper=False)
    return torch.linalg.solve_triangular(L.t(), Z, upper=True)


def block_marginal_from_factors(
    L: torch.Tensor,
    R: Optional[torch.Tensor],
    t: int,
    eps_M: float = 1e-6,
    eigen_floor: float = 1e-9,
    precision_ridge: float = 0.0
) -> torch.Tensor:
    """
    Extract 2x2 marginal S_t from Σ = (A + R R^T)^{-1}, with A = L L^T.

    Args:
        L: [D, D] lower triangular banded Cholesky factor
        R: [D, r] low-rank tail (or None)
        t: timestep index
        eps_M: Woodbury jitter
        eigen_floor: PSD eigenvalue floor
        precision_ridge: Optional ridge λ for A <- A + λI

    Returns:
        S_t: [2, 2] covariance block (torch.float64)
    """
    Dloc = L.shape[0]

    # Harmonize R dtype/device
    if (R is not None) and (R.numel() > 0):
        if R.dtype != L.dtype or R.device != L.device:
            R = R.to(dtype=L.dtype, device=L.device)

    # Selection matrix for (x_t, y_t) block
    i, j = 2 * t, 2 * t + 1
    E = torch.zeros(Dloc, 2, dtype=L.dtype, device=L.device)
    E[i, 0] = 1.0
    E[j, 1] = 1.0

    # Two paths: with or without precision ridge
    if precision_ridge and precision_ridge > 0.0:
        A = L @ L.t() + precision_ridge * torch.eye(Dloc, dtype=L.dtype, device=L.device)
        Ld = torch.linalg.cholesky(A)
        Z = torch.cholesky_solve(E, Ld)  # (A+λI)^{-1} E
        if (R is None) or (R.numel() == 0):
            X = Z
        else:
            AinvR = torch.cholesky_solve(R, Ld)
            RtZ = R.t() @ Z
            M = torch.eye(R.shape[1], dtype=L.dtype, device=L.device) + R.t() @ AinvR
            M = 0.5 * (M + M.t()) + eps_M * torch.eye(M.shape[0], dtype=L.dtype, device=L.device)
            W = torch.linalg.solve(M, RtZ)
            X = Z - AinvR @ W
    else:
        Z = solve_band_cholesky(L, E)  # A^{-1} E
        if (R is None) or (R.numel() == 0):
            X = Z
        else:
            AinvR = solve_band_cholesky(L, R)
            RtZ = R.t() @ Z
            M = torch.eye(R.shape[1], dtype=L.dtype, device=L.device) + R.t() @ AinvR
            M = 0.5 * (M + M.t()) + eps_M * torch.eye(M.shape[0], dtype=L.dtype, device=L.device)
            W = torch.linalg.solve(M, RtZ)
            X = Z - AinvR @ W

    # Extract 2x2 marginal
    S = X[i:j+1, :]
    S = 0.5 * (S + S.t())

    # PSD clamp in numpy for robustness
    S_np = S.detach().cpu().numpy()
    w, V = np.linalg.eigh(0.5 * (S_np + S_np.T))
    w = np.clip(w, eigen_floor, None)
    S_np = (V * w) @ V.T

    return torch.from_numpy(S_np).to(dtype=L.dtype, device=L.device)


def _module_dtype_device(mod: torch.nn.Module) -> Tuple[torch.dtype, torch.device]:
    """Robustly get dtype/device from a module."""
    for p in mod.parameters():
        return p.dtype, p.device
    for b in mod.buffers():
        return b.dtype, b.device
    return torch.float32, torch.device('cpu')


def _head_factors_eval(
    head: torch.nn.Module,
    Y_white_numpy: np.ndarray
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
    """
    Evaluate precision head to get factors (L, R) in float64.

    Args:
        head: Trained PrecisionHead module
        Y_white_numpy: [B, D] whitened trajectories (numpy)

    Returns:
        L_list: List of B lower-triangular factors (torch.float64)
        R_list: List of B low-rank tails or None (torch.float64)
    """
    dtype, device = _module_dtype_device(head)
    Y_t = torch.from_numpy(np.asarray(Y_white_numpy)).to(device=device, dtype=dtype)
    with torch.no_grad():
        L_list, R_list = head(Y_t)

    # Convert to float64 for stable linear algebra
    L_list = [L.to(torch.float64) for L in L_list]
    R_list = [(R.to(torch.float64) if (R is not None) else None) for R in R_list]

    return L_list, R_list


def unwhiten_block(S_white: np.ndarray, t: int, W_VEC: np.ndarray) -> np.ndarray:
    """
    Unwhiten a 2x2 covariance block.

    Args:
        S_white: [2, 2] covariance in whitened space
        t: timestep index
        W_VEC: [D] whitening vector (1/std per dimension)

    Returns:
        S: [2, 2] covariance in original space
    """
    D_t = np.diag(1.0 / W_VEC[2*t:2*t+2])
    return D_t @ S_white @ D_t


# ==============================================================================
# Unconditional Tube Generation
# ==============================================================================

def generate_unconditional_tubes(
    head: torch.nn.Module,
    X_train: np.ndarray,
    N_tubes: int,
    T: int,
    whiten: bool = False,
    W_VEC: Optional[np.ndarray] = None,
    precision_ridge: float = 1e-7,
    woodbury_jitter: float = 1e-5,
    eigen_floor: float = 1e-9,
    batch_size: int = 256,
    s_vec: Optional[np.ndarray] = None,
    gamma: Optional[float] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate N unconditional tubes from precision head with optional calibration.

    Process:
    1. Sample N trajectories from training data
    2. Pass through precision head to get (L, R) factors
    3. Extract per-timestep 2x2 covariances S_t (unwhitened if needed)
    4. Optionally apply calibration: S_calibrated = (gamma * s_t)^2 * S_raw

    Args:
        head: Trained PrecisionHead module
        X_train: [N_train, D] training trajectories (flattened)
        N_tubes: Number of tubes to generate
        T: Number of timesteps
        whiten: Whether whitening is used
        W_VEC: [D] whitening vector (required if whiten=True)
        precision_ridge: Ridge parameter for numerical stability
        woodbury_jitter: Jitter for Woodbury identity
        eigen_floor: Eigenvalue floor for PSD clamping
        batch_size: Batch size for head evaluation
        s_vec: Optional [T] per-horizon calibration scales (default: None = no calibration)
        gamma: Optional scalar path multiplier (default: None = no calibration)

    Returns:
        tubes: List of N tuples (mu, S_blocks)
            - mu: [T, 2] mean trajectory
            - S_blocks: [T, 2, 2] per-timestep covariances (calibrated if s_vec/gamma provided)
    """
    if whiten and W_VEC is None:
        raise ValueError("W_VEC required when whiten=True")

    # Sample N trajectories from training data
    indices = np.random.choice(X_train.shape[0], size=N_tubes, replace=False)
    X_sampled = X_train[indices]  # [N_tubes, D]

    # Whiten if needed
    if whiten:
        X_sampled_white = X_sampled * W_VEC
    else:
        X_sampled_white = X_sampled

    # Evaluate head in batches
    L_all, R_all = [], []
    for i0 in range(0, N_tubes, batch_size):
        batch = X_sampled_white[i0:i0+batch_size]
        L_batch, R_batch = _head_factors_eval(head, batch)
        L_all.extend(L_batch)
        R_all.extend(R_batch)

    # Extract covariances for each tube
    tubes = []
    for k in range(N_tubes):
        mu = X_sampled[k].reshape(T, 2)  # Original (unwhitened) mean
        S_blocks = np.zeros((T, 2, 2), dtype=np.float64)
# TODO: understand why the ipynb one produce unstable ellipse, but this one not
        for t in range(T):
            # Get whitened covariance block
            S_w = block_marginal_from_factors(
                L_all[k], R_all[k], t,
                eps_M=woodbury_jitter,
                eigen_floor=eigen_floor,
                precision_ridge=precision_ridge
            ).cpu().numpy()

            # Unwhiten if needed
            if whiten:
                S_t = unwhiten_block(S_w, t, W_VEC)
            else:
                S_t = S_w

            # Apply calibration if provided
            if s_vec is not None and gamma is not None:
                # S_calibrated = (gamma * s_t)^2 * S_raw
                scale_factor = (gamma * s_vec[t]) ** 2
                S_blocks[t] = scale_factor * S_t
            else:
                S_blocks[t] = S_t

        tubes.append((mu, S_blocks))

    return tubes


# ==============================================================================
# Coverage and NLL Computation
# ==============================================================================

def compute_pathwise_coverage_single_tube(
    tube: Tuple[np.ndarray, np.ndarray],
    X_test: np.ndarray,
    T: int,
    alpha: float = 0.95
) -> float:
    """
    Compute path-wise coverage for ONE tube against test trajectories.

    Coverage = (1/M) * |{m : ∀t, (y_{m,t} - μ_t)^T S_t^{-1} (y_{m,t} - μ_t) ≤ τ}|

    A test trajectory is covered IF AND ONLY IF it stays inside the tube
    at ALL timesteps (all-or-nothing).

    Args:
        tube: (mu, S_blocks)
            - mu: [T, 2] mean trajectory
            - S_blocks: [T, 2, 2] per-timestep covariances
        X_test: [M, D] test trajectories (flattened)
        T: Number of timesteps
        alpha: Coverage level (default 0.95)

    Returns:
        coverage: float in [0, 1]
    """
    mu, S_blocks = tube
    tau = chi2_ppf_df2(alpha)
    M = X_test.shape[0]

    covered_count = 0
    for m in range(M):
        y_traj = X_test[m].reshape(T, 2)

        # Check if trajectory is covered at ALL timesteps
        covered = True
        for t in range(T):
            diff = y_traj[t] - mu[t]
            m2 = mahalanobis_sq(diff, S_blocks[t])
            if m2 > tau:
                covered = False
                break

        if covered:
            covered_count += 1

    return float(covered_count) / float(M)


def compute_nll_single_tube(
    tube: Tuple[np.ndarray, np.ndarray],
    X_test: np.ndarray,
    T: int
) -> float:
    """
    Compute NLL for ONE tube against test trajectories.

    NLL = (1/M) Σ_m Σ_t [½log|2πS_t| + ½(y_{m,t} - μ_t)^T S_t^{-1} (y_{m,t} - μ_t)]

    Args:
        tube: (mu, S_blocks)
            - mu: [T, 2] mean trajectory
            - S_blocks: [T, 2, 2] per-timestep covariances
        X_test: [M, D] test trajectories (flattened)
        T: Number of timesteps

    Returns:
        nll_avg: float (average NLL per trajectory)
    """
    mu, S_blocks = tube
    M = X_test.shape[0]

    nll_sum = 0.0
    for m in range(M):
        y_traj = X_test[m].reshape(T, 2)

        nll_m = 0.0
        for t in range(T):
            # Log-det term: ½ log|2πS_t|
            S = S_blocks[t]
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                # Degenerate covariance, use large penalty
                logdet = 20.0
            log_2pi_S = logdet + 2.0 * math.log(2.0 * math.pi)

            # Mahalanobis term
            diff = y_traj[t] - mu[t]
            m2 = mahalanobis_sq(diff, S)

            nll_m += 0.5 * log_2pi_S + 0.5 * m2

        nll_sum += nll_m

    return nll_sum / float(M)


# ==============================================================================
# Sharpness Metrics (comparing predicted vs ground-truth covariances)
# ==============================================================================

def compute_sharpness_metrics(
    tube: Tuple[np.ndarray, np.ndarray],
    S_true: np.ndarray,
    T: int
) -> Dict:
    """
    Compute sharpness metrics comparing predicted vs ground-truth covariances.

    Metrics:
    - RS_t = sqrt(det(S_pred_t)) / sqrt(det(S_true_t)) per timestep
    - RS_mean = (1/T) * sum_t RS_t
    - PTV_ratio = sum_t sqrt(det(S_pred_t)) / sum_t sqrt(det(S_true_t))

    Args:
        tube: (mu, S_blocks)
            - mu: [T, 2] mean trajectory
            - S_blocks: [T, 2, 2] predicted covariances
        S_true: [T, 2, 2] ground-truth covariances
        T: Number of timesteps

    Returns:
        metrics: dict with keys:
            - 'rs_per_t': [T] relative size per timestep
            - 'rs_mean': mean relative size
            - 'ptv_ratio': path total volume ratio
            - 'mean_area_pred': mean ellipse area (predicted)
            - 'mean_area_true': mean ellipse area (ground truth)
    """
    mu, S_pred = tube

    rs_per_t = np.zeros(T, dtype=np.float64)
    sqrt_det_pred = np.zeros(T, dtype=np.float64)
    sqrt_det_true = np.zeros(T, dtype=np.float64)

    for t in range(T):
        # Compute sqrt(det(S)) using slogdet for numerical stability
        sign_pred, logdet_pred = np.linalg.slogdet(S_pred[t])
        sign_true, logdet_true = np.linalg.slogdet(S_true[t])

        # Handle degenerate cases
        if sign_pred <= 0 or sign_true <= 0:
            # Degenerate covariance, use fallback
            sqrt_det_pred[t] = 1e-12
            sqrt_det_true[t] = 1e-12
            rs_per_t[t] = 1.0
        else:
            sqrt_det_pred[t] = np.exp(0.5 * logdet_pred)
            sqrt_det_true[t] = np.exp(0.5 * logdet_true)
            rs_per_t[t] = sqrt_det_pred[t] / (sqrt_det_true[t] + 1e-16)

    # Aggregate metrics
    rs_mean = float(np.mean(rs_per_t))
    ptv_ratio = float(np.sum(sqrt_det_pred) / (np.sum(sqrt_det_true) + 1e-16))
    mean_area_pred = float(np.mean(sqrt_det_pred))
    mean_area_true = float(np.mean(sqrt_det_true))

    return {
        'rs_per_t': rs_per_t,
        'rs_mean': rs_mean,
        'ptv_ratio': ptv_ratio,
        'mean_area_pred': mean_area_pred,
        'mean_area_true': mean_area_true
    }


# ==============================================================================
# Full Distribution-Level Evaluation
# ==============================================================================

def evaluate_distribution_level(
    head: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    T: int,
    N_tubes: int = 50,
    alpha: float = 0.95,
    whiten: bool = False,
    W_VEC: Optional[np.ndarray] = None,
    precision_ridge: float = 1e-7,
    woodbury_jitter: float = 1e-5,
    eigen_floor: float = 1e-9,
    batch_size: int = 256,
    verbose: bool = True,
    s_vec: Optional[np.ndarray] = None,
    gamma: Optional[float] = None,
    S_true: Optional[np.ndarray] = None
) -> Dict:
    """
    Full distribution-level evaluation of precision head with optional calibration.

    Procedure:
    1. Generate N unconditional tubes from training data (with optional calibration)
    2. Compute path-wise coverage and NLL for each tube on test data
    3. Optionally compute sharpness metrics if ground truth covariances provided
    4. Aggregate: (Cov_mean, Cov_max, NLL_mean, NLL_min, RS_mean, PTV_ratio)

    Args:
        head: Trained PrecisionHead module
        X_train: [N_train, D] training trajectories
        X_test: [M, D] test trajectories
        T: Number of timesteps
        N_tubes: Number of tubes to generate
        alpha: Coverage level (default 0.95)
        whiten: Whether whitening is used
        W_VEC: Whitening vector (required if whiten=True)
        precision_ridge: Ridge for precision matrix
        woodbury_jitter: Jitter for Woodbury
        eigen_floor: Eigenvalue floor
        batch_size: Batch size for head evaluation
        verbose: Print progress
        s_vec: Optional [T] per-horizon calibration scales (None = no calibration)
        gamma: Optional scalar path multiplier (None = no calibration)
        S_true: Optional [T, 2, 2] ground-truth covariances for sharpness metrics

    Returns:
        results: dict with keys:
            - 'cov_mean': mean coverage across N tubes
            - 'cov_max': max coverage across N tubes
            - 'nll_mean': mean NLL across N tubes
            - 'nll_min': min NLL across N tubes
            - 'cov_per_tube': [N] coverage for each tube
            - 'nll_per_tube': [N] NLL for each tube
            - 'tubes': List of (mu, S_blocks) for visualization
            - 'alpha': coverage level used
            - 'rs_mean': (if S_true provided) mean relative size
            - 'ptv_ratio': (if S_true provided) path total volume ratio
            - 'rs_per_tube': (if S_true provided) [N] RS_mean for each tube
    """
    if verbose:
        calib_str = ""
        if s_vec is not None and gamma is not None:
            calib_str = " (with calibration)"
        print(f"[EVAL] Generating {N_tubes} unconditional tubes{calib_str}...")

    # Generate tubes (with optional calibration)
    tubes = generate_unconditional_tubes(
        head, X_train, N_tubes, T,
        whiten=whiten, W_VEC=W_VEC,
        precision_ridge=precision_ridge,
        woodbury_jitter=woodbury_jitter,
        eigen_floor=eigen_floor,
        batch_size=batch_size,
        s_vec=s_vec,
        gamma=gamma
    )

    if verbose:
        print(f"[EVAL] Computing coverage and NLL for {N_tubes} tubes on {X_test.shape[0]} test trajectories...")

    # Compute metrics per tube
    cov_per_tube = np.zeros(N_tubes, dtype=np.float64)
    nll_per_tube = np.zeros(N_tubes, dtype=np.float64)
    rs_per_tube = np.zeros(N_tubes, dtype=np.float64) if S_true is not None else None

    for i in range(N_tubes):
        cov_per_tube[i] = compute_pathwise_coverage_single_tube(tubes[i], X_test, T, alpha)
        nll_per_tube[i] = compute_nll_single_tube(tubes[i], X_test, T)

        # Compute sharpness metrics if ground truth provided
        if S_true is not None:
            sharpness_metrics = compute_sharpness_metrics(tubes[i], S_true, T)
            rs_per_tube[i] = sharpness_metrics['rs_mean']

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{N_tubes} tubes")

    # Aggregate reliability metrics
    cov_mean = float(np.mean(cov_per_tube))
    cov_max = float(np.max(cov_per_tube))
    nll_mean = float(np.mean(nll_per_tube))
    nll_min = float(np.min(nll_per_tube))

    # Aggregate sharpness metrics if available
    results = {
        'cov_mean': cov_mean,
        'cov_max': cov_max,
        'nll_mean': nll_mean,
        'nll_min': nll_min,
        'cov_per_tube': cov_per_tube,
        'nll_per_tube': nll_per_tube,
        'tubes': tubes,
        'alpha': alpha
    }

    if S_true is not None:
        rs_mean = float(np.mean(rs_per_tube))
        # Compute PTV-ratio across all tubes
        ptv_ratios = []
        for i in range(N_tubes):
            sharpness_metrics = compute_sharpness_metrics(tubes[i], S_true, T)
            ptv_ratios.append(sharpness_metrics['ptv_ratio'])
        ptv_ratio_mean = float(np.mean(ptv_ratios))

        results['rs_mean'] = rs_mean
        results['ptv_ratio'] = ptv_ratio_mean
        results['rs_per_tube'] = rs_per_tube

    if verbose:
        print(f"\n[RESULTS] Distribution-Level Evaluation @ α={alpha:.0%}")
        print(f"  Coverage (mean): {100*cov_mean:.2f}%")
        print(f"  Coverage (max):  {100*cov_max:.2f}%")
        print(f"  NLL (mean):      {nll_mean:.4f}")
        print(f"  NLL (min):       {nll_min:.4f}")
        if S_true is not None:
            print(f"  RS_mean:         {results['rs_mean']:.4f}")
            print(f"  PTV-ratio:       {results['ptv_ratio']:.4f}")

    return results


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_distribution_evaluation(
    results: Dict,
    figsize: Tuple[float, float] = (12, 5)
):
    """
    Visualize distribution-level evaluation results.

    Shows:
    - Left: Histogram of coverage per tube
    - Right: Histogram of NLL per tube (or RS_mean if available)

    Args:
        results: Output from evaluate_distribution_level()
        figsize: Figure size
    """
    alpha = results.get('alpha', 0.95)
    nll_per_tube = results['nll_per_tube']
    has_sharpness = 'rs_mean' in results

    # Adjust layout based on available metrics
    n_plots = 3 if has_sharpness else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 2:
        axes = list(axes)

    # Left: Coverage histogram
    ax = axes[0]
    ax.axvline(100 * results['cov_mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {100*results['cov_mean']:.2f}%")
    ax.axvline(100 * alpha, color='gray', linestyle=':', linewidth=2, label=f"Nominal: {100*alpha:.0f}%")
    ax.set_xlabel('Coverage (%)', fontsize=11)
    ax.set_ylabel('Number of tubes', fontsize=11)
    ax.set_title(f'Path-wise Coverage Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Middle: NLL histogram
    ax = axes[1]
    ax.hist(nll_per_tube, bins=20, alpha=0.7, color='C1', edgecolor='black')
    ax.axvline(results['nll_mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['nll_mean']:.4f}")
    ax.axvline(results['nll_min'], color='green', linestyle='--', linewidth=2, label=f"Min: {results['nll_min']:.4f}")
    ax.set_xlabel('NLL', fontsize=11)
    ax.set_ylabel('Number of tubes', fontsize=11)
    ax.set_title('Negative Log-Likelihood Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Sharpness histogram (if available)
    if has_sharpness:
        ax = axes[2]
        rs_per_tube = results['rs_per_tube']
        ax.hist(rs_per_tube, bins=20, alpha=0.7, color='C2', edgecolor='black')
        ax.axvline(results['rs_mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['rs_mean']:.4f}")
        ax.axvline(1.0, color='gray', linestyle=':', linewidth=2, label='Perfect calibration')
        ax.set_xlabel('RS_mean (Relative Size)', fontsize=11)
        ax.set_ylabel('Number of tubes', fontsize=11)
        ax.set_title('Sharpness Distribution', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_tube_with_test_samples(
    tube: Tuple[np.ndarray, np.ndarray],
    X_test: np.ndarray,
    T: int,
    alpha: float = 0.95,
    n_show: int = 30,
    ellipse_stride: int = 1,
    figsize: Tuple[float, float] = (10, 7),
    title: Optional[str] = None
):
    """
    Visualize one tube with test trajectory samples overlaid.

    Shows:
    - Tube mean trajectory
    - Per-timestep α-ellipses (raw, no calibration)
    - Test trajectories colored by coverage status:
        - Green: covered (inside ALL ellipses)
        - Red: not covered (outside at least one ellipse)

    Args:
        tube: (mu, S_blocks)
        X_test: [M, D] test trajectories
        T: Number of timesteps
        alpha: Coverage level
        n_show: Max number of test trajectories to show
        ellipse_stride: Draw ellipse every N timesteps
        figsize: Figure size
        title: Plot title
    """
    mu, S_blocks = tube
    tau = chi2_ppf_df2(alpha)
    rad = math.sqrt(tau)

    M = X_test.shape[0]
    n_show = min(n_show, M)

    # Determine coverage status for displayed trajectories
    coverage_status = []
    for m in range(n_show):
        y_traj = X_test[m].reshape(T, 2)
        covered = True
        for t in range(T):
            diff = y_traj[t] - mu[t]
            m2 = mahalanobis_sq(diff, S_blocks[t])
            if m2 > tau:
                covered = False
                break
        coverage_status.append(covered)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot test trajectories
    for m in range(n_show):
        y_traj = X_test[m].reshape(T, 2)
        color = 'green' if coverage_status[m] else 'red'
        alpha_traj = 0.6 if coverage_status[m] else 0.4
        label = 'Covered' if (m == 0 and coverage_status[m]) else ('Not covered' if (m == 0 and not coverage_status[m]) else None)
        ax.plot(y_traj[:, 0], y_traj[:, 1], '-', color=color, alpha=alpha_traj, lw=0.8, label=label)

    # Plot tube mean
    ax.plot(mu[:, 0], mu[:, 1], 'k-', lw=2.5, label='Tube mean', zorder=10)
    ax.plot(mu[0, 0], mu[0, 1], 'go', ms=8, zorder=11)
    ax.plot(mu[-1, 0], mu[-1, 1], 'rs', ms=8, zorder=11)

    # Draw ellipses
    for t in range(0, T, ellipse_stride):
        S = S_blocks[t]

        # Eigendecomposition
        w, V = np.linalg.eigh(0.5 * (S + S.T))
        w = np.clip(w, 1e-12, None)

        # Sort for consistency
        order = np.argsort(w)[::-1]
        w = w[order]
        V = V[:, order]

        width = 2 * rad * np.sqrt(w[0])
        height = 2 * rad * np.sqrt(w[1])
        angle = math.degrees(math.atan2(V[1, 0], V[0, 0]))

        ell = Ellipse(
            xy=mu[t],
            width=width,
            height=height,
            angle=angle,
            facecolor='none',
            edgecolor='blue',
            linewidth=1.2,
            alpha=0.7,
            label=f'{int(100*alpha)}% ellipse' if t == 0 else None
        )
        ax.add_patch(ell)

    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)

    if title is None:
        covered_count = sum(coverage_status)
        title = f'Tube with Test Samples | {covered_count}/{n_show} covered @ α={int(100*alpha)}%'
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Risk-Size Pareto Curve Evaluation
# ==============================================================================

def evaluate_risk_size_curve(
    head: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    S_true: np.ndarray,
    T: int,
    s_table: np.ndarray,
    gamma_vec: np.ndarray,
    alphas: np.ndarray,
    N_scenes: int = 50,
    whiten: bool = False,
    W_VEC: Optional[np.ndarray] = None,
    precision_ridge: float = 1e-7,
    woodbury_jitter: float = 1e-5,
    eigen_floor: float = 1e-9,
    batch_size: int = 256,
    verbose: bool = True
) -> Dict:
    """
    Evaluate Risk-Size Pareto curve by sweeping alpha values.

    For each alpha level:
    1. Extract corresponding s_vec and gamma from calibration tables
    2. Generate calibrated tubes
    3. Compute empirical coverage and sharpness metrics

    Args:
        head: Trained PrecisionHead module
        X_train: [N_train, D] training trajectories
        X_test: [M, D] test trajectories
        S_true: [T, 2, 2] ground-truth covariances
        T: Number of timesteps
        s_table: [T, n_alphas] per-horizon calibration scales
        gamma_vec: [n_alphas] path multipliers
        alphas: [n_alphas] alpha levels to evaluate
        N_scenes: Number of tubes to generate per alpha
        whiten: Whether whitening is used
        W_VEC: Whitening vector
        precision_ridge: Ridge parameter
        woodbury_jitter: Jitter for Woodbury
        eigen_floor: Eigenvalue floor
        batch_size: Batch size for head evaluation
        verbose: Print progress

    Returns:
        results: dict with keys:
            - 'alphas': alpha values evaluated
            - 'coverage': [n_alphas] empirical coverage at each alpha
            - 'rs_mean': [n_alphas] mean relative size at each alpha
            - 'ptv_ratio': [n_alphas] path total volume ratio at each alpha
            - 'mean_area': [n_alphas] mean ellipse area at each alpha
    """
    n_alphas = len(alphas)
    coverage_results = np.zeros(n_alphas, dtype=np.float64)
    rs_mean_results = np.zeros(n_alphas, dtype=np.float64)
    ptv_ratio_results = np.zeros(n_alphas, dtype=np.float64)
    mean_area_results = np.zeros(n_alphas, dtype=np.float64)

    if verbose:
        print(f"[RISK-SIZE CURVE] Evaluating {n_alphas} alpha levels...")

    for i, alpha in enumerate(alphas):
        if verbose:
            print(f"\n[Alpha {i+1}/{n_alphas}] α={alpha:.2f}")

        # Get calibration parameters for this alpha
        s_vec = s_table[:, i]
        gamma = gamma_vec[i]

        # Evaluate at this calibration level
        results_alpha = evaluate_distribution_level(
            head=head,
            X_train=X_train,
            X_test=X_test,
            T=T,
            N_SCENES=N_scenes,
            alpha=alpha,
            whiten=whiten,
            W_VEC=W_VEC,
            precision_ridge=precision_ridge,
            woodbury_jitter=woodbury_jitter,
            eigen_floor=eigen_floor,
            batch_size=batch_size,
            verbose=False,  # Suppress per-alpha verbose output
            s_vec=s_vec,
            gamma=gamma,
            S_true=S_true
        )

        # Store results
        coverage_results[i] = results_alpha['cov_mean']
        rs_mean_results[i] = results_alpha['rs_mean']
        ptv_ratio_results[i] = results_alpha['ptv_ratio']

        # Compute mean area across tubes
        mean_areas = []
        for tube in results_alpha['tubes']:
            sharpness = compute_sharpness_metrics(tube, S_true, T)
            mean_areas.append(sharpness['mean_area_pred'])
        mean_area_results[i] = float(np.mean(mean_areas))

        if verbose:
            print(f"  Coverage: {100*coverage_results[i]:.2f}%, "
                  f"RS_mean: {rs_mean_results[i]:.4f}, "
                  f"PTV-ratio: {ptv_ratio_results[i]:.4f}")

    if verbose:
        print(f"\n[RISK-SIZE CURVE] Evaluation complete")

    return {
        'alphas': alphas,
        'coverage': coverage_results,
        'rs_mean': rs_mean_results,
        'ptv_ratio': ptv_ratio_results,
        'mean_area': mean_area_results
    }


def plot_risk_size_curve(
    risk_size_results: Dict,
    figsize: Tuple[float, float] = (14, 5)
):
    """
    Visualize Risk-Size Pareto curve.

    Shows three panels:
    - Left: Coverage vs Alpha (calibration reliability check)
    - Middle: Coverage vs RS_mean (Pareto frontier)
    - Right: Coverage vs PTV-ratio (alternative Pareto view)

    Args:
        risk_size_results: Output from evaluate_risk_size_curve()
        figsize: Figure size
    """
    alphas = risk_size_results['alphas']
    coverage = risk_size_results['coverage']
    rs_mean = risk_size_results['rs_mean']
    ptv_ratio = risk_size_results['ptv_ratio']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Left: Coverage vs Alpha (calibration check)
    ax = axes[0]
    ax.plot(alphas, coverage, 'o-', lw=2, ms=8, color='C0', label='Empirical')
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=2, label='Perfect calibration')
    ax.set_xlabel('Nominal Coverage (α)', fontsize=11)
    ax.set_ylabel('Empirical Coverage', fontsize=11)
    ax.set_title('Calibration Reliability', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([alphas.min() - 0.05, alphas.max() + 0.05])
    ax.set_ylim([max(0, coverage.min() - 0.1), min(1, coverage.max() + 0.1)])

    # Middle: Coverage vs RS_mean (Pareto frontier)
    ax = axes[1]
    ax.plot(rs_mean, coverage, 'o-', lw=2, ms=8, color='C1')
    for i, alpha in enumerate(alphas):
        ax.annotate(f'{alpha:.2f}', (rs_mean[i], coverage[i]),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.axvline(1.0, ls=':', color='gray', lw=1.5, label='Perfect sharpness')
    ax.set_xlabel('RS_mean (Relative Size)', fontsize=11)
    ax.set_ylabel('Empirical Coverage', fontsize=11)
    ax.set_title('Risk-Size Pareto Frontier (RS_mean)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Coverage vs PTV-ratio (alternative view)
    ax = axes[2]
    ax.plot(ptv_ratio, coverage, 'o-', lw=2, ms=8, color='C2')
    for i, alpha in enumerate(alphas):
        ax.annotate(f'{alpha:.2f}', (ptv_ratio[i], coverage[i]),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.axvline(1.0, ls=':', color='gray', lw=1.5, label='Perfect sharpness')
    ax.set_xlabel('PTV-ratio (Path Total Volume)', fontsize=11)
    ax.set_ylabel('Empirical Coverage', fontsize=11)
    ax.set_title('Risk-Size Pareto Frontier (PTV)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
