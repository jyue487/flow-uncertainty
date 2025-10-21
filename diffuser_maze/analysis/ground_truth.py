"""
Ground truth Hessian analysis.

Copy from notebook:
- Cell 1: analyze_ground_truth_hessian() function
- Cell 0: Ground truth comparison script (can be adapted as standalone function)

This module estimates ground truth Hessian spectrum from diffuser's
score function using Monte Carlo approximation with random probe vectors.

Dependencies:
- torch
- numpy
- diffuser_maze.training.hvp.compute_hvp

Usage:
    from diffuser_maze.analysis.ground_truth import analyze_ground_truth_hessian

    results = analyze_ground_truth_hessian(
        diffuser_model=diffuser,
        trajectory_norm=traj,
        num_probes=512
    )
"""

# TODO: Copy from Cell 1 - analyze_ground_truth_hessian()
import torch
import numpy as np
from ..training.hvp import compute_hvp
#
# def analyze_ground_truth_hessian(diffuser_model, trajectory_norm, num_probes=512):
#     """
#     Estimate ground truth Hessian spectrum from diffuser's score function.
#
#     Method: Monte Carlo approximation using random probe vectors
#     - Generate random probe vectors
#     - Compute HVPs for each probe
#     - Analyze eigenvalue statistics
#     - Compute Rayleigh quotients
#
#     Args:
#         diffuser_model: Diffusion model with compute_terminal_score()
#         trajectory_norm: [B, HORIZON, 4] normalized trajectory
#         num_probes: Number of random probes for estimation
#
#     Returns:
#         results: dict with eigenvalue statistics and analysis
#     """
#     ...



def analyze_ground_truth_hessian(diffuser_model, trajectory_norm, num_probes=512, normalizer=None):
    """
    Analyze the ground truth Hessian H = -∇²log p(y) from diffuser's score function.
    
    Uses random probe vectors to approximate the Hessian eigenvalue spectrum via:
    - Compute HVPs: h_i = -∇_y (s_θ(y)^T u_i) for random u_i
    - Build approximate Hessian via Monte Carlo: H ≈ (1/n) Σ h_i u_i^T
    
    Args:
        diffuser_model: Trained diffusion model
        trajectory_norm: [1, T, 4] single trajectory (normalized positions + raw actions)
        num_probes: Number of random probe vectors to estimate Hessian spectrum
    
    Returns:
        analysis: Dict with ground truth Hessian statistics
    """
    print("="*80)
    print("GROUND TRUTH HESSIAN ANALYSIS FROM DIFFUSER")
    print("="*80)
    
    batch_size = trajectory_norm.shape[0]
    T = trajectory_norm.shape[1]
    dim = 2 * T  # Position dimensions only
    
    print(f"\nTrajectory shape: {trajectory_norm.shape}")
    print(f"Hessian dimension: {dim} × {dim} (positions only)")
    print(f"Using {num_probes} random probes for eigenvalue estimation...\n")
    
    # Storage for HVP results
    hvp_list = []
    probe_list = []
    
    # Generate random probe vectors and compute HVPs
    for i in range(num_probes):
        # Random probe vector (Rademacher: ±1)
        probe = torch.randn(batch_size, dim, device=trajectory_norm.device)
        probe = probe / torch.norm(probe, dim=1, keepdim=True)  # Normalize
        
        # Compute HVP: h = -∇_y (s_θ(y)^T u)
        hvp, _ = compute_hvp(
            diffuser_model,
            trajectory_norm,
            probe,
            use_noised_trajectory=False
        )
        
        hvp_list.append(hvp[0].cpu().numpy())  # [dim]
        probe_list.append(probe[0].cpu().numpy())  # [dim]
        
        if (i + 1) % 100 == 0:
            print(f"  Computed {i+1}/{num_probes} HVPs...")
    
    # Convert to arrays
    H_probes = np.array(hvp_list)  # [num_probes, dim]
    U_probes = np.array(probe_list)  # [num_probes, dim]
    
    print(f"\nHVP matrix shape: {H_probes.shape}")
    
    # Analyze HVP statistics
    print(f"\n1. HVP Statistics:")
    hvp_norms = np.linalg.norm(H_probes, axis=1)
    print(f"   HVP norms - Mean: {hvp_norms.mean():.3e}, Std: {hvp_norms.std():.3e}")
    print(f"   HVP norms - Range: [{hvp_norms.min():.3e}, {hvp_norms.max():.3e}]")
    
    # Estimate Hessian eigenvalues via Lanczos/power iteration approximation
    # Method: H ≈ (H_probes^T @ H_probes) / num_probes gives eigenvalue estimates
    print(f"\n2. Eigenvalue Estimation via HVP Monte Carlo:")
    
    # Approximate H via outer products: H ≈ (1/n) Σ h_i u_i^T
    # For eigenvalues, we can use: λ ≈ u^T H u for random u
    # Or build gram matrix and use its eigenvalues as proxy
    
    # Method 1: Rayleigh quotients
    rayleigh_quotients = []
    for i in range(min(num_probes, 100)):
        u = U_probes[i]
        Hu = H_probes[i]
        rq = (Hu @ u) / (u @ u + 1e-16)  # λ ≈ u^T H u / u^T u
        rayleigh_quotients.append(rq)
    
    rayleigh_quotients = np.array(rayleigh_quotients)
    print(f"   Rayleigh quotients (λ ≈ u^T H u):")
    print(f"     Range: [{rayleigh_quotients.min():.3e}, {rayleigh_quotients.max():.3e}]")
    print(f"     Mean: {rayleigh_quotients.mean():.3e}, Std: {rayleigh_quotients.std():.3e}")
    
    # Method 2: Build approximate Hessian and compute eigenvalues
    # H ≈ (1/n) Σ h_i u_i^T  (outer products)
    # For small dim, we can build this explicitly
    if dim <= 512:  # Only for manageable sizes
        print(f"\n   Building approximate Hessian matrix...")
        H_approx = np.zeros((dim, dim), dtype=np.float64)
        for i in range(num_probes):
            H_approx += np.outer(H_probes[i], U_probes[i])
        H_approx = H_approx / num_probes
        
        # Symmetrize
        H_approx = 0.5 * (H_approx + H_approx.T)
        
        # Compute eigenvalues
        print(f"   Computing eigenvalues of approximate Hessian...")
        H_eigs = np.linalg.eigvalsh(H_approx)
        
        print(f"\n3. Approximate Hessian Eigenvalue Spectrum:")
        print(f"   Range: [{H_eigs.min():.3e}, {H_eigs.max():.3e}]")
        print(f"   Mean: {H_eigs.mean():.3e}, Median: {np.median(H_eigs):.3e}")
        print(f"   Std: {H_eigs.std():.3e}")
        print(f"   Condition number: {H_eigs.max() / (np.abs(H_eigs).min() + 1e-16):.3e}")
        
        # Check for negative eigenvalues
        num_negative = np.sum(H_eigs < 0)
        num_positive = np.sum(H_eigs > 0)
        print(f"\n   Sign distribution:")
        print(f"     Positive eigenvalues: {num_positive}/{len(H_eigs)} ({100*num_positive/len(H_eigs):.1f}%)")
        print(f"     Negative eigenvalues: {num_negative}/{len(H_eigs)} ({100*num_negative/len(H_eigs):.1f}%)")
        
        if num_negative > 0:
            print(f"   ⚠️  WARNING: Hessian has {num_negative} negative eigenvalues!")
            print(f"      H = -∇²log p should be positive semi-definite for log-concave p")
            print(f"      Negative eigenvalues indicate: score function is not gradient of log p")
            print(f"      Or: numerical errors in HVP computation")
        
        # Expected covariance eigenvalues (if Hessian is precision)
        print(f"\n4. Implied Covariance Eigenvalues (if H is used as precision):")
        # Σ = H^{-1}, so λ_Σ = 1/λ_H for positive eigenvalues
        positive_eigs = H_eigs[H_eigs > 1e-16]
        if len(positive_eigs) > 0:
            cov_eigs = 1.0 / positive_eigs
            print(f"   Range: [{cov_eigs.min():.3e}, {cov_eigs.max():.3e}]")
            print(f"   Mean: {cov_eigs.mean():.3e}, Median: {np.median(cov_eigs):.3e}")
            
            # Typical ellipse radius
            typical_radius_2sigma = 2 * np.sqrt(np.median(cov_eigs))
            print(f"   Typical 2σ ellipse radius (normalized space): {typical_radius_2sigma:.3f}")
            
            # Unnormalize to raw space
            std_vals = normalizer["state"].std.cpu().numpy()
            typical_pos_std = np.mean(std_vals)
            typical_radius_raw = typical_radius_2sigma * typical_pos_std
            print(f"   Typical 2σ ellipse radius (raw space): {typical_radius_raw:.3f}")
        
        analysis = {
            'H_eigenvalues': H_eigs,
            'rayleigh_quotients': rayleigh_quotients,
            'hvp_norms': hvp_norms,
            'H_approx': H_approx,
            'num_negative': num_negative
        }
    else:
        print(f"\n   Dimension {dim} too large, skipping full Hessian construction")
        print(f"   Using only Rayleigh quotient estimates")
        
        analysis = {
            'rayleigh_quotients': rayleigh_quotients,
            'hvp_norms': hvp_norms
        }
    
    return analysis
