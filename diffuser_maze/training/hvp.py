"""
Hessian-Vector Product computation and trajectory sampling.

Copy from notebook Cell 22:
1. compute_hvp() function
2. sample_trajectories_for_precision_training() function

These functions are core to the precision head training:
- compute_hvp(): Computes HVP using terminal score function
- sample_trajectories_for_precision_training(): Generates training samples

Dependencies:
- torch
- torch.autograd

Usage:
    from diffuser_maze.training.hvp import compute_hvp

    # Compute HVP for a trajectory
    hvp, terminal_score = compute_hvp(
        diffusion_model=diffuser,
        trajectory=trajectories,
        probe_vector=probe,
        use_noised_trajectory=False
    )
"""

import torch
from diffuser_maze.config import device
# from torch.autograd import grad


def compute_hvp(
    diffusion_model,
    trajectory,
    probe_vector,
    use_noised_trajectory=False,
    t_evals=None,
    curvature_floor=None,
    return_diagnostics=False
):
    """
    Compute Hessian-Vector Product (HVP) using multi-sigma averaging.

    Enhanced version with:
    - Multi-sigma evaluation for robust curvature estimation
    - Curvature floor regularization to prevent unbounded covariances
    - Optional diagnostics for monitoring training

    Based on FlowUncertainty_v8-4 implementation for improved robustness.

    Args:
        diffusion_model: The trained diffusion model
        trajectory: [B, T, 4] trajectories with NORMALIZED positions + raw actions
        probe_vector: [B, 2T] probe vector for positions only (in normalized space)
        use_noised_trajectory: Whether to add noise to trajectory at each sigma
                               False (default): Use clean trajectory (simpler)
                               True: Add noise for mathematical correctness
        t_evals: List of time values (NOT actual sigma!) for multi-scale probing
                 These are time parameters passed to the diffusion model.
                 Defaults to [0.005, 0.01, 0.02] from config
        curvature_floor: Regularization constant δ, adds δ·u to HVP
                         Defaults to 0.1 from config
        return_diagnostics: If True, return detailed diagnostics dict
                           Defaults to False for backward compatibility

    Returns:
        If return_diagnostics=False:
            hvp: [B, 2T] Averaged Hessian-vector product (in normalized space)
            terminal_score: [B, T, 4] last computed score for debugging

        If return_diagnostics=True:
            hvp: [B, 2T] Averaged Hessian-vector product
            terminal_score: [B, T, 4] last computed score
            diagnostics: Dict with:
                - 'hvp_per_sigma': List of [B, 2T] HVP at each time value
                - 'score_norm_per_sigma': List of scalar score norms
                - 'hvp_norm_per_sigma': List of scalar HVP norms
                - 'hvp_variance': Variance of HVP across time values
                - 'curvature_floor_contribution': ||δ·u|| / ||h_avg||
                - 'time_values_used': List of time values used
    """
    from diffuser_maze.config import T_EVALS, CURVATURE_FLOOR
    if use_noised_trajectory == False:
        assert t_evals is None, "If using clean trajectory, t_evals must be None to use defaults"

    # TODO: for clean traj, set a default sigma (smallest) for single eval
    batch_size = trajectory.shape[0]
    horizon_length = trajectory.shape[1]
    device = trajectory.device

    # Extract positions for HVP computation (first 2 dimensions, already normalized)
    # trajectory_positions: [B, T, 2] -> [B, 2T]
    trajectory_positions = trajectory[:, :, :2].reshape(batch_size, -1)
    trajectory_positions = trajectory_positions.requires_grad_(True)

    # Accumulator for multi-sigma HVP
    hvp_accumulated = torch.zeros_like(trajectory_positions)
    terminal_score = None  # Store last score for debugging

    # Diagnostics storage (if requested)
    hvp_list = [] if return_diagnostics else None
    score_norm_list = [] if return_diagnostics else None
    hvp_norm_list = [] if return_diagnostics else None

    # Loop over time values for robust curvature estimation
    for t_eval in t_evals:
        # Create time tensor at this time level
        t = torch.full((batch_size,), t_eval, device=device, dtype=torch.float32)

        # Reshape positions for score computation
        traj_pos_reshaped = trajectory_positions.reshape(batch_size, horizon_length, 2)

        # Concatenate tracked normalized positions with original raw actions
        trajectory_for_score = torch.cat([
            traj_pos_reshaped,       # Tracked normalized positions [B, T, 2]
            trajectory[:, :, 2:]      # Original raw actions [B, T, 2]
        ], dim=-1)  # Result: [B, T, 4]

        # Optionally add noise for mathematical correctness
        if use_noised_trajectory:
            alpha, sigma = diffusion_model.get_noise_schedule(t)
            # Reshape for broadcasting: [B] -> [B, 1, 1]
            alpha = alpha.view(-1, 1, 1)
            sigma = sigma.view(-1, 1, 1)
            noise = torch.randn_like(trajectory_for_score)
            trajectory_for_score = alpha * trajectory_for_score + sigma * noise

        # Compute score at this sigma level using score_function API
        score = diffusion_model.score_function(
            x=trajectory_for_score,
            t=t,
            condition=None,
            use_ema=True,
            requires_grad=True,
            add_terminal_noise=False  # We handle noise above if needed
        )  # [B, T, 4]

        terminal_score = score  # Keep last one for debugging

        # Extract score for positions only
        score_positions = score[:, :, :2].reshape(batch_size, -1)  # [B, 2T]

        # Compute scalar product: s_θ(ỹ, σ)^T u
        scalar_product = torch.sum(score_positions * probe_vector)

        # Compute HVP for this sigma: h_σ = ∇_ỹ (s_θ(ỹ, σ)^T u)
        hvp_sigma = torch.autograd.grad(
            outputs=scalar_product,
            inputs=trajectory_positions,
            create_graph=False,
            retain_graph=True  # Need to retain for next sigma iteration
        )[0]

        # Collect diagnostics if requested
        if return_diagnostics:
            hvp_list.append(hvp_sigma.detach().clone())
            score_norm_list.append(torch.norm(score_positions).item())
            hvp_norm_list.append(torch.norm(hvp_sigma).item())

        # Accumulate
        hvp_accumulated = hvp_accumulated + hvp_sigma

    # Average across time values
    hvp_avg = hvp_accumulated / float(len(t_evals))

    # Add curvature floor regularization: h = h_avg + δ·u
    # This prevents unbounded covariances and improves numerical stability
    curvature_floor_term = None
    if curvature_floor > 0.0:
        curvature_floor_term = float(curvature_floor) * probe_vector
        hvp_avg = hvp_avg + curvature_floor_term

    hvp_avg = hvp_avg.detach()

    # Prepare diagnostics if requested
    if return_diagnostics:
        # Compute variance across sigma values
        if len(hvp_list) > 1:
            hvp_stack = torch.stack(hvp_list, dim=0)  # [K, B, 2T]
            hvp_variance = torch.var(hvp_stack, dim=0).mean().item()
        else:
            hvp_variance = 0.0

        # Compute curvature floor contribution ratio
        hvp_avg_norm = torch.norm(hvp_avg).item()
        if curvature_floor_term is not None and hvp_avg_norm > 1e-10:
            floor_norm = torch.norm(curvature_floor_term).item()
            curvature_floor_contribution = floor_norm / hvp_avg_norm
        else:
            curvature_floor_contribution = 0.0

        diagnostics = {
            'hvp_per_sigma': hvp_list,
            'score_norm_per_sigma': score_norm_list,
            'hvp_norm_per_sigma': hvp_norm_list,
            'hvp_variance': hvp_variance,
            'curvature_floor_contribution': curvature_floor_contribution,
            'time_values_used': t_evals  # Changed from sigma_values_used
        }

        return hvp_avg, terminal_score, diagnostics
    else:
        return hvp_avg, terminal_score

def sample_trajectories_for_precision_training(diffuser_model, num_samples, sample_steps=0, traj_dataloader=None, normalizer=None):
    """
    Sample trajectories for precision head training using dataset samples (efficient approach).

    Instead of expensive diffusion sampling, we use trajectories from the dataset
    and compute terminal scores on them. This is much faster and mathematically valid
    since we're learning H = ∇²log p(y) which can be computed at any point y.

    IMPORTANT: Normalizes positions using the same normalizer as the diffuser training,
    following the specification: ỹ = D^(-1)(y - μ_T)

    Args:
        diffuser_model: The trained diffusion model (not used for sampling, only for validation)
        num_samples: Number of trajectory samples needed
        sample_steps: Kept for API compatibility but not used

    Returns:
        trajectories_norm: [num_samples, HORIZON, 4] trajectory with normalized positions + raw actions
        positions_flat_norm: [num_samples, 2*HORIZON] flattened normalized positions for precision head
    """
    # Get batch from dataset instead of expensive sampling
    batch_trajectories = []
    samples_collected = 0

    # Collect enough samples from dataset
    for batch in traj_dataloader:
        batch_data = batch['trajectory'].to(device)  # [B, HORIZON, 4]

        # Add to collection
        batch_trajectories.append(batch_data)
        samples_collected += batch_data.shape[0]

        # Stop when we have enough samples
        if samples_collected >= num_samples:
            break

    # Concatenate and take exactly num_samples
    all_trajectories_raw = torch.cat(batch_trajectories, dim=0)[:num_samples]

    # Extract positions and actions
    positions_raw = all_trajectories_raw[:, :, :2]  # [num_samples, HORIZON, 2]
    actions_raw = all_trajectories_raw[:, :, 2:]    # [num_samples, HORIZON, 2]

    # Normalize positions using state normalizer: ỹ = (y - μ) / σ
    positions_norm = normalizer["state"].normalize(positions_raw)  # [num_samples, HORIZON, 2]

    # Flatten normalized positions for precision head input
    positions_flat_norm = positions_norm.reshape(num_samples, -1)  # [num_samples, 2*HORIZON]

    # Reconstruct normalized trajectory (normalized positions + raw actions)
    trajectories_norm = torch.cat([positions_norm, actions_raw], dim=-1)  # [num_samples, HORIZON, 4]

    return trajectories_norm, positions_flat_norm