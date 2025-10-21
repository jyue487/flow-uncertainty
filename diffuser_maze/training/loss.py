"""
Loss functions and batch generation for precision head training.

Copy from notebook Cell 26:
1. precision_loss_function() - Main loss computation
2. make_rademacher_probes() - Generate Rademacher ±1 probe vectors
3. make_gaussian_probes() - Generate Gaussian N(0,1) probe vectors
4. generate_training_batch() - Complete batch generation

Loss components:
- Consistency loss: ||Λφ(y)u + h||² (main objective)
- L2 regularization on R
- Temporal smoothness on R
- Positive definiteness regularization

Dependencies:
- torch
- torch.nn.functional

Usage:
    from diffuser_maze.training.loss import precision_loss_function

    loss, components = precision_loss_function(
        predicted_hvp=pred,
        true_hvp=true,
        components=precision_components,
        config=precision_config
    )
"""

# TODO: Copy from Cell 26 - precision_loss_function()
import torch
import torch.nn.functional as F
from diffuser_maze.training.hvp import sample_trajectories_for_precision_training
from diffuser_maze.config import HORIZON, TRAJ_DIM, device

def precision_loss_function(predicted_hvp, true_hvp, components, config):
    """
    Compute the precision head loss with regularization.
    
    Loss = ||Λφ(y)u + h||_2^2 + regularization terms
    Note: the target is -∇^2 log p(y), hence the plus sign.
    
    Args:
        predicted_hvp: [B, 2T] predicted Hessian-vector product from precision head
        true_hvp: [B, 2T] true HVP computed from score function
        components: Dict with L and R matrix components from forward pass
        config: Training configuration dictionary
        
    Returns:
        total_loss: Scalar loss
        loss_components: Dict with individual loss terms
    """
    batch_size = predicted_hvp.shape[0]
    
    # Main self-consistency loss: ||Λφ(y)u + h||_2^2
    # Note: We want Λφ(y)u ≈ -h, so we minimize ||Λφ(y)u + h||^2
    consistency_loss = F.smooth_l1_loss(predicted_hvp, -true_hvp, reduction='mean', beta=1.0)
    
    # L2 regularization on low-rank R component (if present)
    l2_reg_r = torch.tensor(0.0, device=predicted_hvp.device)
    if 'r_matrix' in components and components['r_matrix'] is not None:
        r_matrix = components['r_matrix']  # [B, 2T, rank]
        l2_reg_r = config['l2_reg_r'] * torch.mean(r_matrix ** 2)
    
    # Temporal smoothness regularization on R component
    temporal_reg = torch.tensor(0.0, device=predicted_hvp.device)
    if config['temporal_smooth_reg'] > 0 and 'r_matrix' in components and components['r_matrix'] is not None:
        r_matrix = components['r_matrix']  # [B, 2T, rank]
        # Reshape to [B, T, 2, rank] to get per-timestep state components
        r_time = r_matrix.view(batch_size, HORIZON, 2, -1)
        
        # Temporal smoothness: ||R_{t+1} - R_t||^2
        r_diff = r_time[:, 1:] - r_time[:, :-1]  # [B, T-1, 2, rank]
        temporal_reg = config['temporal_smooth_reg'] * torch.mean(r_diff ** 2)
    
    # Positive definiteness regularization for L diagonal blocks
    # With new structure, diagonal elements are already enforced positive via softplus
    # But we can add small regularization to keep them away from eps
    l_diag = components['l_diagonal']  # [B, T, 2, 2]
    
    # Extract diagonal elements from each 2x2 block
    diag_00 = l_diag[:, :, 0, 0]  # [B, T] - x diagonal
    diag_11 = l_diag[:, :, 1, 1]  # [B, T] - y diagonal
    
    # Small penalty if diagonal gets too small (already has eps floor)
    min_diag_value = 0.01  # Minimum desired diagonal value
    pd_reg = 1e-4 * (
        torch.mean(torch.clamp(min_diag_value - diag_00, min=0) ** 2) +
        torch.mean(torch.clamp(min_diag_value - diag_11, min=0) ** 2)
    )
    
    # Total loss
    total_loss = consistency_loss + l2_reg_r + temporal_reg + pd_reg
    
    loss_components = {
        'consistency': consistency_loss.item(),
        'l2_reg_r': l2_reg_r.item(), 
        'temporal_smooth': temporal_reg.item(),
        'pd_reg': pd_reg.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_components


def make_rademacher_probes(batch_size, num_probes, dim, device):
    """Generate Rademacher probe vectors (±1 entries with equal probability)."""
    # ±1 with equal prob
    probes = torch.empty(batch_size, num_probes, dim, device=device).bernoulli_(0.5).mul_(2.).sub_(1.)
    # unit-norm per probe (or divide by sqrt(dim) for equal L2)
    probes = probes / (probes.norm(dim=-1, keepdim=True) + 1e-9)
    probes.requires_grad_(False)
    return probes


def make_gaussian_probes(batch_size, num_probes, dim, device, normalize=True):
    """
    Generate Gaussian N(0,1) probe vectors.

    Args:
        batch_size: Number of samples in batch
        num_probes: Number of probe vectors per sample
        dim: Dimension of each probe vector
        device: torch device
        normalize: If True, normalize to unit L2 norm (default: True)
                   If False, keep raw N(0,1) samples

    Returns:
        probes: [batch_size, num_probes, dim] tensor

    Note: Both normalized and unnormalized Gaussian probes are valid for
          Hutchinson trace estimation. Normalized version has more stable
          variance across different dimensions.

    References:
        - FlowUncertainty_v8-4.ipynb uses unnormalized Gaussian probes
        - Rademacher probes often have lower variance empirically
    """
    probes = torch.randn(batch_size, num_probes, dim, device=device)

    if normalize:
        # Normalize to unit L2 norm for stability
        probes = probes / (probes.norm(dim=-1, keepdim=True) + 1e-9)

    probes.requires_grad_(False)
    return probes


def generate_training_batch(diffuser_model, batch_size, config, traj_dataloader, normalizer):
    """
    Generate a training batch of trajectories and probe vectors.

    Args:
        diffuser_model: Trained diffusion model
        batch_size: Batch size
        config: Training configuration dictionary
        traj_dataloader: DataLoader for trajectories
        normalizer: Normalizer for data

    Config keys:
        - num_probes: Number of probe vectors per trajectory
        - sample_steps: Diffusion sampling steps (0 = use dataset samples)
        - probe_type: 'rademacher' (default) or 'gaussian'
        - normalize_gaussian: bool, whether to normalize Gaussian probes (default: True)

    Returns:
        trajectories: [batch_size, T, 4] trajectory samples (normalized positions + raw actions)
        positions_flat: [batch_size, 2T] flattened normalized positions
        probe_vectors: [batch_size, num_probes, 2T] probe vectors
    """
    # Sample trajectories
    trajectories, positions_flat = sample_trajectories_for_precision_training(
        diffuser_model,
        num_samples=batch_size,
        sample_steps=config['sample_steps'],
        traj_dataloader=traj_dataloader,
        normalizer=normalizer
    )

    # Generate probe vectors (choose type from config)
    num_probes = config['num_probes']
    probe_type = config.get('probe_type', 'rademacher')  # default: Rademacher

    if probe_type == 'gaussian':
        probe_vectors = make_gaussian_probes(
            batch_size=batch_size,
            num_probes=num_probes,
            dim=TRAJ_DIM,
            device=device,
            normalize=config.get('normalize_gaussian', True)
        )
    elif probe_type == 'rademacher':
        probe_vectors = make_rademacher_probes(
            batch_size=batch_size,
            num_probes=num_probes,
            dim=TRAJ_DIM,
            device=device
        )
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}. Use 'rademacher' or 'gaussian'")

    return trajectories, positions_flat, probe_vectors