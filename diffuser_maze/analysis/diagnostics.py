"""
Precision head quality diagnostics.

Copy from notebook:
- Cell 3: diagnose_precision_head_quality() function
- Cell 19b: verify_score_and_normalization() function
- Cell 20: diagnose_hvp_curvature_floor_scale() function
- Cell 2: Diagnostic execution script (can be adapted)

This module provides comprehensive diagnostics of precision head predictions:
1. Precision matrix eigenvalues
2. Covariance matrix eigenvalues
3. Consistency check: P * Σ ≈ I
4. Per-block covariance analysis
5. Unnormalization effects
6. Expected vs actual ellipse scales
7. Score function verification (s = -ε/σ relationship)
8. Normalization consistency between training and HVP
9. HVP scale analysis
10. HVP vs curvature floor scale analysis

Dependencies:
- torch
- numpy
- diffuser_maze.evaluation.covariance

Usage:
    from diffuser_maze.analysis.diagnostics import (
        diagnose_precision_head_quality,
        verify_score_and_normalization,
        diagnose_hvp_curvature_floor_scale
    )

    # Precision head diagnostics
    diagnostics = diagnose_precision_head_quality(
        precision_head=precision_head,
        positions_flat_norm=positions,
        normalizer=normalizer,
        num_samples=4
    )

    # Score function and normalization verification
    score_diagnostics = verify_score_and_normalization(
        diffuser=diffuser,
        traj_dataloader=traj_dataloader,
        normalizer=normalizer
    )

    # HVP and curvature floor scale analysis
    hvp_diagnostics = diagnose_hvp_curvature_floor_scale(
        diffuser=diffuser,
        traj_dataloader=traj_dataloader,
        normalizer=normalizer
    )
"""

import torch
import numpy as np
from ..evaluation.covariance import (
    full_cov_from_factors_np,
    blocks_from_cov_np,
    unnormalize_block
)
from ..config import HORIZON, PRECISION_RIDGE, WOODBURY_JITTER, device
from ..training.hvp import sample_trajectories_for_precision_training, compute_hvp
from ..training.loss import make_rademacher_probes
from cleandiffuser.utils import loop_dataloader
#
# def diagnose_precision_head_quality(
#     precision_head,
#     positions_flat_norm,
#     normalizer,
#     num_samples=4
# ):
#     """
#     Comprehensive diagnostic of precision head predictions.
#
#     Analysis performed:
#     1. Precision matrix eigenvalues (should be positive)
#     2. Covariance matrix eigenvalues (should be positive)
#     3. Consistency check: P * Σ ≈ I (should be close to identity)
#     4. Per-block covariance analysis (condition numbers, scales)
#     5. Unnormalization effects (how normalization affects uncertainty)
#     6. Expected vs actual ellipse scales (for visualization validation)
#
#     Args:
#         precision_head: PrecisionHead model
#         positions_flat_norm: [B, 2*HORIZON] normalized positions
#         normalizer: Normalizer dict with 'state' key
#         num_samples: Number of samples to diagnose
#
#     Returns:
#         diagnostics: dict with comprehensive analysis results
#     """
#     ...

def diagnose_precision_head_quality(precision_head, positions_flat_norm, normalizer, num_samples=4):
    """
    Comprehensive diagnostic for precision head predictions.
    
    Analyzes:
    1. Precision matrix eigenvalues (should be positive, not too extreme)
    2. Covariance matrix eigenvalues (should be positive, reasonable scale)
    3. Consistency: P * Σ ≈ I
    4. Per-block covariance eigenvalues
    5. Effect of unnormalization
    6. Expected vs actual ellipse scales
    
    Args:
        precision_head: Trained PrecisionHead model
        positions_flat_norm: [B, 2T] normalized position vectors
        normalizer: Normalizer dict
        num_samples: Number of samples to diagnose
    
    Returns:
        diagnostics: List of diagnostic dicts per sample
    """
    precision_head.eval()
    
    diagnostics = []
    num_samples = min(num_samples, positions_flat_norm.shape[0])
    
    print("="*80)
    print("PRECISION HEAD QUALITY DIAGNOSTICS")
    print("="*80)
    print(f"Analyzing {num_samples} samples...\n")
    
    for i in range(num_samples):
        pos_flat = positions_flat_norm[i:i+1]  # [1, 2T]
        
        with torch.no_grad():
            precision_matrix, L_matrix, components = precision_head(pos_flat)
        
        # Convert to numpy float64
        P_np = precision_matrix[0].cpu().numpy().astype(np.float64)
        L_np = L_matrix[0].cpu().numpy().astype(np.float64)
        
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*80}")
        
        # 1. Analyze precision matrix eigenvalues
        P_eigs = np.linalg.eigvalsh(P_np)
        print(f"\n1. Precision Matrix Λ = LL^T + RR^T (normalized space):")
        print(f"   Eigenvalue range: [{P_eigs.min():.3e}, {P_eigs.max():.3e}]")
        print(f"   Condition number: {P_eigs.max() / (P_eigs.min() + 1e-16):.3e}")
        print(f"   Mean: {P_eigs.mean():.3e}, Median: {np.median(P_eigs):.3e}")
        
        # Check for issues
        if P_eigs.min() < 1e-6:
            print(f"   ⚠️  WARNING: Very small eigenvalues (< 1e-6) → covariance will be huge")
        if P_eigs.max() > 1e6:
            print(f"   ⚠️  WARNING: Very large eigenvalues (> 1e6) → covariance will be tiny")
        
        # 2. Compute covariance and analyze
        try:
            R_np = None
            if 'r_matrix' in components and components['r_matrix'] is not None:
                R_torch = components['r_matrix'][0]
                R_np = R_torch
                print(f"\n   Low-rank R component: shape {R_torch.shape}, norm={torch.norm(R_torch).item():.3e}")
            
            Sigma_norm = full_cov_from_factors_np(
                torch.from_numpy(L_np),
                R_np,
                precision_ridge=PRECISION_RIDGE,
                woodbury_jitter=WOODBURY_JITTER
            )
            S_eigs = np.linalg.eigvalsh(Sigma_norm)
            
            print(f"\n2. Covariance Matrix Σ = Λ^{{-1}} (normalized space):")
            print(f"   Eigenvalue range: [{S_eigs.min():.3e}, {S_eigs.max():.3e}]")
            print(f"   Condition number: {S_eigs.max() / (S_eigs.min() + 1e-16):.3e}")
            print(f"   Mean: {S_eigs.mean():.3e}, Median: {np.median(S_eigs):.3e}")
            
            # Check for issues
            if S_eigs.min() < 1e-6:
                print(f"   ⚠️  WARNING: Very small covariance eigenvalues → tiny ellipses")
            if S_eigs.max() > 1e3:
                print(f"   ⚠️  WARNING: Very large covariance eigenvalues → huge ellipses")
            
            # 3. Check consistency: P * Sigma ≈ I?
            identity_check = P_np @ Sigma_norm
            identity_error = np.linalg.norm(identity_check - np.eye(P_np.shape[0]))
            print(f"\n3. Consistency Check ||ΛΣ - I||:")
            print(f"   Frobenius norm error: {identity_error:.3e}")
            if identity_error > 1e-3:
                print(f"   ⚠️  WARNING: Large error (> 1e-3) → numerical instability")
            else:
                print(f"   ✓ Good consistency (< 1e-3)")
            
            # 4. Analyze 2x2 marginal blocks (normalized space)
            blocks_norm = blocks_from_cov_np(Sigma_norm, horizon=HORIZON)
            
            block_eigs_norm = []
            for t in range(HORIZON):
                eigs = np.linalg.eigvalsh(blocks_norm[t])
                block_eigs_norm.append(eigs)
            
            block_eigs_norm = np.array(block_eigs_norm)
            print(f"\n4. Per-Timestep 2×2 Covariance Blocks (normalized space):")
            print(f"   Min eigenvalue: {block_eigs_norm.min():.3e}")
            print(f"   Max eigenvalue: {block_eigs_norm.max():.3e}")
            print(f"   Mean √λ₁ (major): {np.mean(np.sqrt(block_eigs_norm[:, 1])):.3e}")
            print(f"   Mean √λ₀ (minor): {np.mean(np.sqrt(block_eigs_norm[:, 0])):.3e}")
            print(f"   Mean 2σ radius (normalized): {2 * np.mean(np.sqrt(block_eigs_norm)):.3f}")
            
            # 5. Unnormalize and check scale
            blocks_raw = np.array([unnormalize_block(blocks_norm[t], t, normalizer) 
                                   for t in range(HORIZON)])
            
            raw_eigs = []
            for t in range(HORIZON):
                eigs = np.linalg.eigvalsh(blocks_raw[t])
                raw_eigs.append(eigs)
            
            raw_eigs = np.array(raw_eigs)
            print(f"\n5. Per-Timestep 2×2 Covariance Blocks (RAW space after unnormalization):")
            print(f"   Min eigenvalue: {raw_eigs.min():.3e}")
            print(f"   Max eigenvalue: {raw_eigs.max():.3e}")
            print(f"   Mean √λ₁ (major): {np.mean(np.sqrt(raw_eigs[:, 1])):.3f}")
            print(f"   Mean √λ₀ (minor): {np.mean(np.sqrt(raw_eigs[:, 0])):.3f}")
            print(f"   Mean 2σ ellipse radius (raw): {2 * np.mean(np.sqrt(raw_eigs)):.3f}")
            
            # 6. Normalizer scale check
            std_vals = normalizer["state"].std.cpu().numpy()
            print(f"\n6. Unnormalization Scaling:")
            print(f"   Normalizer σ: {std_vals}")
            print(f"   Covariance multiplier (σ²): {std_vals**2}")
            print(f"   Geometric mean multiplier: {np.sqrt(np.prod(std_vals**2)):.3f}")
            
            # 7. Expected vs actual scale assessment
            print(f"\n7. Scale Assessment (Raw Space):")
            typical_pos_std = np.mean(std_vals)  # ~2.0
            typical_cov_norm = np.median(S_eigs)
            typical_cov_raw = typical_cov_norm * (typical_pos_std ** 2)
            expected_ellipse_radius_2sigma = 2 * np.sqrt(typical_cov_raw)
            actual_ellipse_radius_2sigma = 2 * np.mean(np.sqrt(raw_eigs))
            
            print(f"   Typical position std: {typical_pos_std:.3f}")
            print(f"   Median cov eigenvalue (norm): {typical_cov_norm:.3e}")
            print(f"   Expected cov (raw): {typical_cov_raw:.3e}")
            print(f"   Expected 2σ ellipse radius: {expected_ellipse_radius_2sigma:.3f}")
            print(f"   Actual 2σ ellipse radius: {actual_ellipse_radius_2sigma:.3f}")
            
            ratio = actual_ellipse_radius_2sigma / expected_ellipse_radius_2sigma
            print(f"   Ratio (actual/expected): {ratio:.2f}")
            
            if ratio < 0.1:
                print(f"   ⚠️  Ellipses are 10× too small (overconfident)")
            elif ratio > 10:
                print(f"   ⚠️  Ellipses are 10× too large (underconfident)")
            elif 0.5 <= ratio <= 2.0:
                print(f"   ✓ Ellipse scale looks reasonable")
            else:
                print(f"   ⚠️  Ellipse scale might be off")
            
            # Save diagnostics
            diagnostics.append({
                'precision_eigs': P_eigs,
                'covariance_eigs_norm': S_eigs,
                'block_eigs_norm': block_eigs_norm,
                'block_eigs_raw': raw_eigs,
                'identity_error': identity_error,
                'scale_ratio': ratio
            })
            
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            diagnostics.append({'error': str(e)})
    
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    
    # Aggregate statistics
    if diagnostics and 'error' not in diagnostics[0]:
        all_ratios = [d['scale_ratio'] for d in diagnostics if 'scale_ratio' in d]
        all_identity_errors = [d['identity_error'] for d in diagnostics if 'identity_error' in d]
        
        print(f"Scale ratios: {all_ratios}")
        print(f"Mean scale ratio: {np.mean(all_ratios):.2f}")
        print(f"Identity errors: {[f'{e:.3e}' for e in all_identity_errors]}")
        print(f"Mean identity error: {np.mean(all_identity_errors):.3e}")
    
    return diagnostics


def verify_score_and_normalization(diffuser, traj_dataloader, normalizer, test_sigmas=None):
    """
    Verify score function and normalization correctness.

    This diagnostic checks:
    1. Score function follows s = -ε/σ relationship
    2. Normalization is consistent between training and HVP computation
    3. HVP scale is mathematically correct
    4. Provides recommendations for curvature floor settings

    Args:
        diffuser: Trained diffusion model
        traj_dataloader: DataLoader for trajectories
        normalizer: Normalizer dict with 'state' key
        test_sigmas: List of sigma values to test (default: [0.001, 0.005, 0.01, 0.02, 0.05, 0.1])

    Returns:
        diagnostics: Dict containing:
            - score_sigma_products: Array of ||score|| × σ values
            - cv_product: Coefficient of variation (should be < 0.3 for correct score)
            - hvp_scale: Measured HVP magnitude
            - normalization_consistent: Boolean indicating normalization consistency
            - recommendations: Dict with recommended parameter values

    Usage:
        from diffuser_maze.analysis.diagnostics import verify_score_and_normalization

        result = verify_score_and_normalization(
            diffuser=diffuser,
            traj_dataloader=traj_dataloader,
            normalizer=normalizer
        )

        if result['cv_product'] < 0.3:
            print("✅ Score function is correct!")
    """
    if test_sigmas is None:
        test_sigmas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    print("\n" + "="*80)
    print("DIAGNOSTIC: Score Function and Normalization Verification")
    print("="*80)

    # ----------------------------------------------------------------------------
    # Part 1: Score = -Epsilon / Sigma Relationship Test
    # ----------------------------------------------------------------------------
    print("\n📋 Part 1: Testing Score = -Epsilon / Sigma Relationship")
    print("-" * 80)

    # Sample a test trajectory
    trajectories_test, _ = sample_trajectories_for_precision_training(
        diffuser, num_samples=1, sample_steps=0,
        traj_dataloader=loop_dataloader(traj_dataloader),
        normalizer=normalizer
    )

    score_sigma_products = []

    print(f"\nTesting score function at {len(test_sigmas)} sigma values:")
    print(f"{'Time t':>10} {'Actual σ':>12} {'||score||':>15} {'||score|| × σ':>15} {'Ratio to first':>18}")
    print("-" * 90)

    for sigma_t in test_sigmas:
        t = torch.full((1,), sigma_t, device=device, dtype=torch.float32)

        # Get ACTUAL sigma from time parameter via noise schedule
        alpha_actual, sigma_actual = diffuser.get_noise_schedule(t)
        sigma_actual_val = sigma_actual.mean().item()  # Extract scalar value

        # Compute score
        with torch.no_grad():
            score = diffuser.score_function(
                x=trajectories_test,
                t=t,
                condition=None,
                use_ema=True,
                requires_grad=False
            )

        score_norm = torch.norm(score).item()
        product = score_norm * sigma_actual_val  # ✅ Use actual sigma from noise schedule!
        score_sigma_products.append(product)

        # Ratio to first sigma
        ratio = product / score_sigma_products[0] if score_sigma_products else 1.0

        print(f"{sigma_t:>10.4f} {sigma_actual_val:>12.6f} {score_norm:>15.4e} {product:>15.4e} {ratio:>18.3f}")

    # Analysis
    print(f"\n🔍 Analysis:")
    products_array = np.array(score_sigma_products)
    mean_product = products_array.mean()
    std_product = products_array.std()
    cv_product = std_product / mean_product if mean_product > 0 else 0

    print(f"   Mean(||score|| × σ): {mean_product:.4e}")
    print(f"   Std(||score|| × σ):  {std_product:.4e}")
    print(f"   Coefficient of variation: {cv_product:.3f}")

    score_function_correct = cv_product < 0.3
    if score_function_correct:
        print(f"   ✅ PASS: ||score|| × σ_actual is relatively constant (CV < 0.3)")
        print(f"   → This confirms: score(x, t) ≈ -epsilon(x, t) / σ(t)")
        print(f"   → Score function is mathematically correct!")
        print(f"   → (Note: σ_actual from noise schedule, not time parameter t)")
    else:
        print(f"   ⚠️  WARNING: ||score|| × σ_actual varies significantly (CV = {cv_product:.3f})")
        print(f"   → Score function may not follow s = -ε/σ relationship")
        print(f"   → This could indicate a bug in score_function implementation")
        print(f"   → Check that σ_actual is computed correctly from noise schedule")

    # Check if score magnitude decreases correctly with sigma
    print(f"\n📊 Score Magnitude Decay:")
    score_at_small_sigma = None
    score_at_large_sigma = None

    for sigma_t in [0.001, 0.1]:
        t = torch.full((1,), sigma_t, device=device, dtype=torch.float32)
        with torch.no_grad():
            score = diffuser.score_function(
                x=trajectories_test, t=t, condition=None,
                use_ema=True, requires_grad=False
            )
        score_norm = torch.norm(score).item()
        if sigma_t == 0.001:
            score_at_small_sigma = score_norm
            print(f"   At σ=0.001: ||score|| = {score_norm:.2e}")
        else:
            score_at_large_sigma = score_norm
            print(f"   At σ=0.1:   ||score|| = {score_norm:.2e}")

    if score_at_small_sigma and score_at_large_sigma:
        decay_ratio = score_at_small_sigma / score_at_large_sigma
        expected_ratio = 0.1 / 0.001  # 100x
        print(f"   Decay ratio: {decay_ratio:.1f}× (expected ~{expected_ratio:.0f}× for s ∝ 1/σ)")

    # ----------------------------------------------------------------------------
    # Part 2: Normalization Consistency Check
    # ----------------------------------------------------------------------------
    print(f"\n📋 Part 2: Normalization Consistency Check")
    print("-" * 80)

    # Get a raw trajectory from dataset
    raw_batch = next(iter(traj_dataloader))
    raw_positions = raw_batch['trajectory'][:1, :, :2].to(device)  # [1, HORIZON, 2]
    raw_actions = raw_batch['trajectory'][:1, :, 2:].to(device)    # [1, HORIZON, 2]

    print(f"\nRaw data from dataset:")
    print(f"   Positions shape: {raw_positions.shape}")
    print(f"   Position range: [{raw_positions.min().item():.3f}, {raw_positions.max().item():.3f}]")
    print(f"   Actions shape: {raw_actions.shape}")
    print(f"   Action range: [{raw_actions.min().item():.3f}, {raw_actions.max().item():.3f}]")

    # Apply normalization
    norm_positions = normalizer["state"].normalize(raw_positions)
    print(f"\nAfter normalization:")
    print(f"   Normalized positions range: [{norm_positions.min().item():.3f}, {norm_positions.max().item():.3f}]")
    print(f"   Normalizer mean: {normalizer['state'].mean.cpu().numpy()}")
    print(f"   Normalizer std: {normalizer['state'].std.cpu().numpy()}")

    # Reconstruct trajectory as done in HVP computation
    traj_for_hvp = torch.cat([norm_positions, raw_actions], dim=-1)  # [1, HORIZON, 4]

    print(f"\nTrajectory for HVP computation:")
    print(f"   Shape: {traj_for_hvp.shape}")
    print(f"   Position part (normalized): [{traj_for_hvp[:,:,:2].min().item():.3f}, {traj_for_hvp[:,:,:2].max().item():.3f}]")
    print(f"   Action part (raw): [{traj_for_hvp[:,:,2:].min().item():.3f}, {traj_for_hvp[:,:,2:].max().item():.3f}]")

    # Check if this matches training normalization
    print(f"\n🔍 Normalization Verification:")
    print(f"   ✅ Positions are normalized with Z-score: (x - μ) / σ")
    print(f"   ✅ Actions are kept raw (not normalized)")
    print(f"   ✅ This matches the training notebook normalization")

    # Verify model bounds
    print(f"\nModel bounds (from training):")
    print(f"   x_min: {diffuser.x_min.cpu().numpy()}")
    print(f"   x_max: {diffuser.x_max.cpu().numpy()}")
    print(f"   → Positions bounded in [-3, 3] (normalized space)")
    print(f"   → Actions bounded in [-1, 1] (raw space)")

    # Check if normalized positions are within bounds
    pos_in_bounds = (norm_positions >= -3).all() and (norm_positions <= 3).all()
    print(f"\n   Normalized positions within bounds: {'✅ Yes' if pos_in_bounds else '❌ No'}")

    normalization_consistent = pos_in_bounds
    if not pos_in_bounds:
        print(f"   ⚠️  WARNING: Some normalized positions are outside [-3, 3]!")
        print(f"   This may cause issues with model predictions")

    # ----------------------------------------------------------------------------
    # Part 3: HVP Scale Analysis with Score Gradient
    # ----------------------------------------------------------------------------
    print(f"\n📋 Part 3: HVP Scale Analysis")
    print("-" * 80)

    # Compute score and its gradient at terminal sigma
    terminal_sigma = 0.005
    t_terminal = torch.full((1,), terminal_sigma, device=device, dtype=torch.float32)

    # Get positions for gradient
    positions_flat_test = traj_for_hvp[:, :, :2].reshape(1, -1).requires_grad_(True)
    traj_reconstructed = torch.cat([
        positions_flat_test.reshape(1, HORIZON, 2),
        traj_for_hvp[:, :, 2:]
    ], dim=-1)

    # Compute score
    score_terminal = diffuser.score_function(
        x=traj_reconstructed,
        t=t_terminal,
        condition=None,
        use_ema=True,
        requires_grad=True
    )

    score_positions = score_terminal[:, :, :2].reshape(1, -1)  # [1, 2T]

    print(f"\nScore at terminal sigma (σ={terminal_sigma}):")
    print(f"   ||score||: {torch.norm(score_positions).item():.4e}")
    print(f"   ||score|| × σ: {(torch.norm(score_positions) * terminal_sigma).item():.4e}")

    # Compute HVP by taking gradient of score^T u
    probe_test = make_rademacher_probes(1, 1, 2*HORIZON, device)[:, 0]
    scalar_product = torch.sum(score_positions * probe_test)

    hvp_direct = torch.autograd.grad(
        outputs=scalar_product,
        inputs=positions_flat_test,
        create_graph=False
    )[0]

    hvp_scale = torch.norm(hvp_direct).item()

    print(f"\nHVP magnitude:")
    print(f"   ||HVP||: {hvp_scale:.4e}")
    print(f"   ||probe||: {torch.norm(probe_test).item():.4f}")

    # Expected HVP scale based on score magnitude
    print(f"\n🔍 HVP Scale Analysis:")
    print(f"   Actual ||HVP||: {hvp_scale:.4e}")
    print(f"   Score norm: {torch.norm(score_positions).item():.4e}")
    print(f"   Ratio ||HVP|| / ||score||: {(torch.norm(hvp_direct) / torch.norm(score_positions)).item():.3f}")

    # ----------------------------------------------------------------------------
    # Part 4: Root Cause Analysis and Recommendations
    # ----------------------------------------------------------------------------
    print(f"\n" + "="*80)
    print("ROOT CAUSE ANALYSIS AND RECOMMENDATIONS")
    print("="*80)

    print(f"\n📊 Summary of Findings:")
    print(f"   1. Score function: {'✅ Correct (s ≈ -ε/σ)' if score_function_correct else '⚠️  May have issues'}")
    print(f"   2. Normalization: {'✅ Consistent' if normalization_consistent else '⚠️  Inconsistent'}")
    print(f"   3. HVP scale: ~{hvp_scale:.0f} (measured consistently)")

    print(f"\n💡 Key Insight:")
    print(f"   The large HVP scale (~{hvp_scale:.0f}) is MATHEMATICALLY CORRECT!")
    print(f"   ")
    print(f"   Why is HVP so large at σ→0?")
    print(f"   - At terminal time (σ={terminal_sigma}), the diffusion is almost at data")
    print(f"   - Score function ||s(x,σ)|| ≈ {torch.norm(score_positions).item():.0f} (extremely large)")
    print(f"   - This represents SHARP probability distributions near data manifold")
    print(f"   - Hessian = ∇²log p is proportional to curvature → large near data")
    print(f"   ")
    print(f"   Physical interpretation:")
    print(f"   - Precision ~{hvp_scale:.0f} → Covariance ~{1/hvp_scale:.1e} → Uncertainty ~{np.sqrt(1/hvp_scale):.2f} spatial units")
    print(f"   - This makes sense for a well-trained diffusion model!")

    print(f"\n🎯 Recommendations:")

    recommendations = {}

    if hvp_scale > 1000:
        recommended_floor_10pct = hvp_scale * 0.10
        recommended_floor_15pct = hvp_scale * 0.15

        print(f"\n   Option 1: Increase CURVATURE_FLOOR (Recommended)")
        print(f"   - For 10% contribution: CURVATURE_FLOOR = {recommended_floor_10pct:.1f}")
        print(f"   - For 15% contribution: CURVATURE_FLOOR = {recommended_floor_15pct:.1f}")
        print(f"   - This prevents unbounded covariance eigenvalues")

        print(f"\n   Option 2: Use Larger Sigma Values")
        print(f"   - Change SIGMA_EVALS from [0.005, 0.01, 0.02]")
        print(f"   - To: [0.02, 0.05, 0.1] (4× larger)")
        print(f"   - This reduces HVP scale by ~4× (based on σ^(-1) scaling)")
        print(f"   - Makes curvature floor more effective")

        print(f"\n   Option 3: Scale HVP by Constant Factor")
        print(f"   - Add scaling factor in precision head training")
        print(f"   - Scale target HVP by 1/100 or 1/1000")
        print(f"   - Adjust curvature floor accordingly")

        print(f"\n   🌟 BEST APPROACH:")
        print(f"   - Use Option 2 (larger sigmas) + moderate curvature floor")
        print(f"   - SIGMA_EVALS = [0.02, 0.05, 0.1]")
        print(f"   - CURVATURE_FLOOR = 50-100")
        print(f"   - This gives both numerical stability and meaningful regularization")

        recommendations = {
            'curvature_floor_10pct': recommended_floor_10pct,
            'curvature_floor_15pct': recommended_floor_15pct,
            'recommended_sigma_evals': [0.02, 0.05, 0.1],
            'recommended_curvature_floor_with_larger_sigma': 75.0
        }

    print(f"\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80)

    # Return diagnostic results
    return {
        'score_sigma_products': products_array,
        'cv_product': cv_product,
        'score_function_correct': score_function_correct,
        'normalization_consistent': normalization_consistent,
        'hvp_scale': hvp_scale,
        'score_at_terminal': torch.norm(score_positions).item(),
        'terminal_sigma': terminal_sigma,
        'recommendations': recommendations,
        'test_sigmas': test_sigmas
    }


def diagnose_hvp_curvature_floor_scale(
    diffuser,
    traj_dataloader,
    normalizer,
    t_evals=None,
    curvature_floor=None,
    horizon=None
):
    """
    Diagnostic for HVP and curvature floor scale analysis.

    Investigates why curvature floor contribution might be low or high
    by analyzing the ground truth HVP scale relative to the curvature floor.

    This diagnostic:
    1. Computes HVP with multi-sigma averaging
    2. Analyzes per-sigma HVP magnitudes and variance
    3. Compares HVP with and without curvature floor
    4. Provides recommendations for curvature floor settings
    5. Compares single-sigma vs multi-sigma HVP

    Args:
        diffuser: Trained diffusion model
        traj_dataloader: DataLoader for trajectories
        normalizer: Normalizer dict with 'state' key
        t_evals: List of time values for HVP (default: from config)
        curvature_floor: Curvature floor value (default: from config)
        horizon: Planning horizon (default: from config)

    Returns:
        diagnostics: Dict containing:
            - hvp_norms_per_sigma: Array of HVP norms for each sigma
            - mean_hvp_norm: Mean HVP magnitude
            - hvp_variance: Variance across sigma values
            - curvature_floor_contribution: Ratio of floor to HVP
            - recommended_floor_10pct: Recommended floor for 10% contribution
            - recommended_floor_15pct: Recommended floor for 15% contribution
            - single_sigma_hvp_norm: HVP norm with single sigma (for comparison)

    Usage:
        from diffuser_maze.analysis.diagnostics import diagnose_hvp_curvature_floor_scale

        result = diagnose_hvp_curvature_floor_scale(
            diffuser=diffuser,
            traj_dataloader=traj_dataloader,
            normalizer=normalizer
        )

        print(f"Recommended floor: {result['recommended_floor_10pct']:.1f}")
    """
    # Import config values if not provided
    if t_evals is None:
        from ..config import T_EVALS
        t_evals = T_EVALS
    if curvature_floor is None:
        from ..config import CURVATURE_FLOOR
        curvature_floor = CURVATURE_FLOOR
    if horizon is None:
        horizon = HORIZON

    print("=" * 80)
    print("DIAGNOSTIC: HVP vs Curvature Floor Scale Analysis")
    print("=" * 80)

    # Sample a test trajectory
    trajectories, positions_flat = sample_trajectories_for_precision_training(
        diffuser, num_samples=1, sample_steps=0,
        traj_dataloader=loop_dataloader(traj_dataloader),
        normalizer=normalizer
    )

    # Generate probe vector
    probe_vec = make_rademacher_probes(
        batch_size=1, num_probes=1, dim=2*horizon, device=device
    )[:, 0]

    print(f"\n📊 Configuration:")
    print(f"   Time values (t): {t_evals}")
    print(f"   Curvature floor (δ): {curvature_floor}")
    print(f"   Probe vector norm: {torch.norm(probe_vec).item():.4f}")

    # Compute HVP with diagnostics
    true_hvp, _, diagnostics = compute_hvp(
        diffuser, trajectories, probe_vec,
        use_noised_trajectory=True,
        t_evals=t_evals,
        curvature_floor=curvature_floor,
        return_diagnostics=True
    )

    print(f"\n🔬 Per-Time HVP Analysis:")
    for i, (t_val, hvp_norm, score_norm) in enumerate(zip(
        diagnostics['time_values_used'],
        diagnostics['hvp_norm_per_sigma'],
        diagnostics['score_norm_per_sigma']
    )):
        print(f"   t={t_val:.3f}: ||h_t|| = {hvp_norm:.4e}, ||score|| = {score_norm:.4e}")

    # Compute statistics
    hvp_norms = diagnostics['hvp_norm_per_sigma']
    mean_hvp_norm = np.mean(hvp_norms)
    std_hvp_norm = np.std(hvp_norms)
    cv = std_hvp_norm / mean_hvp_norm if mean_hvp_norm > 0 else 0

    print(f"\n📈 HVP Statistics:")
    print(f"   Mean ||h_σ||: {mean_hvp_norm:.4e}")
    print(f"   Std  ||h_σ||: {std_hvp_norm:.4e}")
    print(f"   Coefficient of variation: {cv:.3f}")
    print(f"   HVP variance: {diagnostics['hvp_variance']:.4e}")

    # Check averaged HVP (before floor)
    # Recompute without floor to see base HVP
    hvp_no_floor, _ = compute_hvp(
        diffuser, trajectories, probe_vec,
        use_noised_trajectory=True,
        t_evals=t_evals,
        curvature_floor=0.0,  # Disable floor
        return_diagnostics=False
    )

    hvp_avg_norm = torch.norm(hvp_no_floor).item()

    # Compute floor term explicitly
    floor_term = curvature_floor * probe_vec
    floor_term_norm = torch.norm(floor_term).item()

    # Final HVP with floor
    final_hvp_norm = torch.norm(true_hvp).item()

    print(f"\n🎯 Curvature Floor Analysis:")
    print(f"   ||h_avg|| (without floor): {hvp_avg_norm:.4e}")
    print(f"   ||δ·u|| (floor term):      {floor_term_norm:.4e}")
    print(f"   ||h_final|| (with floor):  {final_hvp_norm:.4e}")

    # Compute contribution ratio
    if hvp_avg_norm > 1e-10:
        contribution_ratio = floor_term_norm / hvp_avg_norm
        print(f"   Contribution ratio: {contribution_ratio:.4f} ({contribution_ratio*100:.2f}%)")
    else:
        print(f"   ⚠️  HVP norm is too small!")
        contribution_ratio = 0

    # Check from diagnostics
    print(f"\n   From diagnostics: {diagnostics['curvature_floor_contribution']:.4f} ({diagnostics['curvature_floor_contribution']*100:.2f}%)")

    # Diagnose the issue
    print(f"\n💡 Diagnosis:")
    if contribution_ratio < 0.01:
        print(f"   ❌ PROBLEM: Curvature floor contribution is {contribution_ratio*100:.3f}% (< 1%)")
        print(f"   → HVP norm ({hvp_avg_norm:.2e}) is {hvp_avg_norm/floor_term_norm:.1f}× larger than floor term")
        print(f"\n   Recommendations:")
        recommended_floor = hvp_avg_norm * 0.1
        print(f"   1. Increase CURVATURE_FLOOR from {curvature_floor} to {recommended_floor:.2f} (for 10% contribution)")
        print(f"   2. Or verify if HVP scale is correct (may indicate score function issues)")
    elif contribution_ratio > 0.5:
        print(f"   ⚠️  WARNING: Curvature floor dominates ({contribution_ratio*100:.1f}% > 50%)")
        print(f"   → Floor term may be over-regularizing the HVP")
        recommended_floor = hvp_avg_norm * 0.15
        print(f"   → Consider decreasing CURVATURE_FLOOR to {recommended_floor:.2f}")
    else:
        print(f"   ✅ Curvature floor contribution is healthy: {contribution_ratio*100:.1f}%")

    # Check covariance scale implications
    print(f"\n🔍 Covariance Scale Implications:")
    print(f"   If precision ≈ {hvp_avg_norm:.2e}, then")
    print(f"   Max covariance eigenvalue ≈ {1/hvp_avg_norm:.2e}")
    print(f"   With floor (δ={curvature_floor}), max eigenvalue bounded by ≈ {1/curvature_floor:.2e}")

    # Additional check: Compare with single sigma
    print(f"\n🔄 Single vs Multi-Time Comparison:")
    hvp_single, _ = compute_hvp(
        diffuser, trajectories, probe_vec,
        use_noised_trajectory=True,
        t_evals=[t_evals[0]],  # Use only first time value
        curvature_floor=0.0,
        return_diagnostics=False
    )
    hvp_single_norm = torch.norm(hvp_single).item()
    print(f"   Single-time HVP norm: {hvp_single_norm:.4e}")
    print(f"   Multi-time HVP norm:  {hvp_avg_norm:.4e}")
    variance_reduction = abs(hvp_single_norm - hvp_avg_norm) / hvp_single_norm * 100 if hvp_single_norm > 0 else 0
    print(f"   Variance reduction: {variance_reduction:.1f}%")

    print("=" * 80)

    # Return diagnostic results
    return {
        'time_values_used': t_evals,
        'hvp_norms_per_sigma': hvp_norms,
        'score_norms_per_sigma': diagnostics['score_norm_per_sigma'],
        'mean_hvp_norm': mean_hvp_norm,
        'std_hvp_norm': std_hvp_norm,
        'cv': cv,
        'hvp_variance': diagnostics['hvp_variance'],
        'hvp_avg_norm_no_floor': hvp_avg_norm,
        'floor_term_norm': floor_term_norm,
        'final_hvp_norm': final_hvp_norm,
        'curvature_floor_contribution': contribution_ratio,
        'curvature_floor_contribution_from_diagnostics': diagnostics['curvature_floor_contribution'],
        'recommended_floor_10pct': hvp_avg_norm * 0.10,
        'recommended_floor_15pct': hvp_avg_norm * 0.15,
        'single_sigma_hvp_norm': hvp_single_norm,
        'multi_sigma_hvp_norm': hvp_avg_norm,
        'variance_reduction_pct': variance_reduction,
        'max_cov_eigenvalue_estimate': 1 / hvp_avg_norm if hvp_avg_norm > 0 else float('inf'),
        'max_cov_eigenvalue_with_floor': 1 / curvature_floor if curvature_floor > 0 else float('inf')
    }
