"""
Structure validation tests for precision head.

Copy from notebook Cell 23 - test_precision_head_properties()

This function validates mathematical correctness of precision head output:
1. L is globally lower triangular
2. L has correct block-bidiagonal structure
3. Diagonal blocks have positive diagonal elements
4. Precision matrix is symmetric positive definite
5. Block structure validation

Dependencies:
- numpy
- matplotlib
- torch

Usage:
    from diffuser_maze.evaluation.testing import test_precision_head_properties

    test_results = test_precision_head_properties(
        precision_head=precision_head,
        num_test_samples=5
    )

    # Check results
    if all(test_results.values()):
        print("All tests passed!")
"""

# TODO: Copy from Cell 23 - test_precision_head_properties()
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffuser_maze.config import HORIZON, TRAJ_DIM, device, precision_savepath
import os
#
# def test_precision_head_properties(precision_head, num_test_samples=5, traj_dim=512):
#     """
#     Validate mathematical correctness of precision head output.
#
#     Tests performed:
#     1. L is globally lower triangular
#     2. L has correct block-bidiagonal structure
#     3. Diagonal blocks have positive diagonal elements
#     4. Precision matrix is symmetric
#     5. Precision matrix is positive definite
#     6. Block structure validation
#
#     Args:
#         precision_head: PrecisionHead model
#         num_test_samples: Number of random inputs to test
#         traj_dim: Trajectory dimension (2*HORIZON)
#
#     Returns:
#         test_results: dict with boolean values for each test
#     """
def test_precision_head_properties(precision_head, num_test_samples=5, visualize=True):
    """
    Test that the Precision Head produces mathematically correct structures:
    1. L is block lower-bidiagonal
    2. L is lower-triangular with positive diagonal entries in each block
    3. Precision matrix is SPD and block-tridiagonal (when R is small)
    
    Args:
        precision_head: The PrecisionHead model to test
        num_test_samples: Number of random trajectories to test
        visualize: Whether to plot the matrix structures
    
    Returns:
        test_results: Dict with detailed test outcomes
    """
    print("=" * 70)
    print("Testing Precision Head Mathematical Properties")
    print("=" * 70)
    
    test_results = {
        'all_passed': True,
        'L_tests': {},
        'precision_tests': {},
        'samples': []
    }
    
    # Generate test trajectories
    test_trajectories = torch.randn(num_test_samples, TRAJ_DIM, device=device)
    
    for sample_idx in range(num_test_samples):
        print(f"\nTesting sample {sample_idx + 1}/{num_test_samples}...")
        
        # Get single trajectory
        traj = test_trajectories[sample_idx:sample_idx+1]
        
        # Forward pass
        with torch.no_grad():
            precision_matrix, L_matrix, components = precision_head(traj)
        
        # Convert to numpy for analysis
        L_np = L_matrix[0].cpu().numpy()
        P_np = precision_matrix[0].cpu().numpy()
        
        sample_results = {
            'L_matrix': L_np,
            'precision_matrix': P_np,
            'tests': {}
        }
        
        # ====== Test 1: L is globally lower triangular ======
        print("  1. Testing L is lower triangular...")
        upper_triangle_sum = np.sum(np.abs(np.triu(L_np, k=1)))
        is_lower_triangular = upper_triangle_sum < 1e-10
        sample_results['tests']['L_lower_triangular'] = {
            'passed': is_lower_triangular,
            'upper_sum': upper_triangle_sum
        }
        if is_lower_triangular:
            print(f"    ✅ L is lower triangular (upper sum: {upper_triangle_sum:.2e})")
        else:
            print(f"    ❌ L is NOT lower triangular (upper sum: {upper_triangle_sum:.2e})")
            test_results['all_passed'] = False
        
        # ====== Test 2: L is block lower-bidiagonal ======
        print("  2. Testing L block structure...")
        horizon = precision_head.horizon
        block_size = 2
        
        # Check each block position
        blocks_correct = True
        block_errors = []
        
        for i in range(horizon):
            for j in range(horizon):
                i_start, i_end = i * block_size, (i + 1) * block_size
                j_start, j_end = j * block_size, (j + 1) * block_size
                block = L_np[i_start:i_end, j_start:j_end]
                block_norm = np.linalg.norm(block)
                
                # Block should be non-zero only on diagonal and first sub-diagonal
                if i == j:  # Diagonal block
                    if block_norm < 1e-10:
                        blocks_correct = False
                        block_errors.append(f"Diagonal block ({i},{j}) is zero")
                elif i == j + 1:  # Sub-diagonal block
                    # Sub-diagonal blocks can be non-zero
                    pass
                else:  # All other blocks should be zero
                    if block_norm > 1e-10:
                        blocks_correct = False
                        block_errors.append(f"Block ({i},{j}) should be zero but has norm {block_norm:.2e}")
        
        sample_results['tests']['L_block_bidiagonal'] = {
            'passed': blocks_correct,
            'errors': block_errors
        }
        
        if blocks_correct:
            print("    ✅ L has correct block-bidiagonal structure")
        else:
            print(f"    ❌ L block structure incorrect: {block_errors[:3]}")  # Show first 3 errors
            test_results['all_passed'] = False
        
        # ====== Test 3: Diagonal blocks are lower triangular with positive diagonal ======
        print("  3. Testing diagonal block properties...")
        diag_blocks_correct = True
        diag_block_errors = []
        
        l_diag_blocks = components['l_diagonal'][0].cpu().numpy()  # [T, 2, 2]
        
        for t in range(horizon):
            block = l_diag_blocks[t]
            
            # Check lower triangular (top-right should be zero)
            if abs(block[0, 1]) > 1e-10:
                diag_blocks_correct = False
                diag_block_errors.append(f"Block {t}: not lower triangular (top-right = {block[0, 1]:.2e})")
            
            # Check positive diagonal
            if block[0, 0] <= 0:
                diag_blocks_correct = False
                diag_block_errors.append(f"Block {t}: diagonal[0,0] = {block[0, 0]:.2e} <= 0")
            if block[1, 1] <= 0:
                diag_blocks_correct = False
                diag_block_errors.append(f"Block {t}: diagonal[1,1] = {block[1, 1]:.2e} <= 0")
        
        sample_results['tests']['diagonal_blocks'] = {
            'passed': diag_blocks_correct,
            'errors': diag_block_errors
        }
        
        if diag_blocks_correct:
            print(f"    ✅ All {horizon} diagonal blocks are lower triangular with positive diagonals")
        else:
            print(f"    ❌ Diagonal block issues: {diag_block_errors[:3]}")
            test_results['all_passed'] = False
        
        # ====== Test 4: Precision matrix is symmetric ======
        print("  4. Testing precision matrix symmetry...")
        symmetry_error = np.linalg.norm(P_np - P_np.T) / np.linalg.norm(P_np)
        is_symmetric = symmetry_error < 1e-10
        
        sample_results['tests']['precision_symmetric'] = {
            'passed': is_symmetric,
            'error': symmetry_error
        }
        
        if is_symmetric:
            print(f"    ✅ Precision matrix is symmetric (error: {symmetry_error:.2e})")
        else:
            print(f"    ❌ Precision matrix NOT symmetric (error: {symmetry_error:.2e})")
            test_results['all_passed'] = False
        
        # ====== Test 5: Precision matrix is positive definite ======
        print("  5. Testing precision matrix positive definiteness...")
        eigenvalues = np.linalg.eigvalsh(P_np)  # Use eigvalsh for symmetric matrices
        min_eigenvalue = np.min(eigenvalues)
        max_eigenvalue = np.max(eigenvalues)
        condition_number = max_eigenvalue / min_eigenvalue if min_eigenvalue > 0 else np.inf
        is_positive_definite = min_eigenvalue > 0
        
        sample_results['tests']['precision_positive_definite'] = {
            'passed': is_positive_definite,
            'min_eigenvalue': min_eigenvalue,
            'max_eigenvalue': max_eigenvalue,
            'condition_number': condition_number
        }
        
        if is_positive_definite:
            print(f"    ✅ Precision matrix is positive definite")
            print(f"       Min eigenvalue: {min_eigenvalue:.2e}")
            print(f"       Max eigenvalue: {max_eigenvalue:.2e}")
            print(f"       Condition number: {condition_number:.2e}")
        else:
            print(f"    ❌ Precision matrix NOT positive definite (min eigenvalue: {min_eigenvalue:.2e})")
            test_results['all_passed'] = False
        
        # ====== Test 6: Precision matrix block structure (when R is small) ======
        print("  6. Testing precision matrix block-tridiagonal structure...")
        
        # Check block norms to see structure
        block_norms = np.zeros((horizon, horizon))
        for i in range(horizon):
            for j in range(horizon):
                i_start, i_end = i * block_size, (i + 1) * block_size
                j_start, j_end = j * block_size, (j + 1) * block_size
                block = P_np[i_start:i_end, j_start:j_end]
                block_norms[i, j] = np.linalg.norm(block)
        
        # Check if approximately block-tridiagonal (allowing for R component)
        total_norm = np.sum(block_norms)
        tridiag_norm = 0.0
        for i in range(horizon):
            for j in range(max(0, i-1), min(horizon, i+2)):
                tridiag_norm += block_norms[i, j]
        
        tridiag_ratio = tridiag_norm / total_norm if total_norm > 0 else 0
        is_mostly_tridiagonal = tridiag_ratio > 0.95  # 95% of norm in tridiagonal blocks
        
        sample_results['tests']['precision_block_tridiagonal'] = {
            'passed': is_mostly_tridiagonal,
            'tridiag_ratio': tridiag_ratio,
            'block_norms': block_norms
        }
        
        if is_mostly_tridiagonal:
            print(f"    ✅ Precision matrix is approximately block-tridiagonal ({tridiag_ratio:.1%} of norm)")
        else:
            print(f"    ⚠️  Precision matrix has significant off-tridiagonal components ({tridiag_ratio:.1%} of norm)")
            # This is a warning, not a failure, since R adds full structure
        
        # Store sample results
        test_results['samples'].append(sample_results)
    
    # ====== Visualization ======
    if visualize and num_test_samples > 0:
        print("\n" + "=" * 40)
        print("Visualizing matrix structures...")
        
        # Use first sample for visualization
        L_vis = test_results['samples'][0]['L_matrix']
        P_vis = test_results['samples'][0]['precision_matrix']
        block_norms = test_results['samples'][0]['tests']['precision_block_tridiagonal']['block_norms']
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Plot 1: L matrix structure
        ax = axes[0]
        im = ax.imshow(np.log10(np.abs(L_vis) + 1e-10), cmap='viridis', aspect='auto')
        ax.set_title('L Matrix (log scale)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)
        
        # Plot 2: L block structure (zoomed to first 20 timesteps)
        ax = axes[1]
        zoom_size = min(40, L_vis.shape[0])  # Show first 20 timesteps = 40 dimensions
        L_zoom = L_vis[:zoom_size, :zoom_size]
        im = ax.imshow(np.abs(L_zoom), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(f'L Matrix (first {zoom_size//2} timesteps)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Add grid to show 2x2 blocks
        for i in range(0, zoom_size+1, 2):
            ax.axhline(i-0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(i-0.5, color='gray', linewidth=0.5, alpha=0.5)
        plt.colorbar(im, ax=ax)
        
        # Plot 3: Precision matrix structure
        ax = axes[2]
        im = ax.imshow(np.log10(np.abs(P_vis) + 1e-10), cmap='viridis', aspect='auto')
        ax.set_title('Precision Matrix (log scale)')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)
        
        # Plot 4: Block norm structure
        ax = axes[3]
        im = ax.imshow(np.log10(block_norms + 1e-10), cmap='hot', aspect='auto')
        ax.set_title('Block Norms (log scale)')
        ax.set_xlabel('Block Column')
        ax.set_ylabel('Block Row')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(precision_savepath, "precision_head_structure_test.png"), dpi=150)
        plt.show()
    
    # ====== Summary ======
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    # Count passed tests
    total_tests = 0
    passed_tests = 0
    
    for sample in test_results['samples']:
        for test_name, test_result in sample['tests'].items():
            total_tests += 1
            if test_result['passed']:
                passed_tests += 1
    
    print(f"Total tests: {passed_tests}/{total_tests} passed")
    
    if test_results['all_passed']:
        print("✅ ALL TESTS PASSED! Precision head produces correct mathematical structure.")
    else:
        print("❌ Some tests failed. Review the structure before training.")
    
    print("=" * 70)
    
    return test_results
