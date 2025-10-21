"""
Visualization utilities for uncertainty quantification.

Copy from notebook Cell 32 - plot_trajectory_with_uncertainty_tubes()

This function creates visualizations of trajectories with uncertainty ellipses:
- Left panel: Trajectory with n-sigma ellipses
- Right panel: Ellipse radii evolution over time

Dependencies:
- matplotlib
- matplotlib.patches.Ellipse
- numpy

Usage:
    from diffuser_maze.evaluation.visualization import plot_trajectory_with_uncertainty_tubes

    plot_trajectory_with_uncertainty_tubes(
        trajectory_positions=traj,  # [T, 2]
        covariance_blocks=blocks,   # [T, 2, 2]
        n_std=2.0,
        stride=1,
        title="Trajectory with 2-sigma uncertainty"
    )
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math
#
# def plot_trajectory_with_uncertainty_tubes(
#     trajectory_positions,
#     covariance_blocks,
#     n_std=2.0,
#     stride=1,
#     title="Trajectory with Uncertainty"
# ):
#     """
#     Visualize trajectory with uncertainty ellipses.
#
#     Creates two panels:
#     - Left: Trajectory with n-sigma ellipses at each timestep
#     - Right: Evolution of major/minor axes over time
#
#     Ellipse computation:
#     - Eigenvalue decomposition of covariance
#     - Major/minor axes: 2 * n_std * sqrt(eigenvalue)
#     - Orientation: angle of major eigenvector
#
#     Args:
#         trajectory_positions: [T, 2] numpy array of (x, y) positions
#         covariance_blocks: [T, 2, 2] numpy array of covariance blocks
#         n_std: Number of standard deviations for ellipse
#         stride: Plot every stride-th ellipse
#         title: Plot title
#     """
def plot_trajectory_with_uncertainty_tubes(trajectory_positions, covariance_blocks, 
                                           n_std=2.0, stride=1, title="Trajectory with Uncertainty"):
    """
    Visualize trajectory with uncertainty ellipses (no calibration).
    
    Args:
        trajectory_positions: [T, 2] numpy array of (x, y) positions in raw space
        covariance_blocks: [T, 2, 2] numpy array of per-timestep covariance in raw space
        n_std: Radius in standard deviations (e.g., 2.0 for ~95% coverage under Gaussian assumption)
        stride: Plot ellipse every stride timesteps to avoid clutter
        title: Plot title
    
    Creates two-panel figure:
        - Left: Trajectory with uncertainty ellipses
        - Right: Evolution of ellipse radii over time
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: Trajectory with uncertainty ellipses
    T = len(trajectory_positions)
    ax1.plot(trajectory_positions[:, 0], trajectory_positions[:, 1], 
             'b-', lw=2, label='Trajectory', zorder=3)
    ax1.scatter(trajectory_positions[0, 0], trajectory_positions[0, 1], 
                c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(trajectory_positions[-1, 0], trajectory_positions[-1, 1], 
                c='red', s=100, marker='*', label='Goal', zorder=5)
    
    # Plot uncertainty ellipses
    for t in range(0, T, stride):
        S = covariance_blocks[t]
        
        # Eigenvalue decomposition for ellipse parameters
        w, V = np.linalg.eigh(S)
        w = np.clip(w, 1e-12, None)  # Ensure positive
        
        # Ellipse dimensions: semi-axes = n_std * sqrt(eigenvalue)
        width = 2 * n_std * np.sqrt(w[1])   # Major axis (larger eigenvalue)
        height = 2 * n_std * np.sqrt(w[0])  # Minor axis (smaller eigenvalue)
        
        # Ellipse orientation: angle of major eigenvector
        angle = math.degrees(math.atan2(V[1, 1], V[0, 1]))
        
        # Create and add ellipse patch
        e = Ellipse(xy=trajectory_positions[t], width=width, height=height, 
                   angle=angle, facecolor='none', edgecolor='C0', lw=1.0, alpha=0.6, zorder=2)
        ax1.add_patch(e)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'{title} ({n_std}σ ellipses)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', 'box')
    
    # Right panel: Uncertainty evolution over time
    major_radii = []
    minor_radii = []
    
    for t in range(T):
        w, _ = np.linalg.eigh(covariance_blocks[t])
        w = np.clip(w, 1e-12, None)
        major_radii.append(n_std * np.sqrt(w[1]))
        minor_radii.append(n_std * np.sqrt(w[0]))
    
    timesteps = np.arange(T)
    ax2.plot(timesteps, major_radii, label='Major axis', lw=1.5, color='C0')
    ax2.plot(timesteps, minor_radii, label='Minor axis', lw=1.5, color='C1')
    ax2.fill_between(timesteps, minor_radii, major_radii, alpha=0.2, color='C0')
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel(f'Ellipse Radius ({n_std}σ)')
    ax2.set_title('Uncertainty Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
