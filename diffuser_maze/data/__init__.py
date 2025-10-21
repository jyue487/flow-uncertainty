"""
Dataset utilities for trajectory data.

Includes:
- MinariTrajectoryDataset: Load real trajectory data from Minari
- Marginal Gaussian datasets: Synthetic datasets with known ground-truth covariances
"""

# from .dataset import MinariTrajectoryDataset, space_dim

# Marginal Gaussian datasets for uncertainty quantification
from .marginal_datasets import (
    # 2D Isotropic dataset
    sample_marginal_gaussian_dataset,
    MarginalGaussianDataset,
    build_ground_truth_covariances,
    # 1D Y-only noise dataset
    sample_1d_gaussian_dataset,
    OneDGaussianDataset,
    build_1d_mean_path,
    build_1d_covariances,
    # Visualization
    plot_1d_dataset_overview,
    plot_1d_variance_comparison,
    # Evaluation
    extract_y_std_from_covariance,
    evaluate_1d_precision_accuracy,
    plot_precision_comparison_1d,
    # Utilities
    marginal_variance_schedule,
    chi2_ppf_df2
)
