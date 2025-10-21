# Diffuser Maze - Modular Structure

This directory contains a modularized version of the `diffuser_maze.ipynb` notebook for uncertainty quantification in diffusion models.

## Directory Structure

```
diffuser_maze/
├── __init__.py                   # Package initialization
├── config.py                     # Configuration and constants
├── models/                       # Neural network models
│   ├── precision_head.py         # PrecisionHead class (Cell 21)
│   └── normalizer.py             # TorchGaussianNormalizer (Cell 18)
├── data/                         # Dataset utilities
│   └── dataset.py                # MinariTrajectoryDataset (Cells 8, 12)
├── training/                     # Training infrastructure
│   ├── hvp.py                    # HVP computation (Cell 22)
│   ├── loss.py                   # Loss functions (Cell 26)
│   ├── monitor.py                # TrainingMonitor (Cell 24)
│   └── train.py                  # Training loop (Cell 27)
├── evaluation/                   # Evaluation utilities
│   ├── covariance.py             # Covariance computation (Cell 31)
│   ├── visualization.py          # Plotting functions (Cell 32)
│   ├── testing.py                # Structure tests (Cell 23)
│   └── evaluate.py               # Evaluation pipeline (Cell 34)
└── analysis/                     # Experimental analysis
    ├── ground_truth.py           # Ground truth Hessian (Cells 0, 1)
    └── diagnostics.py            # Quality diagnostics (Cells 2, 3)

scripts/
├── setup_diffuser.py             # Diffuser initialization (Cells 6, 15, 17-19)
├── train_precision_head.py       # Training entry point
└── eval_precision_head.py        # Evaluation entry point
```

## Code Migration Guide

Each file contains detailed comments indicating which notebook cells to copy. Follow this order:

### Step 1: Configuration (config.py)
Copy constants and configuration from:
- Cell 6: `device`
- Cell 10: `HORIZON`
- Cell 15: `TRANSITION_DIM`, diffuser config
- Cell 21: `TRAJ_DIM`, precision head config
- Cell 25: `precision_config` dict
- Cell 30: Numerical stability constants

### Step 2: Data Layer
**data/dataset.py:**
- Cell 8: `space_dim()` function
- Cell 12: `MinariTrajectoryDataset` class

### Step 3: Models
**models/normalizer.py:**
- Cell 18: `TorchGaussianNormalizer` class

**models/precision_head.py:**
- Cell 21: `PrecisionHead` class (entire class)

### Step 4: Training Infrastructure
**training/hvp.py:**
- Cell 22: `compute_hvp()` function
- Cell 22: `sample_trajectories_for_precision_training()` function

**training/loss.py:**
- Cell 26: `precision_loss_function()`
- Cell 26: `make_rademacher_probes()`
- Cell 26: `generate_training_batch()`

**training/monitor.py:**
- Cell 24: `TrainingMonitor` class (entire class)

**training/train.py:**
- Cell 27: Training loop (adapt as function)

### Step 5: Evaluation
**evaluation/covariance.py:**
- Cell 31: `full_cov_from_factors_np()`
- Cell 31: `blocks_from_cov_np()`
- Cell 31: `unnormalize_block()`

**evaluation/visualization.py:**
- Cell 32: `plot_trajectory_with_uncertainty_tubes()`

**evaluation/testing.py:**
- Cell 23: `test_precision_head_properties()`
# TODO: This step ===================================
**evaluation/evaluate.py:**
- Cell 34: `get_module_device()`
- Cell 34: Evaluation pipeline (adapt as function)

### Step 6: Scripts
**scripts/setup_diffuser.py:**
- Cells 6, 10, 11, 15, 17-19: Setup code

**scripts/train_precision_head.py:**
- Cell 25: Optimizer/scheduler setup
- Integrate with training modules

**scripts/eval_precision_head.py:**
- Cell 34: Adapt evaluation code

### Step 7: Optional - Analysis
**analysis/ground_truth.py:**
- Cell 1: `analyze_ground_truth_hessian()`
- Cell 0: Ground truth comparison script

**analysis/diagnostics.py:**
- Cell 3: `diagnose_precision_head_quality()`
- Cell 2: Diagnostic execution script

## Usage After Migration

### Training
```bash
python scripts/train_precision_head.py \
    --dataset_id "pointmaze/my-rrt-umaze-large-v0" \
    --diffuser_checkpoint "./results/diffuser_maze_znorm/model_ckpt/model_best.pt" \
    --save_dir "./results/precision_head/" \
    --device "cuda:5" \
    --num_epochs 400
```

### Evaluation
```bash
python scripts/eval_precision_head.py \
    --dataset_id "pointmaze/my-rrt-umaze-large-v0" \
    --diffuser_checkpoint "./results/diffuser_maze_znorm/model_ckpt/model_best.pt" \
    --precision_checkpoint "./results/precision_head/precision_head_epoch_400.pth" \
    --device "cuda:5" \
    --num_samples 16 \
    --num_show 4
```

### Programmatic Usage
```python
from diffuser_maze.models.precision_head import PrecisionHead
from diffuser_maze.training.train import train_precision_head
from scripts.setup_diffuser import setup_diffuser

# Setup
diffuser, dataloader, normalizer, device = setup_diffuser(...)

# Initialize model
precision_head = PrecisionHead(
    traj_dim=512,
    state_dim=2,
    horizon=256,
    hidden_dim=256,
    low_rank_dim=16,
    use_low_rank=True
).to(device)

# Train
train_precision_head(
    precision_head=precision_head,
    diffuser=diffuser,
    traj_dataloader=dataloader,
    normalizer=normalizer,
    ...
)
```

## Key Design Principles

1. **Modular**: Each component is independently importable
2. **No Global State**: All dependencies passed explicitly
3. **Reusable**: Components can be used in different contexts
4. **Testable**: Clear interfaces for unit testing
5. **Documented**: Each file has detailed docstrings

## Notes

- Each file contains TODO comments indicating exactly what to copy
- Cell numbers are referenced for easy lookup
- All imports and dependencies are noted
- Preserve the exact code structure when copying
- Test each module after migration to ensure correctness

## Dependencies

- torch
- numpy
- matplotlib
- minari
- gymnasium
- cleandiffuser

## Next Steps

1. Copy code from notebook cells to corresponding files
2. Uncomment import statements in `__init__.py` files
3. Test each module independently
4. Run training script to verify integration
5. Run evaluation script to verify end-to-end pipeline
