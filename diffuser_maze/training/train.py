"""
Main training loop for precision head.

This module contains the main training function that:
1. Generates training batches
2. Computes predicted and true HVPs
3. Calculates loss
4. Performs optimization step
5. Monitors training health
6. Saves checkpoints

Dependencies:
- torch
- torch.nn.utils (for gradient clipping)
- All training submodules

Usage:
    from diffuser_maze.training.train import train_precision_head

    train_precision_head(
        precision_head=precision_head,
        diffuser=diffuser,
        optimizer=optimizer,
        scheduler=scheduler,
        monitor=monitor,
        config=precision_config,
        save_dir="./results/precision_head/"
    )
"""

import os
import numpy as np
import torch
import torch.nn.utils as nn_utils
from tqdm import tqdm
from .hvp import compute_hvp
from .loss import generate_training_batch, precision_loss_function


def train_precision_head(
    precision_head,
    diffuser,
    optimizer,
    scheduler,
    monitor,
    config,
    save_dir
):
    """
    Train the precision head model with comprehensive monitoring.

    Copied from diffuser_maze.ipynb Cell 27 (lines 2784-3020)

    Args:
        precision_head: PrecisionHead model to train
        diffuser: Frozen diffusion model for computing true HVPs
        optimizer: torch optimizer (AdamW)
        scheduler: Learning rate scheduler (CosineAnnealingWarmRestarts)
        monitor: TrainingMonitor instance for diagnostics
        config: Training configuration dict (precision_config)
        save_dir: Directory to save checkpoints and monitoring data
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("TRAINING PRECISION HEAD")
    print("=" * 70)

    # Training metrics
    training_losses = []
    epoch_logs = []

    # Set models to appropriate modes
    diffuser.eval()  # Keep diffuser frozen for score computation
    precision_head.train()

    # Training loop with monitoring
    pbar = tqdm(range(config['num_epochs']), desc="Training Precision Head")

    try:
        for epoch in pbar:
            epoch_losses = []
            epoch_loss_components = {
                'consistency': 0.0,
                'l2_reg_r': 0.0,
                'temporal_smooth': 0.0,
                'pd_reg': 0.0,
                'total': 0.0
            }

            # Storage for epoch monitoring
            epoch_grad_stats = []
            epoch_loss_analyses = []
            epoch_matrix_healths = []

            # Training steps per epoch
            steps_per_epoch = 10  # Number of batches per epoch

            for step in range(steps_per_epoch):
                optimizer.zero_grad()

                # Generate training batch
                trajectories, positions_flat, probe_vectors = generate_training_batch(
                    diffuser,
                    batch_size=config['batch_size'],
                    config=config
                )

                batch_loss = 0.0
                batch_loss_components = {key: 0.0 for key in epoch_loss_components.keys()}
                batch_predicted_hvp = None
                batch_true_hvp = None
                batch_components = None

                # Process each probe vector for the batch
                for probe_idx in range(config['num_probes']):
                    probe_vec = probe_vectors[:, probe_idx]  # [B, 2T]

                    # Forward through precision head (updated API)
                    precision_matrix, L_matrix, components = precision_head(positions_flat)

                    # Apply precision matrix to probe vector
                    predicted_hvp = precision_head.apply_to_probe(positions_flat, probe_vec)

                    # Compute true HVP using score function
                    true_hvp, _ = compute_hvp(diffuser, trajectories, probe_vec, use_noised_trajectory=False)

                    # Compute loss for this probe (updated function signature)
                    probe_loss, probe_loss_components = precision_loss_function(
                        predicted_hvp, true_hvp, components, config
                    )

                    # Accumulate losses
                    batch_loss += probe_loss / config['num_probes']
                    for key in batch_loss_components:
                        batch_loss_components[key] += probe_loss_components[key] / config['num_probes']

                    # Store for monitoring (use first probe)
                    if probe_idx == 0:
                        batch_predicted_hvp = predicted_hvp.detach()
                        batch_true_hvp = true_hvp.detach()
                        batch_components = components
                        batch_precision_matrix = precision_matrix.detach()
                        batch_L_matrix = L_matrix.detach()

                # Backward pass
                batch_loss.backward()

                # MONITORING: Analyze gradients after backward pass
                grad_stats = monitor.analyze_gradients(precision_head, f"epoch_{epoch}_step_{step}")
                epoch_grad_stats.append(grad_stats)

                # Check for gradient explosion and adjust
                if grad_stats['total_norm'] > monitor.grad_explosion_threshold:
                    print(f"\n⚠️  Gradient explosion detected! Norm: {grad_stats['total_norm']:.2e}")
                    # Scale down gradients more aggressively
                    torch.nn.utils.clip_grad_norm_(
                        precision_head.parameters(),
                        max_norm=0.1  # Very aggressive clipping
                    )
                else:
                    # Normal gradient clipping
                    if config['grad_clip_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            precision_head.parameters(),
                            config['grad_clip_norm']
                        )

                optimizer.step()

                # MONITORING: Analyze loss components and matrix health
                if batch_predicted_hvp is not None:
                    loss_analysis = monitor.analyze_loss_components(
                        batch_loss_components, batch_predicted_hvp, batch_true_hvp
                    )
                    matrix_health = monitor.analyze_precision_matrix_health(
                        batch_precision_matrix, batch_L_matrix
                    )
                    epoch_loss_analyses.append(loss_analysis)
                    epoch_matrix_healths.append(matrix_health)

                # Accumulate epoch statistics
                epoch_losses.append(batch_loss.item())
                for key in epoch_loss_components:
                    epoch_loss_components[key] += batch_loss_components[key] / steps_per_epoch

                # Early stopping for catastrophic failures
                if batch_loss.item() > 1e10:
                    print(f"\n🛑 Catastrophic loss detected: {batch_loss.item():.2e}")
                    print("Stopping training to prevent further damage.")
                    break

            # Update learning rate
            scheduler.step()

            # Compute epoch statistics
            epoch_avg_loss = np.mean(epoch_losses)
            training_losses.append(epoch_avg_loss)

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{epoch_avg_loss:.2e}",
                'Consistency': f"{epoch_loss_components['consistency']:.2e}",
                'Grad': f"{epoch_grad_stats[-1]['total_norm']:.2e}" if epoch_grad_stats else "N/A",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

            # Detailed monitoring
            if (epoch + 1) % config.get('monitor_every', 1) == 0:
                if epoch_grad_stats and epoch_loss_analyses and epoch_matrix_healths:
                    latest_grad = epoch_grad_stats[-1]
                    latest_loss = epoch_loss_analyses[-1]
                    latest_health = epoch_matrix_healths[-1]

                    # Print detailed monitoring
                    monitor.print_monitoring_summary(epoch + 1, latest_grad, latest_loss, latest_health)

                    # Plot dashboard every 10 epochs
                    if (epoch + 1) % 10 == 0:
                        dashboard_path = os.path.join(save_dir, f"monitoring_dashboard_epoch_{epoch+1}.png")
                        monitor.plot_monitoring_dashboard(save_path=dashboard_path)

            # Logging
            if (epoch + 1) % config['log_every'] == 0:
                tqdm.write(f"\nEpoch {epoch + 1}/{config['num_epochs']}:")
                tqdm.write(f"  Average Loss: {epoch_avg_loss:.2e}")
                tqdm.write(f"  Loss Components:")
                for key, value in epoch_loss_components.items():
                    if key != 'total':
                        tqdm.write(f"    {key:20s}: {value:.2e}")

                # Print warnings if there are issues
                if epoch_grad_stats and epoch_loss_analyses and epoch_matrix_healths:
                    all_issues = (epoch_grad_stats[-1]['issues'] +
                                epoch_loss_analyses[-1]['balance_issues'] +
                                epoch_matrix_healths[-1]['issues'])
                    if all_issues:
                        tqdm.write(f"  ⚠️  Issues: {', '.join(all_issues[:3])}")

            # Save checkpoint
            if (epoch + 1) % config['save_every'] == 0:
                checkpoint_path = os.path.join(save_dir, f"precision_head_epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': precision_head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': epoch_avg_loss,
                    'training_losses': training_losses,
                    'config': config,
                    'monitoring_data': {
                        'gradient_history': monitor.gradient_history,
                        'loss_component_history': monitor.loss_component_history,
                        'matrix_health_history': monitor.matrix_health_history
                    }
                }, checkpoint_path)
                tqdm.write(f"  💾 Saved checkpoint: {checkpoint_path}")

            # Store epoch log
            epoch_log = {
                'epoch': epoch + 1,
                'loss': epoch_avg_loss,
                'loss_components': epoch_loss_components.copy(),
                'lr': optimizer.param_groups[0]['lr']
            }
            epoch_logs.append(epoch_log)

            # Adaptive learning rate adjustment for numerical issues
            if epoch_matrix_healths and epoch_matrix_healths[-1]['precision_matrix']['condition_number'] > monitor.condition_number_threshold:
                # Reduce learning rate for numerical stability
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                tqdm.write(f"  🔄 Reduced LR to {param_group['lr']:.2e} due to conditioning")

    finally:
        pbar.close()

        # Save final model with comprehensive monitoring data
        final_path = os.path.join(save_dir, "precision_head_final.pt")
        torch.save({
            'epoch': len(training_losses),
            'model_state_dict': precision_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': training_losses[-1] if training_losses else float('inf'),
            'training_losses': training_losses,
            'epoch_logs': epoch_logs,
            'config': config,
            'monitoring_data': {
                'gradient_history': monitor.gradient_history,
                'loss_component_history': monitor.loss_component_history,
                'matrix_health_history': monitor.matrix_health_history
            }
        }, final_path)
        print(f"\n💾 Saved final model: {final_path}")

        # Save final monitoring dashboard
        final_dashboard_path = os.path.join(save_dir, "training_final_dashboard.png")
        monitor.plot_monitoring_dashboard(save_path=final_dashboard_path)
        print(f"📊 Saved final dashboard: {final_dashboard_path}")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
