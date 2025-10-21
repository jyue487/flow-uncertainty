"""
Training monitoring and diagnostics.

Copy from notebook Cell 24 - TrainingMonitor class.

This class provides comprehensive training diagnostics:
- Gradient analysis (explosion/vanishing detection)
- Loss component tracking
- Matrix health monitoring (conditioning, eigenvalues)
- HVP quality metrics (cosine similarity, magnitude ratios)
- Adaptive learning rate adjustment
- Visualization dashboards

Dependencies:
- torch
- matplotlib

Usage:
    from diffuser_maze.training.monitor import TrainingMonitor

    monitor = TrainingMonitor(precision_head, config)

    # During training
    grad_stats = monitor.analyze_gradients(precision_head)
    loss_analysis = monitor.analyze_loss_components(
        predicted_hvp, true_hvp, loss_components
    )
    matrix_health = monitor.check_matrix_health(precision_matrix, L_matrix)

    # Epoch summary
    monitor.log_epoch(epoch, avg_loss, grad_stats, ...)

    # Save dashboard
    monitor.save_monitoring_dashboard(save_path)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
#
# class TrainingMonitor:
#     """
#     Comprehensive training diagnostics and health monitoring.
#
#     Tracks:
#     - Gradient statistics per component
#     - Loss balancing
#     - Matrix conditioning and eigenvalues
#     - HVP quality metrics
#     """
class TrainingMonitor:
    """Comprehensive training monitor for debugging large losses and training issues."""
    
    def __init__(self, precision_head, config):
        self.precision_head = precision_head
        self.config = config
        self.epoch_logs = []
        
        # Monitoring thresholds
        self.grad_explosion_threshold = 10.0
        self.grad_vanishing_threshold = 1e-7
        self.loss_explosion_threshold = 1e6
        self.condition_number_threshold = 1e12
        
        # Storage for monitoring data
        self.gradient_history = []
        self.loss_component_history = []
        self.matrix_health_history = []
        self.hvp_quality_history = []
        
    def analyze_gradients(self, model, step_info=""):
        """Analyze gradient statistics across all parameters."""
        grad_stats = {
            'step': step_info,
            'parameter_groups': {},
            'total_norm': 0.0,
            'issues': []
        }
        
        # Group parameters by component
        param_groups = {
            'encoder': [],
            'l_diagonal': [],
            'l_subdiag': [], 
            'r_matrix': []
        }
        
        # Collect parameters by group
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'encoder' in name:
                    param_groups['encoder'].append(param.grad)
                elif 'l_diagonal' in name:
                    param_groups['l_diagonal'].append(param.grad)
                elif 'l_subdiag' in name:
                    param_groups['l_subdiag'].append(param.grad)
                elif 'r_head' in name:
                    param_groups['r_matrix'].append(param.grad)
        
        # Analyze each group
        for group_name, grads in param_groups.items():
            if grads:
                # Concatenate all gradients in group
                flat_grads = torch.cat([g.flatten() for g in grads])
                
                group_stats = {
                    'norm': torch.norm(flat_grads).item(),
                    'mean': torch.mean(flat_grads).item(),
                    'std': torch.std(flat_grads).item(),
                    'min': torch.min(flat_grads).item(),
                    'max': torch.max(flat_grads).item(),
                    'num_params': len(flat_grads)
                }
                
                grad_stats['parameter_groups'][group_name] = group_stats
                grad_stats['total_norm'] += group_stats['norm'] ** 2
        
        grad_stats['total_norm'] = grad_stats['total_norm'] ** 0.5
        
        # Check for issues
        if grad_stats['total_norm'] > self.grad_explosion_threshold:
            grad_stats['issues'].append(f"Gradient explosion: {grad_stats['total_norm']:.2e}")
        elif grad_stats['total_norm'] < self.grad_vanishing_threshold:
            grad_stats['issues'].append(f"Gradient vanishing: {grad_stats['total_norm']:.2e}")
            
        self.gradient_history.append(grad_stats)
        return grad_stats
    
    def analyze_loss_components(self, loss_components, predicted_hvp, true_hvp):
        """Analyze loss component balance and HVP quality."""
        analysis = {
            'components': loss_components.copy(),
            'hvp_analysis': {},
            'balance_issues': []
        }
        
        # HVP quality analysis - compute per-sample metrics then average
        # predicted_hvp, true_hvp are [B, 2T]

        # Compute per-sample norms [B]
        pred_norms = torch.norm(predicted_hvp, dim=1)  # [B]
        true_norms = torch.norm(true_hvp, dim=1)       # [B]

        # Average norms across batch
        pred_norm = pred_norms.mean().item()
        true_norm = true_norms.mean().item()

        if true_norm > 1e-10:
            # Cosine similarity per sample between predicted and -true (the target)
            # F.cosine_similarity with dim=1 computes per-row similarity
            cosine_sims = F.cosine_similarity(
                predicted_hvp,   # [B, 2T]
                -true_hvp,       # [B, 2T] - negative because target is -h
                dim=1            # Compute per sample
            )  # [B]
            cosine_sim = cosine_sims.mean().item()

            # Magnitude ratio per sample (more meaningful than batch-level ratio)
            magnitude_ratios = pred_norms / (true_norms + 1e-10)  # [B]
            magnitude_ratio = magnitude_ratios.mean().item()

            # Also track std for better diagnostics
            cosine_sim_std = cosine_sims.std().item()
            magnitude_ratio_std = magnitude_ratios.std().item()

            analysis['hvp_analysis'] = {
                'predicted_norm': pred_norm,
                'true_norm': true_norm,
                'magnitude_ratio': magnitude_ratio,
                'magnitude_ratio_std': magnitude_ratio_std,
                'cosine_similarity': cosine_sim,
                'cosine_similarity_std': cosine_sim_std,
                'consistency_error': loss_components['consistency']
            }

            # Check for HVP issues
            if magnitude_ratio > 10 or magnitude_ratio < 0.1:
                analysis['balance_issues'].append(f"HVP magnitude mismatch: {magnitude_ratio:.2f}±{magnitude_ratio_std:.2f}")
            if cosine_sim < 0.1:
                analysis['balance_issues'].append(f"Poor HVP direction alignment: {cosine_sim:.3f}±{cosine_sim_std:.3f}")
        
        # Check loss component balance
        total_loss = loss_components['total']
        if total_loss > self.loss_explosion_threshold:
            analysis['balance_issues'].append(f"Loss explosion: {total_loss:.2e}")
        
        consistency_ratio = loss_components['consistency'] / total_loss if total_loss > 0 else 0
        if consistency_ratio < 0.5:
            analysis['balance_issues'].append(f"Consistency loss too small: {consistency_ratio:.1%}")
        elif consistency_ratio > 0.99:
            analysis['balance_issues'].append(f"Regularization too weak: {consistency_ratio:.1%}")
        
        self.loss_component_history.append(analysis)
        return analysis
    
    def analyze_precision_matrix_health(self, precision_matrix, L_matrix):
        """Analyze precision matrix numerical health."""
        P_np = precision_matrix[0].detach().cpu().numpy()
        L_np = L_matrix[0].detach().cpu().numpy()
        
        health = {
            'precision_matrix': {},
            'L_matrix': {},
            'issues': []
        }
        
        # Precision matrix analysis
        eigenvals = np.linalg.eigvalsh(P_np)
        min_eig = np.min(eigenvals)
        max_eig = np.max(eigenvals)
        condition_num = max_eig / min_eig if min_eig > 0 else np.inf
        
        health['precision_matrix'] = {
            'condition_number': condition_num,
            'min_eigenvalue': min_eig,
            'max_eigenvalue': max_eig,
            'frobenius_norm': np.linalg.norm(P_np, 'fro'),
            'symmetry_error': np.linalg.norm(P_np - P_np.T) / np.linalg.norm(P_np)
        }
        
        # L matrix analysis  
        L_diag_elements = np.diag(L_np)
        min_diag = np.min(L_diag_elements)
        
        health['L_matrix'] = {
            'min_diagonal': min_diag,
            'max_diagonal': np.max(L_diag_elements),
            'frobenius_norm': np.linalg.norm(L_np, 'fro'),
            'num_small_diagonals': np.sum(L_diag_elements < 1e-6)
        }
        
        # Check for issues
        if condition_num > self.condition_number_threshold:
            health['issues'].append(f"Ill-conditioned precision matrix: {condition_num:.2e}")
        if min_eig <= 0:
            health['issues'].append(f"Non-positive definite: min eigenvalue = {min_eig:.2e}")
        if min_diag <= 0:
            health['issues'].append(f"Non-positive L diagonal: {min_diag:.2e}")
        if health['precision_matrix']['symmetry_error'] > 1e-6:
            health['issues'].append(f"Asymmetric precision matrix: {health['precision_matrix']['symmetry_error']:.2e}")
        
        self.matrix_health_history.append(health)
        return health

    def analyze_multi_sigma_hvp(self, hvp_diagnostics):
        """
        Analyze multi-sigma HVP quality and variance.

        Args:
            hvp_diagnostics: Dict from compute_hvp with return_diagnostics=True

        Returns:
            analysis: Dict with multi-sigma statistics
        """
        analysis = {
            'time_values': hvp_diagnostics['time_values_used'],  # Changed from sigma_values_used
            'num_sigmas': len(hvp_diagnostics['time_values_used']),
            'score_norms': hvp_diagnostics['score_norm_per_sigma'],
            'hvp_norms': hvp_diagnostics['hvp_norm_per_sigma'],
            'hvp_variance': hvp_diagnostics['hvp_variance'],
            'curvature_floor_contribution': hvp_diagnostics['curvature_floor_contribution'],
            'issues': []
        }

        # Compute statistics
        hvp_norms = hvp_diagnostics['hvp_norm_per_sigma']
        if len(hvp_norms) > 1:
            analysis['hvp_norm_mean'] = np.mean(hvp_norms)
            analysis['hvp_norm_std'] = np.std(hvp_norms)
            analysis['hvp_norm_cv'] = analysis['hvp_norm_std'] / (analysis['hvp_norm_mean'] + 1e-10)

            # Check for high variance
            if analysis['hvp_norm_cv'] > 0.5:
                analysis['issues'].append(f"High HVP variance across sigmas: CV={analysis['hvp_norm_cv']:.2f}")

        # Check curvature floor contribution
        floor_ratio = hvp_diagnostics['curvature_floor_contribution']
        if floor_ratio > 0.5:
            analysis['issues'].append(f"Curvature floor dominates: {floor_ratio:.1%}")
        elif floor_ratio < 0.01:
            analysis['issues'].append(f"Curvature floor too weak: {floor_ratio:.1%}")

        # Store in history
        if not hasattr(self, 'multi_sigma_history'):
            self.multi_sigma_history = []
        self.multi_sigma_history.append(analysis)

        return analysis

    def print_monitoring_summary(self, epoch, grad_stats, loss_analysis, matrix_health, multi_sigma_analysis=None):
        """Print comprehensive monitoring summary."""
        print(f"\n📊 Epoch {epoch} Monitoring Summary:")
        print("=" * 50)
        
        # Gradient health
        print(f"🔄 Gradients:")
        print(f"   Total norm: {grad_stats['total_norm']:.2e}")
        for group, stats in grad_stats['parameter_groups'].items():
            print(f"   {group:12s}: norm={stats['norm']:.2e}, std={stats['std']:.2e}")
        if grad_stats['issues']:
            print(f"   ⚠️  Issues: {', '.join(grad_stats['issues'])}")
        
        # Loss components
        print(f"\n📈 Loss Analysis:")
        components = loss_analysis['components']
        total = components['total']
        print(f"   Total: {total:.2e}")
        print(f"   Consistency: {components['consistency']:.2e} ({components['consistency']/total:.1%})")
        print(f"   L2 R: {components['l2_reg_r']:.2e} ({components['l2_reg_r']/total:.1%})")
        print(f"   Temporal: {components['temporal_smooth']:.2e}")
        print(f"   PD Reg: {components['pd_reg']:.2e}")
        
        # HVP Quality
        if 'hvp_analysis' in loss_analysis and loss_analysis['hvp_analysis']:
            hvp = loss_analysis['hvp_analysis']
            print(f"\n🎯 HVP Quality (per-sample average):")
            if 'magnitude_ratio_std' in hvp:
                print(f"   Magnitude ratio: {hvp['magnitude_ratio']:.2f} ± {hvp['magnitude_ratio_std']:.2f}")
                print(f"   Cosine similarity: {hvp['cosine_similarity']:.3f} ± {hvp['cosine_similarity_std']:.3f}")
            else:
                # Backward compatibility
                print(f"   Magnitude ratio: {hvp['magnitude_ratio']:.2f}")
                print(f"   Cosine similarity: {hvp['cosine_similarity']:.3f}")
            print(f"   Predicted norm: {hvp['predicted_norm']:.2e}")
            print(f"   True norm: {hvp['true_norm']:.2e}")
        
        # Matrix health
        print(f"\n🏥 Matrix Health:")
        pm = matrix_health['precision_matrix']
        lm = matrix_health['L_matrix']
        print(f"   Condition number: {pm['condition_number']:.2e}")
        print(f"   Min eigenvalue: {pm['min_eigenvalue']:.2e}")
        print(f"   L min diagonal: {lm['min_diagonal']:.2e}")
        print(f"   Symmetry error: {pm['symmetry_error']:.2e}")

        # Multi-sigma HVP analysis
        if multi_sigma_analysis is not None:
            print(f"\n🔬 Multi-Sigma HVP Analysis:")
            print(f"   Time values (t): {multi_sigma_analysis['time_values']}")
            print(f"   HVP variance: {multi_sigma_analysis['hvp_variance']:.2e}")
            if 'hvp_norm_mean' in multi_sigma_analysis:
                print(f"   HVP norm: {multi_sigma_analysis['hvp_norm_mean']:.2e} ± {multi_sigma_analysis['hvp_norm_std']:.2e}")
                print(f"   Coefficient of variation: {multi_sigma_analysis['hvp_norm_cv']:.2f}")
            print(f"   Curvature floor contribution: {multi_sigma_analysis['curvature_floor_contribution']:.1%}")

        # Issues summary
        all_issues = grad_stats['issues'] + loss_analysis['balance_issues'] + matrix_health['issues']
        if multi_sigma_analysis is not None:
            all_issues += multi_sigma_analysis['issues']
        if all_issues:
            print(f"\n❌ Issues Found:")
            for issue in all_issues[:5]:  # Show top 5 issues
                print(f"   • {issue}")
        else:
            print(f"\n✅ No critical issues detected")

        print("=" * 50)
    
    def plot_monitoring_dashboard(self, save_path=None):
        """Create comprehensive monitoring dashboard with multi-sigma analysis."""
        if not self.gradient_history:
            return

        # Check if we have multi-sigma data
        has_multi_sigma = hasattr(self, 'multi_sigma_history') and len(self.multi_sigma_history) > 0

        # Use 3x3 grid if we have multi-sigma data, otherwise 2x3
        if has_multi_sigma:
            fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Monitoring Dashboard', fontsize=16)
        
        # Plot 1: Gradient norms over time
        ax = axes[0, 0]
        epochs = list(range(len(self.gradient_history)))
        total_norms = [g['total_norm'] for g in self.gradient_history]
        ax.plot(epochs, total_norms, 'b-', linewidth=2)
        ax.axhline(self.grad_explosion_threshold, color='r', linestyle='--', alpha=0.7, label='Explosion')
        ax.axhline(self.grad_vanishing_threshold, color='orange', linestyle='--', alpha=0.7, label='Vanishing')
        ax.set_yscale('log')
        ax.set_title('Gradient Norm Evolution')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Loss components over time
        ax = axes[0, 1]
        if self.loss_component_history:
            consistency_losses = [l['components']['consistency'] for l in self.loss_component_history]
            total_losses = [l['components']['total'] for l in self.loss_component_history]
            ax.plot(epochs, total_losses, 'b-', label='Total Loss', linewidth=2)
            ax.plot(epochs, consistency_losses, 'r-', label='Consistency', linewidth=2)
            ax.set_yscale('log')
            ax.set_title('Loss Component Evolution')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Condition number over time
        ax = axes[0, 2]
        if self.matrix_health_history:
            condition_numbers = [m['precision_matrix']['condition_number'] for m in self.matrix_health_history]
            ax.plot(epochs, condition_numbers, 'g-', linewidth=2)
            ax.axhline(self.condition_number_threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
            ax.set_yscale('log')
            ax.set_title('Matrix Condition Number')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Condition Number')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Gradient components breakdown (latest epoch)
        ax = axes[1, 0]
        if self.gradient_history:
            latest_grads = self.gradient_history[-1]
            groups = list(latest_grads['parameter_groups'].keys())
            norms = [latest_grads['parameter_groups'][g]['norm'] for g in groups]
            ax.bar(groups, norms)
            ax.set_title('Latest Gradient Norms by Component')
            ax.set_ylabel('Gradient Norm')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 5: HVP quality over time
        ax = axes[1, 1]
        if self.loss_component_history:
            hvp_sims = []
            hvp_ratios = []
            for l in self.loss_component_history:
                if 'hvp_analysis' in l and l['hvp_analysis']:
                    hvp_sims.append(l['hvp_analysis']['cosine_similarity'])
                    hvp_ratios.append(l['hvp_analysis']['magnitude_ratio'])
            
            if hvp_sims:
                ax2 = ax.twinx()
                ax.plot(range(len(hvp_sims)), hvp_sims, 'b-', label='Cosine Sim', linewidth=2)
                ax2.plot(range(len(hvp_ratios)), hvp_ratios, 'r-', label='Magnitude Ratio', linewidth=2)
                ax.set_ylabel('Cosine Similarity', color='b')
                ax2.set_ylabel('Magnitude Ratio', color='r')
                ax.set_title('HVP Quality Metrics')
                ax.set_xlabel('Epoch')
                ax.grid(True, alpha=0.3)
        
        # Plot 6: Loss component ratios
        ax = axes[1, 2]
        if self.loss_component_history:
            consistency_ratios = []
            l2_ratios = []
            for l in self.loss_component_history:
                total = l['components']['total']
                if total > 0:
                    consistency_ratios.append(l['components']['consistency'] / total)
                    l2_ratios.append(l['components']['l2_reg_r'] / total)
            
            if consistency_ratios:
                ax.plot(range(len(consistency_ratios)), consistency_ratios, 'b-', label='Consistency', linewidth=2)
                ax.plot(range(len(l2_ratios)), l2_ratios, 'r-', label='L2 Reg', linewidth=2)
                ax.set_title('Loss Component Ratios')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Ratio of Total Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Plot 7: Multi-sigma HVP variance over time (NEW)
        if has_multi_sigma:
            ax = axes[2, 0]
            hvp_variances = [m['hvp_variance'] for m in self.multi_sigma_history]
            ax.plot(range(len(hvp_variances)), hvp_variances, 'purple', linewidth=2)
            ax.set_title('HVP Variance Across Sigmas')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Variance')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        # Plot 8: Curvature floor contribution over time (NEW)
        if has_multi_sigma:
            ax = axes[2, 1]
            floor_contributions = [m['curvature_floor_contribution'] for m in self.multi_sigma_history]
            ax.plot(range(len(floor_contributions)), floor_contributions, 'brown', linewidth=2)
            ax.axhline(0.1, color='gray', linestyle='--', alpha=0.5, label='Target ~10%')
            ax.set_title('Curvature Floor Contribution')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Ratio (δ·u / ||h||)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 9: Per-sigma HVP norms (boxplot) (NEW)
        if has_multi_sigma:
            ax = axes[2, 2]
            # Collect all HVP norms per sigma across epochs
            num_sigmas = self.multi_sigma_history[0]['num_sigmas']
            sigma_data = [[] for _ in range(num_sigmas)]

            for m in self.multi_sigma_history[-20:]:  # Last 20 epochs
                for sigma_idx, norm in enumerate(m['hvp_norms']):
                    sigma_data[sigma_idx].append(norm)

            if sigma_data[0]:  # Check if we have data
                positions = range(1, num_sigmas + 1)
                ax.boxplot(sigma_data, positions=positions, widths=0.6)
                time_labels = [f"{t:.3f}" for t in self.multi_sigma_history[0]['time_values']]
                ax.set_xticks(positions)
                ax.set_xticklabels(time_labels)
                ax.set_title('HVP Norms per Time Value (Last 20 Epochs)')
                ax.set_xlabel('Time Value (t)')
                ax.set_ylabel('HVP Norm')
                ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
