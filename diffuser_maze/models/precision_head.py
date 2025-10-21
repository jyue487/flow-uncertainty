"""
Precision Head Model

Copy from notebook Cell 21 - The complete PrecisionHead class.

This class implements a neural network that predicts trajectory uncertainty
as a structured precision matrix: Λ = L L^T + R R^T

Key components to copy:
1. Class definition: class PrecisionHead(nn.Module)
2. __init__ method with all parameters
3. _construct_l_blocks() method
4. _assemble_block_bidiagonal_matrix() method
5. forward() method
6. apply_to_probe() method

Dependencies:
- torch
- torch.nn
- torch.nn.functional as F

Usage:
    from diffuser_maze.models.precision_head import PrecisionHead

    precision_head = PrecisionHead(
        traj_dim=512,
        state_dim=2,
        horizon=256,
        hidden_dim=256,
        low_rank_dim=16,
        use_low_rank=True,
        eps=1e-6
    )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrecisionHead(nn.Module):
    """
    Precision Head for trajectory uncertainty quantification.

    Takes a flattened trajectory y ∈ R^{2T} (positions only) and outputs
    structured precision matrix components (L, R) such that:
    Λφ(y) = L L^T + R R^T

    where L has block-bidiagonal structure (lower triangular) and R is low-rank (optional).
    """

    def __init__(
        self,
        traj_dim: int,          # 2T for state-only trajectories (2 * HORIZON)
        state_dim: int = 2,     # State dimension (x, y)
        horizon: int = 256,     # Planning horizon
        hidden_dim: int = 256,  # Hidden layer dimension
        low_rank_dim: int = 16, # Rank of R matrix
        use_low_rank: bool = True  # Whether to use low-rank R component
    ):
        super().__init__()

        self.traj_dim = traj_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.low_rank_dim = low_rank_dim
        self.use_low_rank = use_low_rank

        # Register buffer for R component control
        self.register_buffer('enable_R', torch.tensor(use_low_rank))

        # Numerical stability constant
        self.eps = 1e-6

        print(f"Precision Head Configuration:")
        print(f"  Trajectory dim: {traj_dim} (state dimensions only)")
        print(f"  Horizon: {horizon}, State dim: {state_dim}")
        print(f"  Block structure: {horizon} blocks of 2×2 matrices")
        print(f"  Low rank: {low_rank_dim if use_low_rank else 'disabled'}")

        # Trajectory encoder
        self.encoder = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Block-bidiagonal L matrix components (lower triangular)
        # Diagonal blocks: T blocks of 2×2 lower triangular (3 params each)
        self.l_diagonal_head = nn.Linear(hidden_dim, horizon * 3)
        # Sub-diagonal blocks: (T-1) blocks of 2×2 full matrices (4 params each)
        self.l_subdiag_head = nn.Linear(hidden_dim, (horizon - 1) * 4)

        # Low-rank R matrix: R ∈ R^{2T × low_rank_dim} (optional)
        if self.use_low_rank:
            self.r_head = nn.Linear(hidden_dim, traj_dim * low_rank_dim)

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for numerical stability."""
        for module in [self.l_diagonal_head, self.l_subdiag_head]:
            nn.init.normal_(module.weight, 0, 0.001)
            nn.init.zeros_(module.bias)

        if self.use_low_rank:
            nn.init.normal_(self.r_head.weight, 0, 0.01)
            nn.init.zeros_(self.r_head.bias)

    def _construct_l_blocks(self, l_diag_params, l_subdiag_params, batch_size):
        """
        Convert raw parameters to proper block matrices with numerical stability.

        Args:
            l_diag_params: [B, T*3] parameters for diagonal blocks
            l_subdiag_params: [B, (T-1)*4] parameters for sub-diagonal blocks
            batch_size: batch size

        Returns:
            l_diag_blocks: [B, T, 2, 2] diagonal blocks (lower triangular)
            l_subdiag_blocks: [B, T-1, 2, 2] sub-diagonal blocks (full)
        """
        T = self.horizon

        # Diagonal blocks: 2×2 lower triangular with positive diagonal
        l_diag_blocks = []
        for t in range(T):
            params = l_diag_params[:, t*3:(t+1)*3]  # [B, 3]

            # Create 2×2 lower triangular matrix for STATE dimensions
            # [[softplus(p0)+eps,         0        ],
            #  [      p1,         softplus(p2)+eps ]]
            block = torch.zeros(batch_size, 2, 2, device=params.device)
            block[:, 0, 0] = F.softplus(params[:, 0]) + self.eps  # x uncertainty
            block[:, 1, 0] = params[:, 1]                          # x-y coupling
            block[:, 1, 1] = F.softplus(params[:, 2]) + self.eps  # y uncertainty
            l_diag_blocks.append(block)

        # Sub-diagonal blocks: 2×2 unconstrained (temporal coupling)
        l_subdiag_blocks = []
        for t in range(T-1):
            params = l_subdiag_params[:, t*4:(t+1)*4]  # [B, 4]
            block = params.view(batch_size, 2, 2)  # Full 2×2 matrix
            l_subdiag_blocks.append(block)

        return torch.stack(l_diag_blocks, dim=1), torch.stack(l_subdiag_blocks, dim=1)

    def _construct_precision_matrix(self, l_diag_blocks, l_subdiag_blocks, r_matrix, batch_size):
        """
        Construct the full precision matrix from components.

        Args:
            l_diag_blocks: [B, T, 2, 2] diagonal blocks
            l_subdiag_blocks: [B, T-1, 2, 2] sub-diagonal blocks
            r_matrix: [B, 2T, rank] low-rank component (optional)
            batch_size: batch size

        Returns:
            precision: [B, 2T, 2T] precision matrix
            L: [B, 2T, 2T] full L matrix (for API compatibility)
        """
        T = self.horizon

        # Initialize L matrix (block-bidiagonal, lower triangular)
        L = torch.zeros(batch_size, self.traj_dim, self.traj_dim, device=l_diag_blocks.device)

        # Fill diagonal blocks
        for t in range(T):
            start_idx = t * 2
            end_idx = start_idx + 2
            L[:, start_idx:end_idx, start_idx:end_idx] = l_diag_blocks[:, t]

        # Fill sub-diagonal blocks (lower triangular structure)
        for t in range(T-1):
            row_start = (t + 1) * 2
            row_end = row_start + 2
            col_start = t * 2
            col_end = col_start + 2
            L[:, row_start:row_end, col_start:col_end] = l_subdiag_blocks[:, t]

        # Compute L L^T
        precision = torch.matmul(L, L.transpose(-2, -1))

        # Optionally add R R^T
        if self.enable_R and r_matrix is not None:
            # r_matrix: [B, 2T, rank]
            precision = precision + torch.matmul(r_matrix, r_matrix.transpose(-2, -1))

        return precision, L

    def forward(self, traj_positions):
        """
        Forward pass of precision head.

        Args:
            traj_positions: [B, 2T] flattened trajectory positions (states only)

        Returns:
            precision_matrix: [B, 2T, 2T] structured precision matrix
            L_matrix: [B, 2T, 2T] full L matrix (for API compatibility)
            components: Dict with L and R matrix components
        """
        batch_size = traj_positions.shape[0]

        # Encode trajectory
        encoded = self.encoder(traj_positions)  # [B, hidden_dim]

        # Generate L matrix parameters
        l_diag_params = self.l_diagonal_head(encoded)  # [B, T*3]
        l_subdiag_params = self.l_subdiag_head(encoded)  # [B, (T-1)*4]

        # Construct L blocks
        l_diag_blocks, l_subdiag_blocks = self._construct_l_blocks(
            l_diag_params, l_subdiag_params, batch_size
        )

        # Generate R matrix if enabled
        r_matrix = None
        if self.use_low_rank:
            r_params = self.r_head(encoded)  # [B, 2T*rank]
            r_matrix = r_params.view(batch_size, self.traj_dim, self.low_rank_dim)

        # Construct precision matrix: Λ = L L^T + R R^T
        precision_matrix, L_matrix = self._construct_precision_matrix(
            l_diag_blocks, l_subdiag_blocks, r_matrix, batch_size
        )

        # Prepare components dictionary
        components = {
            'l_diagonal': l_diag_blocks,
            'l_subdiagonal': l_subdiag_blocks,
        }

        if self.use_low_rank:
            components['r_matrix'] = r_matrix

        return precision_matrix, L_matrix, components

    def apply_to_probe(self, traj_positions, probe_vector):
        """
        Apply precision matrix to probe vector: Λ(y) * u

        Args:
            traj_positions: [B, 2T] trajectory positions (states only)
            probe_vector: [B, 2T] probe vector

        Returns:
            result: [B, 2T] result of Λ(y) * u
        """
        precision_matrix, _, _ = self.forward(traj_positions)
        result = torch.matmul(precision_matrix, probe_vector.unsqueeze(-1)).squeeze(-1)
        return result

