"""
Minari Trajectory Dataset

Copy from notebook:
- Cell 8: space_dim() utility function
- Cell 12: MinariTrajectoryDataset class

This module provides a PyTorch IterableDataset that yields full trajectory
sequences from Minari datasets for training the precision head.

Key features:
- Sliding window extraction from episodes
- Padding for shorter episodes
- Mixing dataset goals with trajectory endpoints
- Outputs trajectory with positions and actions

Dependencies:
- torch
- torch.utils.data.IterableDataset
- minari
- gymnasium.spaces.utils.flatten

Usage:
    from diffuser_maze.data.dataset import MinariTrajectoryDataset

    dataset = MinariTrajectoryDataset(
        dataset_id="pointmaze/my-rrt-umaze-large-v0",
        horizon=256,
        stride=1
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=4
    )
"""
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Dict, Tuple
from gymnasium.spaces.utils import flatten, flatdim, unflatten
import numpy as np
import torch
from torch.utils.data import IterableDataset
import minari

def space_dim(space):
    """
    Get dimension of a gymnasium space.

    Args:
        space: gymnasium.Space object

    Returns:
        int: Dimension of the space
    """
    
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, MultiBinary):
        return int(np.prod(space.shape))
    if isinstance(space, MultiDiscrete):
        return int(np.sum(space.nvec))
    if isinstance(space, Dict):
        return sum(space_dim(s) for s in space.spaces.values())
    if isinstance(space, Tuple):
        return sum(space_dim(s) for s in space.spaces)
    return int(flatdim(space))

class MinariTrajectoryDataset(IterableDataset):
    """
    Dataset that yields full episode trajectories for diffusion training.
    Each trajectory contains both states and actions: [x, y, dx, dy]
    """
    def __init__(self, dataset_id: str, horizon: int, stride: int = 1):
        super().__init__()
        self.dataset_id = dataset_id
        self.horizon = horizon
        self.stride = stride  # For sliding window over long episodes
        
    def __iter__(self):
        ds = minari.load_dataset(self.dataset_id)
        env = ds.recover_environment()
        obs_space = env.observation_space
        act_space = env.action_space
        
        for ep in ds.iterate_episodes():
            # Get episode length
            ep_len = len(ep.actions)
            
            if ep_len < 10:  # Skip very short episodes
                continue
                
            # Extract observations and actions
            obs_dict = ep.observations
            actions = ep.actions
            
            # Get positions (achieved_goal contains [x, y] position)
            if isinstance(obs_dict, dict) and "achieved_goal" in obs_dict:
                positions = obs_dict["achieved_goal"][:, :2]  # [T, 2] for x, y
                goal = obs_dict["desired_goal"][0, :2]  # Goal position
            else:
                # Fallback: use observation field
                positions = obs_dict["observation"][:, :2] if "observation" in obs_dict else obs_dict[:, :2]
                goal = positions[-1]  # Use last position as goal
            
            # Create trajectory windows with sliding window
            # TODO: avoid stride for diffuser training
            for start_idx in range(0, ep_len - 10, self.stride):
                end_idx = min(start_idx + self.horizon, ep_len)
                window_len = end_idx - start_idx
                
                # Extract window
                window_pos = positions[start_idx:end_idx]
                window_act = actions[start_idx:end_idx]
                
                # Create trajectory [x, y, dx, dy]
                trajectory = np.zeros((self.horizon, 4), dtype=np.float32)
                
                # Fill positions and actions
                trajectory[:window_len, :2] = window_pos  # x, y
                trajectory[:window_len, 2:4] = window_act  # dx, dy
                
                # For padding, repeat last position with zero velocity
                if window_len < self.horizon:
                    trajectory[window_len:, :2] = window_pos[-1]
                    trajectory[window_len:, 2:4] = 0
                
                # Get start and goal for this window
                start_pos = window_pos[0]
                goal_pos = goal if np.random.rand() > 0.5 else window_pos[-1]  # Mix dataset goal and window end
                
                yield {
                    "trajectory": torch.from_numpy(trajectory),
                    "start": torch.from_numpy(start_pos),
                    "goal": torch.from_numpy(goal_pos),
                    "length": window_len
                }