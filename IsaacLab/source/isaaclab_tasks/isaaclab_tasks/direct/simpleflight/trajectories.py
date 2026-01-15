# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory generators for SimpleFlight task."""

from __future__ import annotations

import torch
from torch import Tensor

from .constants import EPISODE_LENGTH_S, SIMULATION_DT


class HoverTrajectory:
    """Stationary hover trajectory at a fixed position."""

    def __init__(self, num_envs: int, device: str, target_pos: Tensor = None):
        """Initialize hover trajectory.
        
        Args:
            num_envs: Number of parallel environments.
            device: Device for tensor computations.
            target_pos: Target hover position [3] or [num_envs, 3]. Defaults to [0, 0, 1].
        """
        self.num_envs = num_envs
        self.device = device

        if target_pos is None:
            self.target = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        else:
            self.target = target_pos.to(device)
        
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(0).expand(num_envs, 3)

    def get_position(self, t: Tensor) -> Tensor:
        """Compute trajectory position at time t (always returns target position).
        
        Args:
            t: Time values [num_envs, num_steps] in seconds.
        
        Returns:
            Positions [num_envs, num_steps, 3] - all identical to target.
        """
        num_steps = t.shape[1]
        # Return the same target position for all timesteps
        return self.target.unsqueeze(1).expand(-1, num_steps, -1)

    def get_velocity(self, t: Tensor) -> Tensor:
        """Compute trajectory velocity at time t (always zero for hover).
        
        Args:
            t: Time values [num_envs, num_steps] in seconds.
        
        Returns:
            Velocities [num_envs, num_steps, 3] - all zeros.
        """
        num_steps = t.shape[1]
        return torch.zeros(self.num_envs, num_steps, 3, device=self.device, dtype=torch.float32)


class LemniscateTrajectory:
    """Figure-8 (lemniscate) trajectory generator."""

    def __init__(self, num_envs: int, device: str, period_s: float = 5.5, origin: Tensor = None):
        """Initialize lemniscate trajectory.
        
        Args:
            num_envs: Number of parallel environments.
            device: Device for tensor computations.
            period_s: Period of one complete figure-8 in seconds.
            origin: Origin position [3] or [num_envs, 3]. Defaults to [0, 0, 1].
        """
        self.num_envs = num_envs
        self.device = device
        self.period = period_s

        if origin is None:
            self.origin = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        else:
            self.origin = origin.to(device)
        
        if self.origin.dim() == 1:
            self.origin = self.origin.unsqueeze(0).expand(num_envs, 3)

    def get_position(self, t: Tensor) -> Tensor:
        """Compute trajectory position at time t.
        
        Args:
            t: Time values [num_envs, num_steps] in seconds.
        
        Returns:
            Positions [num_envs, num_steps, 3].
        """
        omega = 2.0 * torch.pi / self.period
        theta = omega * t

        x = 0.7 * torch.sin(theta)
        y = 0.7 * torch.sin(theta) * torch.cos(theta)
        z = torch.zeros_like(x)

        positions = torch.stack([x, y, z], dim=-1)
        
        origin_expanded = self.origin.unsqueeze(1)
        positions = positions + origin_expanded

        return positions


class ChainedPolynomialTrajectory:
    """Random chained polynomial trajectory generator."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        scale: float = 2.5,
        min_dt: float = 1.5,
        max_dt: float = 4.0,
        origin: Tensor = None,
    ):
        """Initialize chained polynomial trajectory.
        
        Args:
            num_envs: Number of parallel environments.
            device: Device for tensor computations.
            scale: Spatial scale of trajectory.
            min_dt: Minimum segment duration.
            max_dt: Maximum segment duration.
            origin: Origin position [3] or [num_envs, 3].
        """
        self.num_envs = num_envs
        self.device = device
        self.scale = scale
        self.min_dt = min_dt
        self.max_dt = max_dt

        if origin is None:
            self.origin = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        else:
            self.origin = origin.to(device)
        
        if self.origin.dim() == 1:
            self.origin = self.origin.unsqueeze(0).expand(num_envs, 3)

        max_segments = int(EPISODE_LENGTH_S / min_dt) + 2
        self.waypoints = torch.zeros(num_envs, max_segments, 3, device=device, dtype=torch.float32)
        self.segment_times = torch.zeros(num_envs, max_segments, device=device, dtype=torch.float32)
        self.reset()

    def reset(self, env_ids: Tensor = None) -> None:
        """Generate new random trajectory for specified environments.
        
        Args:
            env_ids: Environment indices to reset. None means all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if len(env_ids) == 0:
            return

        num_segments = self.waypoints.shape[1]
        
        random_positions = torch.rand(len(env_ids), num_segments, 3, device=self.device, dtype=torch.float32)
        random_positions[:, :, :2] = random_positions[:, :, :2] * 2.0 - 1.0
        random_positions[:, :, :2] *= self.scale
        random_positions[:, :, 2] = random_positions[:, :, 2] * 0.5 + 0.75
        
        self.waypoints[env_ids] = random_positions + self.origin[env_ids].unsqueeze(1)
        
        durations = torch.rand(len(env_ids), num_segments, device=self.device, dtype=torch.float32)
        durations = durations * (self.max_dt - self.min_dt) + self.min_dt
        self.segment_times[env_ids] = torch.cumsum(durations, dim=1)

    def get_position(self, t: Tensor) -> Tensor:
        """Compute trajectory position at time t using piecewise linear interpolation.
        
        Args:
            t: Time values [num_envs, num_steps] in seconds.
        
        Returns:
            Positions [num_envs, num_steps, 3].
        """
        num_steps = t.shape[1]
        positions = torch.zeros(self.num_envs, num_steps, 3, device=self.device, dtype=torch.float32)

        for env_idx in range(self.num_envs):
            for step_idx in range(num_steps):
                time_val = t[env_idx, step_idx]
                
                segment_idx = torch.searchsorted(self.segment_times[env_idx], time_val, right=False)
                segment_idx = torch.clamp(segment_idx, 0, self.waypoints.shape[1] - 2)
                
                t0 = self.segment_times[env_idx, segment_idx] if segment_idx > 0 else 0.0
                t1 = self.segment_times[env_idx, segment_idx + 1]
                
                alpha = (time_val - t0) / (t1 - t0 + 1e-6)
                alpha = torch.clamp(alpha, 0.0, 1.0)
                
                p0 = self.waypoints[env_idx, segment_idx]
                p1 = self.waypoints[env_idx, segment_idx + 1]
                
                positions[env_idx, step_idx] = p0 * (1.0 - alpha) + p1 * alpha

        return positions
