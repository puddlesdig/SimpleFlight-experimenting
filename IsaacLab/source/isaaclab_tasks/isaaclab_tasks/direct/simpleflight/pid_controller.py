# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PID rate controller for CTBR command tracking."""

from __future__ import annotations

import torch
from torch import Tensor

from isaaclab.utils.math import quat_rotate_inverse

from .constants import (
    PID_RATE_I_LIMIT,
    PID_RATE_KD,
    PID_RATE_KI,
    PID_RATE_KP,
    PID_RATE_OUT_LIMIT,
    PWM_RESOLUTION,
)


class PIDRateController:
    """PID controller for body rate tracking.
    
    Converts CTBR (Collective Thrust Body Rate) commands to individual motor thrust commands.
    Matches the Crazyflie firmware implementation.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        dt: float,
        kp: list[float] = PID_RATE_KP,
        ki: list[float] = PID_RATE_KI,
        kd: list[float] = PID_RATE_KD,
        i_limit: list[float] = PID_RATE_I_LIMIT,
        out_limit: float = PID_RATE_OUT_LIMIT,
    ):
        """Initialize the PID rate controller.
        
        Args:
            num_envs: Number of parallel environments.
            device: Device for tensor computations.
            dt: Control timestep in seconds.
            kp: Proportional gains for [roll, pitch, yaw].
            ki: Integral gains for [roll, pitch, yaw].
            kd: Derivative gains for [roll, pitch, yaw].
            i_limit: Integral term limits for [roll, pitch, yaw].
            out_limit: Output saturation limit.
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt

        self.kp = torch.tensor(kp, device=device, dtype=torch.float32)
        self.ki = torch.tensor(ki, device=device, dtype=torch.float32)
        self.kd = torch.tensor(kd, device=device, dtype=torch.float32)
        self.i_limit = torch.tensor(i_limit, device=device, dtype=torch.float32)
        self.out_limit = out_limit

        self.last_body_rate = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        self.integral = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)

    def reset(self, env_ids: Tensor) -> None:
        """Reset controller state for specified environments.
        
        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None or len(env_ids) == 0:
            return
        self.last_body_rate[env_ids] = 0.0
        self.integral[env_ids] = 0.0

    def compute_control(
        self,
        target_rate_deg_s: Tensor,
        target_thrust_pwm: Tensor,
        root_quat: Tensor,
        angular_velocity: Tensor,
    ) -> Tensor:
        """Compute motor commands from CTBR setpoints.
        
        Args:
            target_rate_deg_s: Target body rates in deg/s [num_envs, 3] (roll, pitch, yaw).
            target_thrust_pwm: Target collective thrust in PWM units [num_envs, 1].
            root_quat: Root orientation quaternion [num_envs, 4].
            angular_velocity: Angular velocity in world frame [num_envs, 3].
        
        Returns:
            Motor commands in normalized range [-1, 1] [num_envs, 4].
        """
        body_rate_rad_s = quat_rotate_inverse(root_quat, angular_velocity)
        body_rate_deg_s = body_rate_rad_s * 180.0 / torch.pi

        rate_error = target_rate_deg_s - body_rate_deg_s

        output_p = rate_error * self.kp.unsqueeze(0)

        derivative = -(body_rate_deg_s - self.last_body_rate) / self.dt
        derivative = torch.nan_to_num(derivative, nan=0.0, posinf=0.0, neginf=0.0)
        output_d = derivative * self.kd.unsqueeze(0)

        self.integral += rate_error * self.dt
        self.integral = torch.clamp(self.integral, -self.i_limit.unsqueeze(0), self.i_limit.unsqueeze(0))
        output_i = self.integral * self.ki.unsqueeze(0)

        output = output_p + output_d + output_i
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        output = torch.clamp(output, -self.out_limit, self.out_limit)

        self.last_body_rate = body_rate_deg_s.clone()

        r = (output[:, 0] / 2.0).unsqueeze(1)
        p = (output[:, 1] / 2.0).unsqueeze(1)
        y = (output[:, 2]).unsqueeze(1)

        m1 = target_thrust_pwm + r - p + y
        m2 = target_thrust_pwm + r + p - y
        m3 = target_thrust_pwm - r + p + y
        m4 = target_thrust_pwm - r - p - y

        motor_cmds = torch.cat([m1, m2, m3, m4], dim=1)

        normalized_cmds = motor_cmds / PWM_RESOLUTION * 2.0 - 1.0
        normalized_cmds = torch.clamp(normalized_cmds, -1.0, 1.0)

        return normalized_cmds
