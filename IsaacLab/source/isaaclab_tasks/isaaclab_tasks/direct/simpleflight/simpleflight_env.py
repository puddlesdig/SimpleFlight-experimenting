# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SimpleFlight trajectory tracking environment."""

from __future__ import annotations

import gymnasium as gym
import torch
from collections import deque
from torch import Tensor

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from .constants import (
    ANGULAR_VELOCITY_NOISE_STD_RAD_S,
    CRAZYFLIE_MASS,
    ENABLE_LATENCY_SIMULATION,
    ENABLE_RANDOM_LATENCY,
    ENABLE_WIND,
    DR_DRAG_COEF_SCALE_RANGE,
    DR_INERTIA_SCALE_RANGE,
    DR_MASS_SCALE_RANGE,
    DR_TAU_SCALE_RANGE,
    GRAVITY_MAGNITUDE,
    LATENCY_STEPS,
    LPF_COEFFICIENT,
    MAX_TRAJECTORY_STEPS,
    MAX_THRUST_RATIO,
    MIN_THRUST_RATIO,
    ORIENTATION_NOISE_STD_RAD,
    POSITION_NOISE_STD_M,
    PWM_RESOLUTION,
    RESET_DISTANCE_THRESHOLD_M,
    RESET_HEIGHT_THRESHOLD_M,
    REWARD_ACTION_NORM_WEIGHT_INIT,
    REWARD_ACTION_NORM_WEIGHT_LR,
    REWARD_ACTION_SMOOTHNESS_WEIGHT_INIT,
    REWARD_ACTION_SMOOTHNESS_WEIGHT_LR,
    REWARD_DISTANCE_SCALE,
    REWARD_NORM_MAX,
    REWARD_SMOOTHNESS_MAX,
    REWARD_SPIN_WEIGHT,
    REWARD_UP_WEIGHT,
    TARGET_RATE_CLIP_DEG_S,
    TRAJECTORY_STEP_SIZE,
    VELOCITY_NOISE_STD_M_S,
    WIND_INTENSITY_RANGE,
)
from .pid_controller import PIDRateController
from .simpleflight_env_cfg import SimpleFlightEnvCfg
from .trajectories import ChainedPolynomialTrajectory, HoverTrajectory, LemniscateTrajectory


class SimpleFlightEnv(DirectRLEnv):
    """SimpleFlight environment for trajectory tracking with Crazyflie quadcopter.
    
    This environment replicates the SimpleFlight task from the original repository,
    adapted to IsaacLab framework with estimator-like observations.
    """

    cfg: SimpleFlightEnvCfg

    def __init__(self, cfg: SimpleFlightEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self._ctbr_commands = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        
        self.pid_controller = PIDRateController(
            num_envs=self.num_envs,
            device=self.device,
            dt=self.step_dt,
        )

        if cfg.trajectory_type == "hover":
            self.trajectory = HoverTrajectory(self.num_envs, self.device)
        elif cfg.trajectory_type == "lemniscate":
            self.trajectory = LemniscateTrajectory(self.num_envs, self.device)
        else:
            self.trajectory = ChainedPolynomialTrajectory(self.num_envs, self.device)

        if cfg.enable_latency and ENABLE_LATENCY_SIMULATION:
            self.obs_buffer = deque(maxlen=LATENCY_STEPS + 1)
        else:
            self.obs_buffer = None

        self.reward_action_smoothness_weight = REWARD_ACTION_SMOOTHNESS_WEIGHT_INIT
        self.reward_action_norm_weight = REWARD_ACTION_NORM_WEIGHT_INIT
        self.update_count = 0

        if cfg.enable_wind and ENABLE_WIND:
            self.wind_intensity = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
            self.wind_freq_components = torch.zeros(self.num_envs, 3, 8, device=self.device, dtype=torch.float32)

        self._robot_mass = CRAZYFLIE_MASS
        self._gravity_magnitude = GRAVITY_MAGNITUDE

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "tracking_error",
                "reward_pos",
                "reward_up",
                "reward_spin",
                "reward_action_smoothness",
                "reward_action_norm",
                "action_error",
            ]
        }

    def _setup_scene(self):
        """Setup scene with robot and terrain."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: Tensor):
        """Process actions before physics step.
        
        Args:
            actions: Policy output [num_envs, 4] - CTBR commands before scaling.
        """
        actions = torch.clamp(actions, -1.0, 1.0)
        self._actions = actions.clone()

        actions_tanh = torch.tanh(actions)
        
        target_rate = actions_tanh[:, :3]
        target_thrust = actions_tanh[:, 3:4]

        ctbr_action = torch.cat([target_rate, target_thrust], dim=-1)
        
        if LPF_COEFFICIENT < 1.0:
            ctbr_action = LPF_COEFFICIENT * ctbr_action + (1.0 - LPF_COEFFICIENT) * self._prev_actions

        action_error = torch.norm(ctbr_action - self._prev_actions, dim=-1)
        self._episode_sums["action_error"] += action_error

        self._prev_actions = ctbr_action.clone()

        target_thrust_clamped = torch.clamp((ctbr_action[:, 3:4] + 1.0) / 2.0, min=MIN_THRUST_RATIO, max=MAX_THRUST_RATIO)
        
        target_rate_scaled = ctbr_action[:, :3] * TARGET_RATE_CLIP_DEG_S
        target_thrust_pwm = target_thrust_clamped * PWM_RESOLUTION

        self._ctbr_commands[:, :3] = target_rate_scaled
        self._ctbr_commands[:, 3:4] = target_thrust_pwm

        motor_cmds = self.pid_controller.compute_control(
            target_rate_deg_s=target_rate_scaled,
            target_thrust_pwm=target_thrust_pwm,
            root_quat=self._robot.data.root_quat_w,
            angular_velocity=self._robot.data.root_ang_vel_w,
        )

        motor_thrusts = (motor_cmds + 1.0) / 2.0
        motor_forces = motor_thrusts * self._robot_mass * self._gravity_magnitude * 4.0 / MAX_THRUST_RATIO

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)
        self._thrust[:, 0, 2] = motor_forces.sum(dim=-1)

        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)

    def _apply_action(self):
        """Apply computed forces to robot."""
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=0)

        if self.cfg.enable_wind and ENABLE_WIND:
            time_s = self.episode_length_buf * self.step_dt
            wind_force_body = self.wind_intensity * torch.sin(time_s.view(-1, 1, 1) * self.wind_freq_components).sum(dim=-1)
            wind_force_world = wind_force_body * self._robot_mass * self._gravity_magnitude
            wind_forces_expanded = wind_force_world.unsqueeze(1)
            self._robot.set_external_force_and_torque(wind_forces_expanded, None, body_ids=0, is_additive=True)

    def _get_observations(self) -> dict:
        """Compute observations with noise and latency simulation.
        
        Returns:
            Dictionary with 'policy' key containing observations [num_envs, 42].
        """
        current_time = self.episode_length_buf * self.step_dt
        future_times = current_time.unsqueeze(1) + torch.arange(
            MAX_TRAJECTORY_STEPS, device=self.device, dtype=torch.float32
        ).unsqueeze(0) * TRAJECTORY_STEP_SIZE * self.step_dt
        
        target_positions = self.trajectory.get_position(future_times)

        root_pos = self._robot.data.root_pos_w.clone()
        root_quat = self._robot.data.root_quat_w.clone()
        linear_vel = self._robot.data.root_lin_vel_w.clone()
        angular_vel = self._robot.data.root_ang_vel_w.clone()

        if self.cfg.enable_observation_noise:
            root_pos += torch.randn_like(root_pos) * POSITION_NOISE_STD_M
            linear_vel += torch.randn_like(linear_vel) * VELOCITY_NOISE_STD_M_S
            angular_vel += torch.randn_like(angular_vel) * ANGULAR_VELOCITY_NOISE_STD_RAD_S
            
            quat_noise = torch.randn(self.num_envs, 3, device=self.device, dtype=torch.float32) * ORIENTATION_NOISE_STD_RAD
            angle = torch.norm(quat_noise, dim=-1, keepdim=True)
            axis = quat_noise / (angle + 1e-8)
            half_angle = angle / 2.0
            noise_quat = torch.cat([
                torch.cos(half_angle),
                axis * torch.sin(half_angle)
            ], dim=-1)
            
            root_quat_w_first = root_quat[:, 0:1]
            root_quat_xyz = root_quat[:, 1:]
            root_quat_combined_w = root_quat_w_first * noise_quat[:, 0:1] - torch.sum(root_quat_xyz * noise_quat[:, 1:], dim=-1, keepdim=True)
            root_quat_combined_xyz = root_quat_w_first * noise_quat[:, 1:] + noise_quat[:, 0:1] * root_quat_xyz + torch.cross(root_quat_xyz, noise_quat[:, 1:], dim=-1)
            root_quat = torch.cat([root_quat_combined_w, root_quat_combined_xyz], dim=-1)
            root_quat = root_quat / torch.norm(root_quat, dim=-1, keepdim=True)

        rel_pos = target_positions - root_pos.unsqueeze(1)
        rel_pos_flat = rel_pos.reshape(self.num_envs, -1)

        # Extract rotation vectors by rotating unit vectors with quaternion
        x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        y_axis = torch.tensor([[0.0, 1.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)
        
        heading = quat_apply(root_quat, x_axis)
        lateral = quat_apply(root_quat, y_axis)
        up = quat_apply(root_quat, z_axis)

        obs = torch.cat([
            rel_pos_flat,
            linear_vel,
            heading,
            lateral,
            up,
        ], dim=-1)

        if self.obs_buffer is not None:
            self.obs_buffer.append(obs.clone())
            
            if len(self.obs_buffer) < LATENCY_STEPS + 1:
                latent_obs = obs
            else:
                if ENABLE_RANDOM_LATENCY:
                    indices = torch.randint(0, len(self.obs_buffer), (self.num_envs,), device=self.device)
                    obs_list = list(self.obs_buffer)
                    latent_obs = torch.stack([obs_list[idx][i] for i, idx in enumerate(indices)])
                else:
                    latent_obs = self.obs_buffer[0]
            
            observations = {"policy": latent_obs}
        else:
            observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> Tensor:
        """Compute rewards based on tracking error and regularization terms.
        
        Returns:
            Reward tensor [num_envs].
        """
        current_time = self.episode_length_buf * self.step_dt
        current_target = self.trajectory.get_position(current_time.unsqueeze(1))[:, 0, :]
        
        distance = torch.norm(self._robot.data.root_pos_w - current_target, dim=-1)
        reward_pos = torch.exp(-REWARD_DISTANCE_SCALE * distance)

        # Up-axis alignment (Z component of rotated up vector)
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)
        up_axis = quat_apply(self._robot.data.root_quat_w, z_axis)
        up_alignment = up_axis[:, 2]
        reward_up = 0.5 * (1.0 + up_alignment)

        spin_vel_z = self._robot.data.root_lin_vel_b[:, 2]
        spin = torch.square(spin_vel_z)
        reward_spin = 0.5 / (1.0 + torch.square(spin))

        action_smoothness = self._episode_sums["action_error"] / (self.episode_length_buf.float() + 1.0)
        reward_action_smoothness = -self.reward_action_smoothness_weight * action_smoothness

        action_norm = torch.norm(self._actions, dim=-1)
        reward_action_norm = -self.reward_action_norm_weight * action_norm

        reward = (
            reward_pos
            + reward_pos * (REWARD_UP_WEIGHT * reward_up + REWARD_SPIN_WEIGHT * reward_spin)
            + reward_action_smoothness
            + reward_action_norm
        )

        self._episode_sums["tracking_error"] += distance
        self._episode_sums["reward_pos"] += reward_pos
        self._episode_sums["reward_up"] += reward_pos * reward_up
        self._episode_sums["reward_spin"] += reward_pos * reward_spin
        self._episode_sums["reward_action_smoothness"] += reward_action_smoothness
        self._episode_sums["reward_action_norm"] += reward_action_norm

        return reward

    def _get_dones(self) -> tuple[Tensor, Tensor]:
        """Determine episode termination conditions.
        
        Returns:
            Tuple of (died, time_out) boolean tensors.
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        current_time = self.episode_length_buf * self.step_dt
        current_target = self.trajectory.get_position(current_time.unsqueeze(1))[:, 0, :]
        distance = torch.norm(self._robot.data.root_pos_w - current_target, dim=-1)
        
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < RESET_HEIGHT_THRESHOLD_M,
            distance > RESET_DISTANCE_THRESHOLD_M
        )
        
        return died, time_out

    def _reset_idx(self, env_ids: Tensor | None):
        """Reset specified environments.
        
        Args:
            env_ids: Environment indices to reset. None means all.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        
        self.pid_controller.reset(env_ids)

        if hasattr(self.trajectory, 'reset'):
            self.trajectory.reset(env_ids)

        if self.obs_buffer is not None:
            self.obs_buffer.clear()

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if self.cfg.enable_domain_randomization:
            self._apply_domain_randomization(env_ids)

        if self.cfg.enable_wind and ENABLE_WIND:
            self.wind_intensity[env_ids] = torch.rand(
                len(env_ids), 1, device=self.device, dtype=torch.float32
            ) * (WIND_INTENSITY_RANGE[1] - WIND_INTENSITY_RANGE[0]) + WIND_INTENSITY_RANGE[0]
            
            self.wind_freq_components[env_ids] = torch.randn(
                len(env_ids), 3, 8, device=self.device, dtype=torch.float32
            )

        self.update_count += 1
        self.reward_action_smoothness_weight = min(
            REWARD_ACTION_SMOOTHNESS_WEIGHT_INIT + REWARD_ACTION_SMOOTHNESS_WEIGHT_LR * self.update_count,
            REWARD_SMOOTHNESS_MAX
        )
        self.reward_action_norm_weight = min(
            REWARD_ACTION_NORM_WEIGHT_INIT + REWARD_ACTION_NORM_WEIGHT_LR * self.update_count,
            REWARD_NORM_MAX
        )

    def _apply_domain_randomization(self, env_ids: Tensor):
        """Apply domain randomization to specified environments.
        
        Args:
            env_ids: Environment indices to randomize.
        """
        if len(env_ids) == 0:
            return

        masses = self._robot_mass * torch.empty(len(env_ids), device=self.device).uniform_(
            DR_MASS_SCALE_RANGE[0], DR_MASS_SCALE_RANGE[1]
        )
        
        self._robot.root_physx_view.set_masses(masses, env_ids)
