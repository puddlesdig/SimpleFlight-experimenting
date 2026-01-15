# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for SimpleFlight trajectory tracking environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG

from .constants import (
    CONTROL_FREQUENCY_HZ,
    ENV_SPACING_M,
    EPISODE_LENGTH_S,
    SIMULATION_DT,
    SIM_SUBSTEPS,
)


@configclass
class SimpleFlightEnvCfg(DirectRLEnvCfg):
    """Configuration for SimpleFlight trajectory tracking environment."""

    episode_length_s = EPISODE_LENGTH_S
    decimation = SIM_SUBSTEPS
    action_space = 4
    observation_space = 42
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=SIMULATION_DT,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=ENV_SPACING_M,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    trajectory_type: str = "hover"
    enable_domain_randomization: bool = False
    enable_wind: bool = False
    enable_observation_noise: bool = True
    enable_latency: bool = True
