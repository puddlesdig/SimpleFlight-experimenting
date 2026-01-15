# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SimpleFlight trajectory tracking environment for quadcopters."""

import gymnasium as gym

from . import agents
from .simpleflight_env_cfg import SimpleFlightEnvCfg
from .simpleflight_env import SimpleFlightEnv

##
# Register Gym environments
##

gym.register(
    id="Isaac-SimpleFlight-Direct-v0",
    entry_point="isaaclab_tasks.direct.simpleflight:SimpleFlightEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SimpleFlightEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleFlightPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
