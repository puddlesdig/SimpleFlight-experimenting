#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training script for SimpleFlight trajectory tracking task using RSL-RL.

This script trains a PPO policy for the SimpleFlight quadcopter trajectory tracking task.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SimpleFlight trajectory tracking with RSL-RL PPO")
parser.add_argument("--video", action="store_true", default=False, help="Record video during evaluation")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video in steps")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval for video recording")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
parser.add_argument("--task", type=str, default=None, help="Name of the task")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training iterations")
parser.add_argument("--enable_dr", action="store_true", help="Enable domain randomization")
parser.add_argument("--trajectory", type=str, default="hover", choices=["hover", "lemniscate", "polynomial"], help="Trajectory type")
parser.add_argument("--log_dir", type=str, default=None, help="Log directory path")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.direct.simpleflight import agents


def main():
    """Train SimpleFlight task with RSL-RL PPO."""
    
    env_cfg = parse_env_cfg(
        "Isaac-SimpleFlight-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    
    if args_cli.enable_dr:
        env_cfg.enable_domain_randomization = True
    
    env_cfg.trajectory_type = args_cli.trajectory

    # Import agent configuration
    agent_cfg = agents.rsl_rl_ppo_cfg.SimpleFlightPPORunnerCfg()
    
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # Set log directory
    if args_cli.log_dir is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    else:
        log_root_path = args_cli.log_dir
    
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Create environment
    env = gym.make("Isaac-SimpleFlight-Direct-v0", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap for video recording if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
