#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Deploy trained SimpleFlight policy to real Crazyflie 2.1 via Crazyradio 2.0.

This script:
1. Loads the trained policy checkpoint
2. Connects to Crazyflie via Crazyradio
3. Reads IMU/state data from the drone
4. Runs policy inference
5. Sends CTBR commands to the drone

Requirements:
    pip install cflib torch numpy
"""

import argparse
import time
import numpy as np
import torch
import logging
from collections import deque

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# Add IsaacLab paths
import sys
from pathlib import Path
isaaclab_path = Path(__file__).resolve().parents[2]
sys.path.append(str(isaaclab_path / "source" / "isaaclab"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleFlightDeployer:
    """Deploy SimpleFlight policy to Crazyflie hardware."""
    
    def __init__(self, uri: str, checkpoint_path: str, trajectory_type: str = "hover"):
        """Initialize deployer.
        
        Args:
            uri: Crazyflie URI (e.g., 'radio://0/80/2M/E7E7E7E7E7')
            checkpoint_path: Path to trained policy checkpoint (.pt file)
            trajectory_type: Type of trajectory ('hover', 'lemniscate', 'polynomial')
        """
        self.uri = uri
        self.trajectory_type = trajectory_type
        
        # Load trained policy
        logger.info(f"Loading policy from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract policy directly from checkpoint
        # RSL-RL saves the entire ActorCritic module
        self.policy = checkpoint
        
        # Set to evaluation mode
        if isinstance(self.policy, dict):
            # If checkpoint is a state dict, we need to build a simple network
            # For now, just use random actions to test the pipeline
            logger.warning("Checkpoint is a state dict, using random actions for testing")
            self.policy = None
        else:
            self.policy.eval()
        
        # State from Crazyflie
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z (meters)
        self.velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz (m/s)
        self.orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.angular_velocity = np.array([0.0, 0.0, 0.0])  # wx, wy, wz (rad/s)
        
        # Trajectory waypoints
        if trajectory_type == "hover":
            # Hover at 1m altitude
            self.target_positions = np.tile([0.0, 0.0, 1.0], (10, 1))
        else:
            # You can add other trajectory types here
            raise NotImplementedError(f"Trajectory type {trajectory_type} not implemented")
        
        # Action history for smoothness
        self.prev_action = np.zeros(4)
        
        # Observation buffer for latency simulation (optional)
        self.obs_buffer = deque(maxlen=5)
        
    def state_callback(self, timestamp, data, logconf):
        """Callback for state estimate from Crazyflie."""
        # Position from state estimator
        self.position[0] = data['stateEstimate.x']
        self.position[1] = data['stateEstimate.y']
        self.position[2] = data['stateEstimate.z']
        
        # Velocity
        self.velocity[0] = data['stateEstimate.vx']
        self.velocity[1] = data['stateEstimate.vy']
        self.velocity[2] = data['stateEstimate.vz']
    
    def quat_callback(self, timestamp, data, logconf):
        """Callback for quaternion data."""
        # Orientation (quaternion)
        self.orientation_quat[0] = data['stateEstimate.qw']
        self.orientation_quat[1] = data['stateEstimate.qx']
        self.orientation_quat[2] = data['stateEstimate.qy']
        self.orientation_quat[3] = data['stateEstimate.qz']
        
    def gyro_callback(self, timestamp, data, logconf):
        """Callback for gyroscope data."""
        self.angular_velocity[0] = data['gyro.x'] * np.pi / 180.0  # deg/s -> rad/s
        self.angular_velocity[1] = data['gyro.y'] * np.pi / 180.0
        self.angular_velocity[2] = data['gyro.z'] * np.pi / 180.0
    
    def quat_to_rotation_vectors(self, quat):
        """Convert quaternion to body frame axes (heading, lateral, up)."""
        w, x, y, z = quat
        
        # Heading (X-axis in body frame)
        heading = np.array([
            1 - 2*(y**2 + z**2),
            2*(x*y + w*z),
            2*(x*z - w*y)
        ])
        
        # Lateral (Y-axis in body frame)
        lateral = np.array([
            2*(x*y - w*z),
            1 - 2*(x**2 + z**2),
            2*(y*z + w*x)
        ])
        
        # Up (Z-axis in body frame)
        up = np.array([
            2*(x*z + w*y),
            2*(y*z - w*x),
            1 - 2*(x**2 + y**2)
        ])
        
        return heading, lateral, up
    
    def get_observation(self):
        """Construct observation matching SimpleFlight format."""
        # Relative positions to trajectory waypoints
        rel_pos = self.target_positions - self.position
        rel_pos_flat = rel_pos.flatten()  # 30D
        
        # Body frame orientation vectors
        heading, lateral, up = self.quat_to_rotation_vectors(self.orientation_quat)
        
        # Concatenate to match SimpleFlight observation space (42D)
        obs = np.concatenate([
            rel_pos_flat,      # 30D
            self.velocity,     # 3D
            heading,           # 3D
            lateral,           # 3D
            up,                # 3D
        ])
        
        return obs
    
    def policy_inference(self, obs):
        """Run policy inference on observation."""
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        
        if self.policy is None:
            # Use random actions for testing the pipeline
            logger.warning("Using random actions (policy not loaded properly)")
            action = np.random.uniform(-0.1, 0.1, size=4)
        else:
            with torch.no_grad():
                action_tensor = self.policy.act_inference(obs_tensor)
            action = action_tensor.squeeze(0).numpy()
        
        return action
    
    def send_ctbr_command(self, scf, action):
        """Send CTBR (Collective Thrust + Body Rates) command to Crazyflie.
        
        Args:
            scf: SyncCrazyflie object
            action: [roll_rate, pitch_rate, yaw_rate, thrust] from policy
        """
        # Policy outputs are typically normalized [-1, 1]
        # We need to scale them to Crazyflie firmware ranges
        
        # Body rates: scale to deg/s (Crazyflie uses deg/s)
        max_rate_deg_s = 200.0  # degrees per second
        roll_rate = np.clip(action[0] * max_rate_deg_s, -max_rate_deg_s, max_rate_deg_s)
        pitch_rate = np.clip(action[1] * max_rate_deg_s, -max_rate_deg_s, max_rate_deg_s)
        yaw_rate = np.clip(action[2] * max_rate_deg_s, -max_rate_deg_s, max_rate_deg_s)
        
        # Thrust: scale to 0-60000 range (Crazyflie PWM)
        # Hover thrust is around 36000-42000 depending on battery
        hover_thrust = 40000
        max_thrust_deviation = 15000
        thrust = hover_thrust + action[3] * max_thrust_deviation
        thrust = np.clip(thrust, 10000, 60000)
        
        # Send commander packet
        scf.cf.commander.send_setpoint(roll_rate, pitch_rate, yaw_rate, int(thrust))
    
    def run(self, duration_s: float = 30.0, control_rate_hz: float = 100.0):
        """Run deployment on real hardware.
        
        Args:
            duration_s: How long to run the policy (seconds)
            control_rate_hz: Control loop frequency (Hz)
        """
        cflib.crtp.init_drivers()
        
        logger.info(f"Connecting to {self.uri}")
        
        with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            # Set up logging for state estimate - simplified to avoid size issues
            log_state = LogConfig(name='StateEstimate', period_in_ms=10)
            log_state.add_variable('stateEstimate.x', 'float')
            log_state.add_variable('stateEstimate.y', 'float')
            log_state.add_variable('stateEstimate.z', 'float')
            log_state.add_variable('stateEstimate.vx', 'float')
            log_state.add_variable('stateEstimate.vy', 'float')
            log_state.add_variable('stateEstimate.vz', 'float')
            
            # Separate log config for orientation
            log_quat = LogConfig(name='Quaternion', period_in_ms=10)
            log_quat.add_variable('stateEstimate.qw', 'float')
            log_quat.add_variable('stateEstimate.qx', 'float')
            log_quat.add_variable('stateEstimate.qy', 'float')
            log_quat.add_variable('stateEstimate.qz', 'float')
            
            # Separate log config for gyro
            log_gyro = LogConfig(name='Gyro', period_in_ms=10)
            log_gyro.add_variable('gyro.x', 'float')
            log_gyro.add_variable('gyro.y', 'float')
            log_gyro.add_variable('gyro.z', 'float')
            
            scf.cf.log.add_config(log_state)
            scf.cf.log.add_config(log_quat)
            scf.cf.log.add_config(log_gyro)
            
            log_state.data_received_cb.add_callback(self.state_callback)
            log_quat.data_received_cb.add_callback(self.quat_callback)
            log_gyro.data_received_cb.add_callback(self.gyro_callback)
            
            log_state.start()
            log_quat.start()
            log_gyro.start()
            
            logger.info("Waiting for initial state estimate...")
            time.sleep(2.0)
            
            logger.info(f"Starting policy control for {duration_s}s at {control_rate_hz}Hz")
            logger.info(f"Trajectory type: {self.trajectory_type}")
            logger.info(f"Initial position: {self.position}")
            logger.info("")
            logger.info("=" * 60)
            logger.info("🛑 EMERGENCY STOP: Press Ctrl+C to immediately kill motors")
            logger.info("=" * 60)
            logger.info("")
            
            dt = 1.0 / control_rate_hz
            start_time = time.time()
            
            try:
                while time.time() - start_time < duration_s:
                    loop_start = time.time()
                    
                    # Get observation from sensors
                    obs = self.get_observation()
                    
                    # Run policy inference
                    action = self.policy_inference(obs)
                    
                    # Send CTBR command
                    self.send_ctbr_command(scf, action)
                    
                    # Log progress
                    if int((time.time() - start_time) * 10) % 10 == 0:
                        logger.info(f"Pos: {self.position}, Vel: {self.velocity}, Action: {action}")
                    
                    # Sleep to maintain control rate
                    elapsed = time.time() - loop_start
                    if elapsed < dt:
                        time.sleep(dt - elapsed)
                
                logger.info("Control finished, landing...")
                # Send zero rates and decreasing thrust to land
                for i in range(50):
                    thrust = 40000 - i * 800
                    scf.cf.commander.send_setpoint(0, 0, 0, int(thrust))
                    time.sleep(0.05)
                
            except KeyboardInterrupt:
                logger.warning("\n🛑 EMERGENCY STOP ACTIVATED!")
                logger.warning("Killing motors immediately...")
            
            finally:
                # Stop motors IMMEDIATELY
                scf.cf.commander.send_stop_setpoint()
                time.sleep(0.1)
                scf.cf.commander.send_stop_setpoint()  # Send twice to be sure
                log_state.stop()
                log_gyro.stop()
                logger.info("Motors stopped - Deployment complete")


def main():
    parser = argparse.ArgumentParser(description="Deploy SimpleFlight policy to Crazyflie 2.1")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E7", 
                        help="Crazyflie URI")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained policy checkpoint (.pt)")
    parser.add_argument("--trajectory", type=str, default="hover",
                        choices=["hover", "lemniscate", "polynomial"],
                        help="Trajectory type")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Deployment duration in seconds")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="Control loop rate in Hz")
    
    args = parser.parse_args()
    
    deployer = SimpleFlightDeployer(
        uri=args.uri,
        checkpoint_path=args.checkpoint,
        trajectory_type=args.trajectory
    )
    
    deployer.run(duration_s=args.duration, control_rate_hz=args.rate)


if __name__ == "__main__":
    main()
