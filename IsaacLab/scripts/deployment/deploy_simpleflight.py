#!/usr/bin/env python3
"""Deploy SimpleFlight policy using their original pipeline approach.

This adapts the SimpleFlight rl_track.py deployment pattern to work with
cflib directly (without ROS 2), loading the policy via MAPPOPolicy.
"""

import argparse
import time
import numpy as np
import torch
import logging
import sys
from pathlib import Path

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Add SimpleFlight path for policy loading
simpleflight_path = Path(__file__).resolve().parents[3]
sys.path.append(str(simpleflight_path))

from omni_drones.learning import MAPPOPolicy
from omni_drones.utils.torchrl import AgentSpec
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec
from tensordict import TensorDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleFlightDeployer:
    """Deploy SimpleFlight using MAPPO policy and FakeTrack-style observations."""
    
    def __init__(self, uri: str, checkpoint_path: str, device='cpu'):
        self.uri = uri
        self.device = device
        
        # Load policy the SimpleFlight way (using MAPPOPolicy)
        logger.info(f"Loading policy from {checkpoint_path}")
        
        # Define agent spec matching SimpleFlight Track environment
        observation_dim = 42  # 30 (traj) + 3 (vel) + 9 (rotation)
        self.agent_spec = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        
        # Create minimal config for MAPPOPolicy
        class PolicyConfig:
            def __init__(self):
                self.name = "mappo"
                self.ppo = type('obj', (object,), {
                    'num_epochs': 8,
                    'num_minibatches': 32,
                    'clip_param': 0.2,
                    'entropy_coef': 0.0,
                    'value_loss_coef': 1.0,
                    'max_grad_norm': 1.0,
                    'gae_lambda': 0.95,
                })()
                self.network = type('obj', (object,), {
                    'rnn': False,
                    'hidden_units': [256, 256, 256],
                    'activation': 'elu',
                })()
        
        cfg = PolicyConfig()
        
        # Create policy
        self.policy = MAPPOPolicy(cfg, agent_spec=self.agent_spec, device=self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        
        logger.info("Policy loaded successfully!")
        
        # State buffers
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        
        # Trajectory tracking
        self.target_positions = np.zeros((10, 3))  # 10 waypoints
        self.origin = np.array([0.0, 0.0, 1.0])  # Hover target
        
    def state_callback(self, timestamp, data, logconf):
        """Callback for state estimate."""
        self.position[0] = data['stateEstimate.x']
        self.position[1] = data['stateEstimate.y']
        self.position[2] = data['stateEstimate.z']
        self.velocity[0] = data['stateEstimate.vx']
        self.velocity[1] = data['stateEstimate.vy']
        self.velocity[2] = data['stateEstimate.vz']
    
    def quat_callback(self, timestamp, data, logconf):
        """Callback for quaternion."""
        self.orientation_quat[0] = data['stateEstimate.qw']
        self.orientation_quat[1] = data['stateEstimate.qx']
        self.orientation_quat[2] = data['stateEstimate.qy']
        self.orientation_quat[3] = data['stateEstimate.qz']
    
    def quat_to_rotation_vectors(self, quat):
        """Convert quaternion to body frame axes (heading, lateral, up)."""
        w, x, y, z = quat
        
        # Heading (forward) direction
        heading = np.array([
            1 - 2*(y**2 + z**2),
            2*(x*y + w*z),
            2*(x*z - w*y)
        ])
        
        # Lateral (right) direction  
        lateral = np.array([
            2*(x*y - w*z),
            1 - 2*(x**2 + z**2),
            2*(y*z + w*x)
        ])
        
        # Up direction
        up = np.array([
            2*(x*z + w*y),
            2*(y*z - w*x),
            1 - 2*(x**2 + y**2)
        ])
        
        return heading, lateral, up
    
    def get_observation(self):
        """Construct observation matching SimpleFlight format."""
        # Update target waypoints (hover at origin for all future steps)
        for i in range(10):
            self.target_positions[i] = self.origin
        
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
        """Run policy inference using SimpleFlight's approach."""
        # Create TensorDict matching SimpleFlight format
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(1)  # [1, 1, 42]
        
        data = TensorDict({
            "agents": {
                "observation": obs_tensor,
            }
        }, batch_size=[1])
        
        with torch.no_grad():
            data = self.policy(data, deterministic=True)
        
        action = torch.tanh(data[("agents", "action")])  # SimpleFlight applies tanh
        action = action.squeeze(0).squeeze(0).numpy()  # [4]
        
        return action
    
    def send_ctbr_command(self, scf, action):
        """Send CTBR command to Crazyflie (SimpleFlight style)."""
        # SimpleFlight action: [roll_rate, pitch_rate, yaw_rate, thrust]
        # Already in [-1, 1] range after tanh
        
        # Scale to Crazyflie firmware ranges
        rpy_scale = 180  # degrees/s (SimpleFlight default)
        min_thrust = 0.0
        max_thrust = 0.9
        
        roll_rate = float(action[0] * rpy_scale)
        pitch_rate = float(action[1] * rpy_scale)
        yaw_rate = float(action[2] * rpy_scale)
        
        # Thrust: [-1, 1] -> [min_thrust, max_thrust] in 0-1 range
        thrust_normalized = (action[3] + 1) / 2  # [-1,1] -> [0,1]
        thrust = np.clip(thrust_normalized, min_thrust, max_thrust)
        thrust_uint16 = int(thrust * 65535)  # Convert to uint16
        
        scf.cf.commander.send_setpoint(roll_rate, pitch_rate, yaw_rate, thrust_uint16)
    
    def run(self, duration_s=10.0, control_rate_hz=100.0):
        """Run policy control loop."""
        cflib.crtp.init_drivers()
        
        logger.info(f"Connecting to {self.uri}")
        
        with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            # Set up logging (split configs to avoid size limit)
            log_state = LogConfig(name='StateEstimate', period_in_ms=10)
            log_state.add_variable('stateEstimate.x', 'float')
            log_state.add_variable('stateEstimate.y', 'float')
            log_state.add_variable('stateEstimate.z', 'float')
            log_state.add_variable('stateEstimate.vx', 'float')
            log_state.add_variable('stateEstimate.vy', 'float')
            log_state.add_variable('stateEstimate.vz', 'float')
            
            log_quat = LogConfig(name='Quaternion', period_in_ms=10)
            log_quat.add_variable('stateEstimate.qw', 'float')
            log_quat.add_variable('stateEstimate.qx', 'float')
            log_quat.add_variable('stateEstimate.qy', 'float')
            log_quat.add_variable('stateEstimate.qz', 'float')
            
            scf.cf.log.add_config(log_state)
            scf.cf.log.add_config(log_quat)
            
            log_state.data_received_cb.add_callback(self.state_callback)
            log_quat.data_received_cb.add_callback(self.quat_callback)
            
            log_state.start()
            log_quat.start()
            
            logger.info("Waiting for initial state estimate...")
            time.sleep(2.0)
            
            logger.info(f"Starting policy control for {duration_s}s at {control_rate_hz}Hz")
            logger.info(f"Hover target: {self.origin}")
            logger.info(f"Initial position: {self.position}")
            logger.info("")
            logger.info("=" * 60)
            logger.info("🛑 EMERGENCY STOP: Press Ctrl+C to immediately kill motors")
            logger.info("=" * 60)
            logger.info("")
            
            dt = 1.0 / control_rate_hz
            num_steps = int(duration_s / dt)
            
            try:
                for step in range(num_steps):
                    # Get observation
                    obs = self.get_observation()
                    
                    # Policy inference
                    action = self.policy_inference(obs)
                    
                    # Send command
                    self.send_ctbr_command(scf, action)
                    
                    # Log every 100ms
                    if step % int(0.1 / dt) == 0:
                        logger.info(f"Pos: {self.position}, Vel: {self.velocity}, Action: {action}")
                    
                    time.sleep(dt)
                    
            except KeyboardInterrupt:
                logger.warning("\n🛑 EMERGENCY STOP - Ctrl+C pressed!")
                # Send zero commands twice
                for _ in range(2):
                    scf.cf.commander.send_stop_setpoint()
                    time.sleep(0.01)
                logger.info("Motors stopped")
                return
            
            logger.info("Control finished, landing...")
            # Send stop command
            for _ in range(2):
                scf.cf.commander.send_stop_setpoint()
                time.sleep(0.01)
            logger.info("Motors stopped - Deployment complete")


def main():
    parser = argparse.ArgumentParser(description="Deploy SimpleFlight policy to Crazyflie")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to policy checkpoint (.pt file)")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E7",
                       help="Crazyflie URI")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Flight duration in seconds")
    parser.add_argument("--rate", type=float, default=100.0,
                       help="Control rate in Hz")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for policy inference (cpu/cuda)")
    
    args = parser.parse_args()
    
    deployer = SimpleFlightDeployer(
        uri=args.uri,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    deployer.run(duration_s=args.duration, control_rate_hz=args.rate)


if __name__ == "__main__":
    main()
