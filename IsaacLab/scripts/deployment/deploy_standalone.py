#!/usr/bin/env python3
"""Deploy SimpleFlight actor network to Crazyflie (Windows standalone).

This uses the extracted actor weights without any OmniDrones/ROS dependencies.
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import logging

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleActorNetwork(nn.Module):
    """Standalone actor network matching SimpleFlight architecture."""
    
    def __init__(self, obs_dim=42, action_dim=4, hidden_dims=[256, 256, 256]):
        super().__init__()
        
        # Observation normalization (running mean/std from training)
        self.obs_norm = nn.Sequential(
            nn.LayerNorm(obs_dim, elementwise_affine=True),
        )
        
        # MLP encoder
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        
        # Action head (Gaussian distribution)
        self.action_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        """Forward pass - returns action mean (deterministic)."""
        # Normalize observation
        x = self.obs_norm(obs)
        # Encode
        x = self.encoder(x)
        # Get action mean
        action = self.action_mean(x)
        return action
    
    def load_weights(self, state_dict):
        """Load weights from extracted actor checkpoint."""
        # Map extracted keys to network structure
        new_state = {}
        
        for key, value in state_dict.items():
            # Remove 'module.' prefix if present
            new_key = key.replace('module.', '')
            
            # Map encoder layers
            if 'encoder.0' in new_key:
                # Layer norm for obs normalization
                new_key = new_key.replace('encoder.0', 'obs_norm.0')
            elif 'encoder.1.layers' in new_key:
                # MLP layers - extract layer index and remap
                # encoder.1.layers.0 -> encoder.0 (Linear)
                # encoder.1.layers.2 -> encoder.1 (LayerNorm)
                # encoder.1.layers.3 -> encoder.3 (Linear)
                parts = new_key.split('.')
                layer_idx = int(parts[3])
                
                # Map to sequential index (0,2,3,5,6,8 -> 0,1,3,4,6,7)
                if layer_idx in [0, 3, 6]:  # Linear layers
                    seq_idx = {0: 0, 3: 3, 6: 6}[layer_idx]
                elif layer_idx in [2, 5, 8]:  # LayerNorm layers
                    seq_idx = {2: 1, 5: 4, 8: 7}[layer_idx]
                else:
                    continue
                
                new_key = f"encoder.{seq_idx}.{'.'.join(parts[4:])}"
            elif 'act_dist.fc_mean' in new_key:
                new_key = new_key.replace('act_dist.fc_mean', 'action_mean')
            elif 'act_dist.log_std' in new_key:
                new_key = 'log_std'
            
            new_state[new_key] = value
        
        self.load_state_dict(new_state, strict=False)
        logger.info(f"Loaded {len(new_state)} parameters into network")


class SimpleFlightDeployer:
    """Deploy SimpleFlight policy using standalone actor network."""
    
    def __init__(self, uri: str, actor_path: str, device='cpu'):
        self.uri = uri
        self.device = device
        
        # Load actor network
        logger.info(f"Loading actor from {actor_path}")
        self.actor = SimpleActorNetwork()
        actor_weights = torch.load(actor_path, map_location=device)
        self.actor.load_weights(actor_weights)
        self.actor.eval()
        logger.info("Actor loaded successfully!")
        
        # State buffers
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        
        # Trajectory (hover at origin)
        self.target_positions = np.zeros((10, 3))
        self.origin = np.array([0.0, 0.0, 1.0])  # Hover at 1m
    
    def state_callback(self, timestamp, data, logconf):
        self.position[0] = data['stateEstimate.x']
        self.position[1] = data['stateEstimate.y']
        self.position[2] = data['stateEstimate.z']
        self.velocity[0] = data['stateEstimate.vx']
        self.velocity[1] = data['stateEstimate.vy']
        self.velocity[2] = data['stateEstimate.vz']
    
    def quat_callback(self, timestamp, data, logconf):
        self.orientation_quat[0] = data['stateEstimate.qw']
        self.orientation_quat[1] = data['stateEstimate.qx']
        self.orientation_quat[2] = data['stateEstimate.qy']
        self.orientation_quat[3] = data['stateEstimate.qz']
    
    def quat_to_rotation_vectors(self, quat):
        """Convert quaternion to body frame axes."""
        w, x, y, z = quat
        heading = np.array([1 - 2*(y**2 + z**2), 2*(x*y + w*z), 2*(x*z - w*y)])
        lateral = np.array([2*(x*y - w*z), 1 - 2*(x**2 + z**2), 2*(y*z + w*x)])
        up = np.array([2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x**2 + y**2)])
        return heading, lateral, up
    
    def get_observation(self):
        """Construct 42D observation matching SimpleFlight."""
        # Update target waypoints
        for i in range(10):
            self.target_positions[i] = self.origin
        
        # Relative positions (30D)
        rel_pos = self.target_positions - self.position
        rel_pos_flat = rel_pos.flatten()
        
        # Body frame orientation (9D)
        heading, lateral, up = self.quat_to_rotation_vectors(self.orientation_quat)
        
        # Concatenate: 30 + 3 + 9 = 42D
        obs = np.concatenate([rel_pos_flat, self.velocity, heading, lateral, up])
        return obs
    
    def policy_inference(self, obs):
        """Run actor network inference."""
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        
        with torch.no_grad():
            action_tensor = self.actor(obs_tensor)
            action = torch.tanh(action_tensor)  # Squash to [-1, 1]
        
        return action.squeeze(0).numpy()
    
    def send_ctbr_command(self, scf, action):
        """Send CTBR command."""
        rpy_scale = 30  # deg/s (reduced for safety)
        
        # Force minimum thrust for testing
        min_thrust = 35000  # ~53% thrust - enough to feel motors
        max_thrust = 45000  # ~68% thrust - safe for hand test
        
        roll_rate = float(action[0] * rpy_scale)
        pitch_rate = float(action[1] * rpy_scale)
        yaw_rate = float(action[2] * rpy_scale)
        
        # Remap action[-1,1] to thrust range
        thrust_normalized = (action[3] + 1) / 2  # [0, 1]
        thrust = min_thrust + thrust_normalized * (max_thrust - min_thrust)
        thrust_uint16 = int(np.clip(thrust, min_thrust, max_thrust))
        
        scf.cf.commander.send_setpoint(roll_rate, pitch_rate, yaw_rate, thrust_uint16)
    
    def run(self, duration_s=10.0, control_rate_hz=100.0):
        """Run policy control loop."""
        cflib.crtp.init_drivers()
        
        logger.info(f"Connecting to {self.uri}")
        
        with SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            # Setup logging
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
            
            # CRITICAL: Disable high-level commander so low-level CTBR works
            logger.info("Disabling high-level commander for low-level control...")
            scf.cf.param.set_value('commander.enHighLevel', '0')
            time.sleep(0.1)
            
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
                    obs = self.get_observation()
                    action = self.policy_inference(obs)
                    self.send_ctbr_command(scf, action)
                    
                    if step % int(0.1 / dt) == 0:
                        logger.info(f"Pos: {self.position}, Vel: {self.velocity}, Action: {action}")
                    
                    time.sleep(dt)
                    
            except KeyboardInterrupt:
                logger.warning("\n🛑 EMERGENCY STOP - Ctrl+C pressed!")
                for _ in range(10):
                    scf.cf.commander.send_stop_setpoint()
                    time.sleep(0.01)
                logger.info("Motors stopped")
                return
            
            logger.info("Control finished, stopping motors...")
            for _ in range(10):
                scf.cf.commander.send_stop_setpoint()
                time.sleep(0.01)
            logger.info("Motors stopped - Deployment complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=str, 
                       default="D:\\coding\\Capstone\\SimpleFlight-experimenting\\models\\actor_standalone.pt",
                       help="Path to extracted actor (.pt file)")
    parser.add_argument("--uri", type=str, default="radio://0/80/2M/E7E7E7E7E7",
                       help="Crazyflie URI")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Flight duration in seconds")
    parser.add_argument("--rate", type=float, default=100.0,
                       help="Control rate in Hz")
    
    args = parser.parse_args()
    
    deployer = SimpleFlightDeployer(uri=args.uri, actor_path=args.actor)
    deployer.run(duration_s=args.duration, control_rate_hz=args.rate)


if __name__ == "__main__":
    main()
