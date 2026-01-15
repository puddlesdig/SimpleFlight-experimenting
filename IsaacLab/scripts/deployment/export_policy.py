#!/usr/bin/env python3
"""Export RSL-RL trained policy to standalone deployment format.

This converts an RSL-RL checkpoint (ActorCritic) to the standalone
SimpleActorNetwork format used by deploy_standalone.py.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path


class SimpleActorNetwork(nn.Module):
    """Standalone actor network matching deployment format."""
    
    def __init__(self, obs_dim=42, action_dim=4, hidden_dims=[256, 256, 256]):
        super().__init__()
        
        # Observation normalization
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
        
        # Action head
        self.action_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs):
        normalized_obs = self.obs_norm(obs)
        features = self.encoder(normalized_obs)
        action_mean = self.action_mean(features)
        return action_mean


def export_policy(checkpoint_path, output_path):
    """Export actor network from RSL-RL checkpoint.
    
    Args:
        checkpoint_path: Path to the training checkpoint (.pt file)
        output_path: Where to save the exported actor network
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # The checkpoint should contain the full model
    if 'model_state_dict' in checkpoint:
        print("Found model_state_dict in checkpoint")
        # Need to reconstruct the ActorCritic model
        # For now, save just the state dict
        torch.save(checkpoint['model_state_dict'], output_path)
    elif hasattr(checkpoint, 'actor'):
        # Full model saved
        print("Extracting actor network")
        torch.save(checkpoint.actor, output_path)
    else:
        print("Saving entire checkpoint")
        torch.save(checkpoint, output_path)
    
    print(f"Exported policy to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to training checkpoint")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for exported policy")
    args = parser.parse_args()
    
    export_policy(args.checkpoint, args.output)
