#!/usr/bin/env python3
"""Export trained policy to a standalone PyTorch model for deployment."""

import argparse
import torch
from pathlib import Path

# Add IsaacLab paths
import sys
isaaclab_path = Path(__file__).resolve().parents[2]
sys.path.append(str(isaaclab_path / "source" / "extensions" / "isaaclab.rl.rsl_rl"))

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, OnPolicyRunner


def export_policy(checkpoint_path: str, output_path: str):
    """Export the actor network from a trained checkpoint.
    
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
