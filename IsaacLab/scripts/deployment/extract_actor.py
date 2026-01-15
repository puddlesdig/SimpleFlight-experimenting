#!/usr/bin/env python3
"""Extract actor network from SimpleFlight deploy.pt for standalone deployment."""

import torch
import argparse
from pathlib import Path


def extract_actor(checkpoint_path: str, output_path: str):
    """Extract actor network from MAPPO checkpoint.
    
    SimpleFlight's deploy.pt contains:
    - actor_params: TensorDictParams (the actor network weights)
    - critic: OrderedDict (not needed for deployment)
    - value_normalizer: OrderedDict (not needed)
    
    We extract just the actor weights into a simple state dict.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("\nCheckpoint keys:", list(checkpoint.keys()))
    
    # Extract actor_params
    if 'actor_params' in checkpoint:
        actor_params = checkpoint['actor_params']
        print(f"Actor params type: {type(actor_params)}")
        
        # TensorDictParams has a _param_td attribute containing the actual parameters
        if hasattr(actor_params, '_param_td'):
            actor_state = {}
            param_td = actor_params._param_td
            
            # Extract all parameters from TensorDict
            for key in param_td.keys(True, True):
                # Convert tuple key to string (e.g., ('actor', 'linear_0', 'weight') -> 'actor.linear_0.weight')
                if isinstance(key, tuple):
                    key_str = '.'.join(str(k) for k in key)
                else:
                    key_str = str(key)
                
                value = param_td[key]
                actor_state[key_str] = value
                print(f"  {key_str}: {value.shape}")
            
            # Save extracted weights
            torch.save(actor_state, output_path)
            print(f"\n✓ Extracted {len(actor_state)} parameters")
            print(f"✓ Saved to: {output_path}")
            
            # Calculate total parameters
            total_params = sum(p.numel() for p in actor_state.values())
            print(f"✓ Total parameters: {total_params:,}")
            
        else:
            print("Error: actor_params doesn't have expected structure")
            print("Available attributes:", dir(actor_params))
    else:
        print("Error: No 'actor_params' found in checkpoint")
        print("Available keys:", list(checkpoint.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
                       default=str(Path(__file__).parents[3] / "models" / "deploy.pt"),
                       help="Path to SimpleFlight deploy.pt")
    parser.add_argument("--output", type=str,
                       default=str(Path(__file__).parents[3] / "models" / "actor_standalone.pt"),
                       help="Output path for extracted actor")
    
    args = parser.parse_args()
    
    extract_actor(args.checkpoint, args.output)
