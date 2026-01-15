#!/usr/bin/env python3
"""Check what's in the SimpleFlight deploy.pt checkpoint."""

import torch
from pathlib import Path

checkpoint_path = Path(__file__).resolve().parents[3] / "models" / "deploy.pt"
print(f"Loading: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n=== Checkpoint Structure ===")
if isinstance(checkpoint, dict):
    print("Type: dict")
    print("Keys:", list(checkpoint.keys())[:20])  # First 20 keys
    
    # Check for known policy structures
    if 'actor' in checkpoint:
        print("\nFound 'actor' key - contains actor network")
    if 'critic' in checkpoint:
        print("Found 'critic' key - contains critic network")
    if 'model_state_dict' in checkpoint:
        print("Found 'model_state_dict' - contains full model")
        
    # Print some sample keys to understand structure
    print("\nFirst few state dict keys:")
    for i, key in enumerate(list(checkpoint.keys())[:10]):
        val = checkpoint[key]
        if torch.is_tensor(val):
            print(f"  {key}: Tensor{tuple(val.shape)}")
        else:
            print(f"  {key}: {type(val).__name__}")
else:
    print(f"Type: {type(checkpoint)}")
    if hasattr(checkpoint, '__dict__'):
        print("Attributes:", list(vars(checkpoint).keys()))

print("\n=== Analysis ===")
# Count parameters
total_params = 0
actor_params = 0
for key, val in checkpoint.items() if isinstance(checkpoint, dict) else []:
    if torch.is_tensor(val):
        params = val.numel()
        total_params += params
        if 'actor' in key.lower():
            actor_params += params

print(f"Total parameters: {total_params:,}")
print(f"Actor parameters: {actor_params:,}")
