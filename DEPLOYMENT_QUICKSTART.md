# Crazyflie Deployment - Quick Start Guide

## The Problem
IsaacLab requires numpy<2, but cflib requires numpy~=2.2. They conflict!

## Solution: Use Separate Environments

### Option 1: Create Deployment Environment (Recommended)

```bash
# Create new Python environment for deployment
python -m venv cf_deploy_env

# Activate it
cf_deploy_env\Scripts\activate

# Install cflib and dependencies
pip install cflib torch

# Copy your trained model
copy logs\rsl_rl\simpleflight_track\2026-01-14_13-40-06\model_999.pt cf_deploy_env\

# Run deployment
python IsaacLab\scripts\deployment\deploy_to_crazyflie.py \
    --checkpoint cf_deploy_env\model_999.pt \
    --uri radio://0/80/2M/E7E7E7E7E7 \
    --trajectory hover \
    --duration 5
```

### Option 2: Quick Test WITHOUT Virtual Environment

Since you want to test NOW, here's a quick test using the Crazyflie client:

#### Step 1: Install Crazyflie Client
Download from: https://github.com/bitcraze/crazyflie-clients-python/releases
(This comes with cflib bundled)

#### Step 2: Test Connection
1. Plug in Crazyradio 2.0
2. Turn on Crazyflie 2.1
3. Open Crazyflie Client
4. Click "Scan" - should find your drone
5. Connect

#### Step 3: Manual Policy Test (Simple Version)

Since cflib conflicts with IsaacLab, let's export the policy and use it separately:

```bash
# 1. Export policy to ONNX (standalone format)
D:\IsaacSim\python.bat -c "
import torch
from rsl_rl.modules import ActorCritic

# Load checkpoint
checkpoint = torch.load('logs/rsl_rl/simpleflight_track/2026-01-14_13-40-06/model_999.pt', map_location='cpu')

# Create policy
policy = ActorCritic(42, 42, 4, [256,256,256], [256,256,256], 'elu')
policy.load_state_dict(checkpoint['model_state_dict'])
policy.eval()

# Export to ONNX
dummy_input = torch.randn(1, 42)
torch.onnx.export(policy.actor, dummy_input, 'policy.onnx', 
                  input_names=['observation'], 
                  output_names=['action'],
                  dynamic_axes={'observation': {0: 'batch'}, 'action': {0: 'batch'}})
print('✓ Policy exported to policy.onnx')
"
```

### Option 3: Test in Simulation First (SAFEST)

Before testing on real hardware, let's visualize what the policy does:

```bash
# Run with GUI to see what policy does
D:\IsaacSim\python.bat IsaacLab\scripts\tools\test_simpleflight.py \
    --num_envs 1 \
    --num_steps 500 \
    --checkpoint logs\rsl_rl\simpleflight_track\2026-01-14_13-40-06\model_999.pt
```

(We need to add --checkpoint support to test script first)

## What to expect with current policy

⚠️ **WARNING**: Your policy only trained for 1000 iterations. It's NOT ready for real flight!

- **Tracking error**: 2.16m (should be <0.1m for good hover)
- **Episode length**: 1 step (crashes immediately)
- **Status**: **NOT FLIGHT READY**

**Recommendation**: Train for 5000-10000 iterations first!

## Safe Testing Steps (When Policy is Ready)

1. **Use a net/cage** - policy will likely crash
2. **Start with 3-second flights**
3. **Have kill switch ready** (hand near drone)
4. **Check battery >3.8V**
5. **Clear flight area**
6. **Use motion capture** for position (don't rely on IMU alone)

## Next Steps

Choose one:
- [ ] Create deployment venv and test (15 min setup)
- [ ] Continue training to 5000 iterations (recommended)
- [ ] Test in IsaacLab simulation with visualization first
