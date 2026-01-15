# Phase 1 Complete: Calibration Setup ✓

## What We Just Built

### 1. Hover Thrust Measurement Tool
**File**: `IsaacLab/scripts/deployment/measure_hover_thrust.py`

Systematically measures your drone's actual hover thrust by:
- Gradually increasing thrust from 20k to 50k
- You feel when drone wants to lift
- Provides normalized values for physics calibration

**Usage**:
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\measure_hover_thrust.py
```

### 2. Policy Export Tool
**File**: `IsaacLab/scripts/deployment/export_policy.py`

Converts RSL-RL checkpoints to standalone deployment format:
- Extracts actor network from training checkpoint
- Removes critic and training-specific components
- Creates deployment-ready `.pt` file

**Usage**:
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\export_policy.py \
  --checkpoint logs/rsl_rl/.../model_5000.pt \
  --output models/my_policy.pt
```

### 3. Enhanced Deployment Script
**File**: `IsaacLab/scripts/deployment/deploy_standalone.py` (updated)

Added configurable thrust calibration:
- `--thrust-offset`: Add constant to thrust (for bias correction)
- `--thrust-scale`: Multiply thrust (for gain correction)
- Real-time logging shows final thrust values

**Usage**:
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py \
  --actor models/my_policy.pt \
  --thrust-offset 0.1 \
  --thrust-scale 1.0 \
  --duration 3
```

### 4. Calibration Guide
**File**: `IsaacLab/scripts/deployment/CALIBRATION_GUIDE.md`

Complete 6-phase workflow:
1. Measure real hardware
2. Update IsaacLab physics
3. Train calibrated policy
4. Export for deployment
5. Test deployment
6. Domain randomization (advanced)

### 5. Quick Start Script
**File**: `calibrate.bat`

Interactive menu for the entire workflow:
- Measure hover thrust
- Train policy
- Export policy
- Deploy to drone
- Test motors
- Open guide

**Usage**:
```bash
calibrate.bat
```

## Next Steps

### Immediate Action: Measure Your Drone

```bash
# Run this NOW to get calibration data
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\measure_hover_thrust.py
```

**What to expect**:
1. Hold drone firmly
2. Motors gradually increase
3. Feel when it pulls upward
4. Press Ctrl+C
5. Note the thrust value (e.g., 35000)

**Record the number** - you'll use it to calibrate simulation!

### Then: Start Training

Once you have hover thrust data:

1. **Update physics** (Optional but recommended):
   - Edit `simpleflight_env_cfg.py`
   - Set mass to your drone's mass
   - Or adjust thrust_to_weight

2. **Train policy**:
   ```bash
   D:\IsaacSim\python.bat IsaacLab\scripts\train_simpleflight.py \
     --task SimpleFlight-Track-Direct-v0 \
     --num_envs 2048 \
     --max_iterations 5000
   ```
   
   Time: 45-90 minutes
   
3. **Export trained model**:
   ```bash
   cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\export_policy.py \
     --checkpoint logs/rsl_rl/simpleflight_track/*/model_5000.pt
   ```

4. **Test deployment**:
   ```bash
   cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py \
     --actor models/my_trained_policy_standalone.pt \
     --duration 3 \
     --thrust-offset 0.0
   ```

## Key Advantages Over SimpleFlight's Pretrained Model

| Aspect | SimpleFlight Pretrained | Your Trained Model |
|--------|------------------------|-------------------|
| **Physics match** | OmniDrones (unknown params) | IsaacLab (you control) |
| **State estimation** | Motion capture expected | Flow Deck trained |
| **Mass calibration** | Their drone | YOUR drone |
| **Iteration speed** | Can't modify | 45 min retraining |
| **Understanding** | Black box | Full transparency |
| **Troubleshooting** | Stuck | Can fix and retrain |

## Expected Timeline

**Day 1 (Today)**:
- ✓ Measure hover thrust (5 min)
- ✓ Train first policy (45-90 min)
- ✓ Export and test deployment (10 min)
- Result: Know if approach works

**Day 2-3**:
- Iterate on thrust calibration
- Add domain randomization if needed
- Fine-tune physics parameters
- Result: Stable hover

**Week 1**:
- Train for different tasks
- Optimize performance
- Add trajectory following
- Result: Flight-ready system

## Comparison: Why This Works When SimpleFlight's Model Didn't

**SimpleFlight's model expects**:
- Mass: ~27g (their setup)
- Hover thrust: ~0.5 normalized (their sim)
- State: Motion capture quality
- Physics: OmniDrones parameters

**Your model will learn**:
- Mass: YOUR drone's actual mass
- Hover thrust: Measured from real hardware
- State: Flow Deck noise and drift
- Physics: Calibrated IsaacLab

**Result**: Sim-to-real transfer that actually transfers!

## Quick Start (Use calibrate.bat)

```bash
# Run interactive menu
calibrate.bat

# Or manual commands:

# 1. Measure (5 min)
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\measure_hover_thrust.py

# 2. Train (45 min)
D:\IsaacSim\python.bat IsaacLab\scripts\train_simpleflight.py --max_iterations 5000

# 3. Export (1 min)
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\export_policy.py --checkpoint <path>

# 4. Deploy (test)
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py --actor <path> --duration 3
```

## Success Criteria

After Phase 1, you should have:
- ✓ Measured hover thrust value
- ✓ Calibration tools installed
- ✓ Understanding of workflow
- ✓ Ready to train first policy

After full calibration:
- ✓ Motors spin reliably
- ✓ Drone attempts liftoff
- ✓ Stable hover achieved
- ✓ Can iterate and improve

## The Big Picture

SimpleFlight worked because they:
1. Measured their hardware
2. Calibrated simulation
3. Trained with matched physics
4. Deployed with tuning

You're doing the same - but with better tools and their lessons learned!

---

**Ready to start?** Run `measure_hover_thrust.py` now! 🚁
