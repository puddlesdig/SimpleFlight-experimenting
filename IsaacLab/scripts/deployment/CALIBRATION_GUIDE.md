# SimpleFlight Calibration Guide

Complete guide for calibrating IsaacLab physics to match real hardware for sim-to-real transfer.

## Phase 1: Measure Real Hardware

### Step 1: Measure Hover Thrust

```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\measure_hover_thrust.py --uri radio://0/80/2M/E7E7E7E7E7
```

**Safety**: Hold drone firmly or use test stand!

**What it does**:
- Gradually increases thrust from 20,000 to 50,000 (in steps of 1,000)
- You feel when drone starts pulling upward
- Press Ctrl+C when you feel lift-off force
- Note the thrust value

**Expected result**: Hover thrust ≈ 30,000-40,000 (0.46-0.61 normalized)

### Step 2: Record Calibration Data

Create a file `calibration_data.txt`:
```
Hover Thrust: 35000 (example)
Normalized: 35000/65535 = 0.534
Battery: 3.7V (or your voltage)
Mass: 35g (including Flow Deck)
```

## Phase 2: Update IsaacLab Physics

### Option A: Quick Fix (Thrust Offset)

Update deployment command:
```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py \
  --thrust-offset 0.034 \
  --thrust-scale 1.0
```

Calculate offset:
- If measured hover = 0.534 normalized
- Policy outputs ≈ 0.5 for hover
- Offset = 0.534 - 0.5 = 0.034

### Option B: Fix Simulation Physics (Recommended)

Edit `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/simpleflight/simpleflight_env_cfg.py`:

```python
# Find SIMPLEFLIGHT_CFG and update mass
mass_props=sim_utils.RigidBodyPropertiesCfg(
    mass=0.035,  # Update to match your drone (kg)
),
```

Or adjust thrust-to-weight:
```python
# In drone config
thrust_to_weight=1.9,  # Tune this value
```

## Phase 3: Train Calibrated Policy

```bash
# Train with calibrated physics
D:\IsaacSim\python.bat scripts\train_simpleflight.py \
  --task SimpleFlight-Track-Direct-v0 \
  --num_envs 2048 \
  --max_iterations 5000
```

Monitor training:
- Episode length should increase (>1)
- Tracking error should decrease (<0.5m)
- Typical training time: 45-90 minutes

## Phase 4: Export Trained Policy

```bash
# After training completes
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\export_policy.py \
  --checkpoint logs/rsl_rl/simpleflight_track/YYYY-MM-DD_HH-MM-SS/model_5000.pt \
  --output models/my_trained_policy.pt
```

## Phase 5: Test Deployment

### Safety Check
1. Remove propellers first!
2. Test motor commands work
3. Add propellers
4. Hand-hold test
5. Test stand if available
6. Free flight in open space

### Deploy with Calibration

```bash
cf_deploy_env\Scripts\python.exe IsaacLab\scripts\deployment\deploy_standalone.py \
  --actor models/my_trained_policy.pt \
  --uri radio://0/80/2M/E7E7E7E7E7 \
  --duration 3 \
  --thrust-offset 0.0 \
  --thrust-scale 1.0
```

Start with short duration (3s), increase gradually.

### Iterative Tuning

If motors don't spin:
1. Increase `--thrust-offset` by 0.1
2. Test again
3. Repeat until motors spin

If too aggressive:
1. Decrease `--thrust-offset` by 0.05
2. Or reduce `--thrust-scale` to 0.9
3. Test again

## Phase 6: Domain Randomization (Advanced)

For robust sim-to-real transfer, add randomization during training.

Edit `simpleflight_env_cfg.py`:

```python
# Add to SIMPLEFLIGHT_CFG
randomization=RandomizationCfg(
    # Randomize mass ±10%
    mass_props=RigidBodyPropertiesCfg(
        mass=UniformRandomization(lower=0.032, upper=0.038)
    ),
    # Randomize thrust-to-weight ±15%
    actuators=dict(
        thrust_to_weight=UniformRandomization(lower=1.6, upper=2.2)
    ),
)
```

Retrain with randomization - policy becomes robust to variations!

## Troubleshooting

### Motors Don't Spin
- ✓ Check thrust value in logs
- ✓ Increase `--thrust-offset`
- ✓ Verify `commander.enHighLevel = 0`
- ✓ Test with `test_motors.py` first

### Drone Flips/Crashes
- Policy not trained enough (run more iterations)
- Wrong physics calibration (measure again)
- Flow Deck position drift (add noise to training)
- Commands too aggressive (reduce `--thrust-scale`)

### Training Not Improving
- Episode length = 1 → reward issue, check task
- High tracking error → physics mismatch
- NaN values → learning rate too high

## Quick Reference

**Measure hover thrust**:
```bash
python measure_hover_thrust.py
```

**Train policy**:
```bash
D:\IsaacSim\python.bat scripts\train_simpleflight.py --max_iterations 5000
```

**Export policy**:
```bash
python export_policy.py --checkpoint logs/.../model_5000.pt
```

**Deploy**:
```bash
python deploy_standalone.py --actor models/my_policy.pt --duration 3
```

**Calibration parameters**:
- `--thrust-offset`: Add to thrust (typical: 0.0 to 0.2)
- `--thrust-scale`: Multiply thrust (typical: 0.8 to 1.2)

## Success Criteria

✓ Motors spin reliably  
✓ Drone attempts takeoff  
✓ Stable hover for >5 seconds  
✓ Tracks trajectory smoothly  
✓ Lands safely  

Each improvement validates your calibration!
