# SimpleFlight Real Hardware Deployment Guide

This guide explains how to deploy your trained SimpleFlight policy to a real Crazyflie 2.1 drone using Crazyradio 2.0.

## Prerequisites

### Hardware
- **Crazyflie 2.1** with IMU (built-in)
- **Crazyradio 2.0 PA** USB dongle
- **Motion capture system** (Vicon, OptiTrack, etc.) OR **Loco Positioning System** for position tracking
- Fully charged battery

### Software
```bash
# Install Crazyflie Python library
pip install cflib

# Already installed from training
# - torch
# - numpy
# - rsl_rl
```

## Step-by-Step Deployment

### 1. Training Completion
After training completes (1000 iterations), you'll have checkpoints saved at:
```
logs/rsl_rl/simpleflight_track/YYYY-MM-DD_HH-MM-SS/model_500.pt
logs/rsl_rl/simpleflight_track/YYYY-MM-DD_HH-MM-SS/model_1000.pt
```

### 2. Configure Crazyflie Firmware

**Enable State Estimator:**
The Crazyflie needs a position estimate. Configure one of these:

**Option A: Motion Capture (Recommended for indoor labs)**
```python
# In Crazyflie client or via cflib
cf.param.set_value('stabilizer.estimator', '2')  # Use EKF estimator
cf.param.set_value('stabilizer.controller', '2')  # Use Mellinger controller as base
```

Then send external position via:
```python
cf.extpos.send_extpos(x, y, z)  # meters, from mocap
```

**Option B: Loco Positioning System**
```python
cf.param.set_value('stabilizer.estimator', '2')  # EKF
# Anchor positions must be pre-configured
```

**Option C: Lighthouse (if you have base stations)**
```python
cf.param.set_value('stabilizer.estimator', '2')
```

### 3. Find Your Crazyflie URI

Scan for available Crazyflies:
```python
import cflib.crtp
cflib.crtp.init_drivers()

print("Scanning for Crazyflies...")
available = cflib.crtp.scan_interfaces()
for i in available:
    print(f"Found: {i[0]}")
```

Typical URIs:
- `radio://0/80/2M/E7E7E7E7E7` (default)
- `radio://0/80/2M/E7E7E7E701` (if you changed address)

### 4. Test Connection

Before deploying the policy, verify connection:
```python
python -c "
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

cflib.crtp.init_drivers()
uri = 'radio://0/80/2M/E7E7E7E7E7'  # Your URI

print(f'Connecting to {uri}...')
with SyncCrazyflie(uri) as scf:
    print('Connected!')
    print(f'Battery: {scf.cf.param.get_value(\"pm.vbat\")}V')
"
```

### 5. Deploy the Policy

**Basic hover deployment:**
```bash
cd D:\coding\Capstone\SimpleFlight-experimenting\IsaacLab

# Find your best checkpoint
dir logs\rsl_rl\simpleflight_track\

# Deploy (30 seconds, hover at 1m altitude)
python scripts\deployment\deploy_to_crazyflie.py \
    --checkpoint logs/rsl_rl/simpleflight_track/2026-01-14_13-40-08/model_1000.pt \
    --uri radio://0/80/2M/E7E7E7E7E7 \
    --trajectory hover \
    --duration 30 \
    --rate 100
```

**Parameters:**
- `--checkpoint`: Path to trained model (`.pt` file)
- `--uri`: Your Crazyflie's radio URI
- `--trajectory`: `hover`, `lemniscate`, or `polynomial`
- `--duration`: Flight duration in seconds
- `--rate`: Control loop frequency (Hz), default 100

### 6. Safety Precautions

⚠️ **IMPORTANT SAFETY STEPS:**

1. **Start with short flights**: Use `--duration 5` first
2. **Test at low altitude**: Modify hover height to 0.3m initially
3. **Clear flight space**: Remove obstacles, have net/cage if possible
4. **Emergency stop**: Keep hand near drone, be ready to catch/kill motors
5. **Battery check**: Ensure >3.7V before flight
6. **Propeller check**: Verify all propellers secure and undamaged

### 7. Observation Mapping (Sim-to-Real)

The deployment script automatically maps:

| Simulator | Real Crazyflie | Source |
|-----------|----------------|--------|
| Position | `stateEstimate.x/y/z` | Mocap/LPS/Lighthouse |
| Velocity | `stateEstimate.vx/vy/vz` | EKF integration |
| Orientation | `stateEstimate.qw/qx/qy/qz` | Complementary filter (IMU) |
| Angular velocity | `gyro.x/y/z` | Gyroscope (deg/s → rad/s) |

### 8. Action Scaling

The policy outputs normalized actions `[-1, 1]` which are scaled to:

| Action | Range | Crazyflie Command |
|--------|-------|-------------------|
| Roll rate | ±200°/s | `commander.send_setpoint(roll, ...)` |
| Pitch rate | ±200°/s | `commander.send_setpoint(_, pitch, ...)` |
| Yaw rate | ±200°/s | `commander.send_setpoint(_, _, yaw, _)` |
| Thrust | 10000-60000 | `commander.send_setpoint(_, _, _, thrust)` |

Hover thrust is typically 36000-42000 (adjust in script if needed).

### 9. Troubleshooting

**Issue: Drone doesn't take off**
- Check thrust scaling in `deploy_to_crazyflie.py` line 178
- Increase `hover_thrust` to 42000 if battery is old
- Verify propellers are on correct motors (CW/CCW)

**Issue: Unstable flight**
- Reduce control rate: `--rate 50`
- Check mocap tracking quality (no occlusions)
- Verify gyro calibration (place flat, don't move during startup)

**Issue: Position drifts**
- Improve mocap calibration
- Check LPS anchor positions
- Enable state estimator reset: `cf.param.set_value('kalman.resetEstimation', '1')`

**Issue: Connection drops**
- Reduce distance to <10m
- Check Crazyradio antenna orientation
- Avoid 2.4GHz interference (WiFi, Bluetooth)

**Issue: Policy performs poorly**
- Sim-to-real gap: Train with more noise/domain randomization
- Re-train with `--enable_dr` flag
- Fine-tune on real hardware (collect real trajectories, continue training)

### 10. Advanced: External Position Input

If using motion capture, you need to stream position to the Crazyflie:

```python
# In a separate thread/process
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

def stream_mocap_position(uri, get_mocap_pos_func):
    """Stream mocap position to Crazyflie."""
    with SyncCrazyflie(uri) as scf:
        while True:
            x, y, z = get_mocap_pos_func()  # Your mocap system
            scf.cf.extpos.send_extpos(x, y, z)
            time.sleep(0.01)  # 100Hz
```

Run this before deploying the policy.

### 11. Logging Flight Data

Modify `deploy_to_crazyflie.py` to log data:

```python
# Add to SimpleFlightDeployer.__init__:
self.log_data = []

# In run() loop, after policy inference:
self.log_data.append({
    'time': time.time() - start_time,
    'position': self.position.copy(),
    'velocity': self.velocity.copy(),
    'action': action.copy(),
})

# After flight:
import pickle
with open('flight_log.pkl', 'wb') as f:
    pickle.dump(self.log_data, f)
```

### 12. Next Steps

**If policy works well:**
- Increase flight duration
- Try lemniscate trajectory
- Deploy on multiple drones (swarm)

**If policy needs improvement:**
- Collect real flight data
- Fine-tune with domain adaptation
- Train with observationnoise matching real sensors
- Use system identification to match sim dynamics

## Example Full Workflow

```bash
# 1. Train (already done)
D:\IsaacSim\python.bat scripts/reinforcement_learning/train_simpleflight.py \
    --headless --num_envs 2048 --trajectory hover --max_iterations 1000

# 2. Test connection
python -m cflib.crtp scan

# 3. Start mocap streaming (if applicable)
# ... your mocap code ...

# 4. Deploy for 10 seconds
python scripts/deployment/deploy_to_crazyflie.py \
    --checkpoint logs/rsl_rl/simpleflight_track/*/model_1000.pt \
    --uri radio://0/80/2M/E7E7E7E7E7 \
    --trajectory hover \
    --duration 10

# 5. Analyze results, iterate!
```

## Safety Reminder

🚨 **Always test policies in a safe environment first!**
- Use a cage/net
- Start with very short flights (5-10 seconds)
- Have a kill switch ready
- Never fly near people

## Support

For issues:
1. Check Crazyflie forums: https://forum.bitcraze.io/
2. Review cflib docs: https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/
3. Check SimpleFlight original repo for reference implementations
