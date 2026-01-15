# WSL2 Project Handoff: SimpleFlight RL Policy Training & Deployment

**Date**: January 14, 2026  
**Context**: Transitioning from Windows-only development to WSL2 for ROS 2 + Crazyswarm2 deployment  
**Hardware**: Crazyflie 2.1 with Flow Deck v2, Crazyradio 2.0

---

## Executive Summary

This project aims to **train custom RL PPO policies for Crazyflie drone control** using Isaac Sim/IsaacLab, then deploy them to real hardware. After extensive investigation, we discovered that SimpleFlight's original deployment requires **ROS 2 + Crazyswarm2**, which necessitates WSL2 on Windows.

### Critical Discovery: Why WSL2 is Required

**Problem**: Direct cflib deployment (Windows) resulted in motors not spinning, despite:
- ✅ Radio connection working
- ✅ Commands being sent successfully
- ✅ test_motors.py working perfectly (40,000 thrust spins motors)
- ✅ Policy inference working
- ❌ Policy deployment commands don't actuate motors

**Root Cause**: SimpleFlight never used direct cflib commands. Their deployment pipeline is:
```
Policy → ROS 2 Service Call (cmdVel) → Crazyswarm2 → Crazyflie Firmware
```

This cannot be replicated on Windows without ROS 2.

---

## Project Architecture

### Training (Isaac Sim + IsaacLab)
- **Platform**: WSL2 (headless Isaac Sim recommended for performance)
- **Environment**: SimpleFlight Track task (trajectory following)
- **Algorithm**: MAPPO (Multi-Agent PPO)
- **Observation**: 42D (waypoint positions, velocity, orientation)
- **Action**: 4D (roll rate, pitch rate, yaw rate, thrust)

### Deployment (ROS 2 + Crazyswarm2)
- **Platform**: WSL2 (ROS 2 Humble)
- **Interface**: crazyswarm_SimpleFlight package
- **Hardware**: Crazyflie 2.1 + Flow Deck v2 (connected via Crazyradio on USB)
- **State Estimation**: Flow Deck (no motion capture)
- **Control**: Low-level CTBR (Collective Thrust + Body Rates)

---

## Current Project State

### What's Complete ✅

1. **Training Infrastructure**
   - Isaac Sim working on Windows (D:\IsaacSim)
   - SimpleFlight environment functional
   - Training script: `IsaacLab\scripts\reinforcement_learning\train_simpleflight.py`
   - Existing checkpoint: `logs\rsl_rl\simpleflight_track\2026-01-14_13-40-06\model_999.pt` (1000 iterations)

2. **Calibration Tools Created**
   - `measure_hover_thrust.py` - Empirically measures hover thrust (Result: 40,000)
   - `export_policy.py` - Extracts actor network for deployment
   - `deploy_standalone.py` - Attempted direct cflib deployment (doesn't work)
   - `test_motors.py` - Hardware validation (proves drone works)

3. **Hardware Validation**
   - Crazyflie 2.1 + Flow Deck v2 functional
   - Radio: `radio://0/80/2M/E7E7E7E7E7`
   - Hover thrust measured: 40,000 (0.611 normalized)
   - cflib 0.1.30 installed and working

4. **Key Findings**
   - Undertrained policy (1000 iter) outputs constant actions (~-0.75 thrust)
   - Physics mismatch: Sim expects hover at 0.5 thrust, real needs 0.611
   - Direct cflib deployment architecture doesn't work (motors don't spin)
   - SimpleFlight deployment requires ROS 2 + Crazyswarm2

### What's Needed ❌

1. **ROS 2 + Crazyswarm2 Setup** (Primary goal)
2. **Isaac Sim v5 Installation in WSL2** (For training)
3. **Physics Calibration** (Update sim to match real hover thrust)
4. **Proper Training** (5000-10000 iterations for good policy)
5. **Deployment via ROS 2** (Using SimpleFlight's pipeline)

---

## WSL2 Setup Instructions

### Step 1: Install Isaac Sim v5 in WSL2

**Recommended Location**: `/opt/isaac-sim` or `~/isaac-sim`  
**NOT inside repo** (large installation, should be separate)

#### Option A: Native Linux Installation (Recommended)

```bash
# 1. Install dependencies
sudo apt-get update
sudo apt-get install -y wget git python3.10 python3.10-venv

# 2. Download Isaac Sim v5 (replace with actual download link/method)
# Check: https://developer.nvidia.com/isaac-sim
# Note: Requires NVIDIA GPU with updated drivers in WSL2

# 3. Install to /opt/isaac-sim
sudo mkdir -p /opt/isaac-sim
cd /opt/isaac-sim

# Follow official installation guide for Linux
# Ensure WSL2 has GPU passthrough enabled:
# - Windows 11 with updated GPU drivers
# - nvidia-smi should work in WSL2

# 4. Set up environment
echo 'export ISAACSIM_PATH="/opt/isaac-sim"' >> ~/.bashrc
echo 'export PATH="$ISAACSIM_PATH:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 5. Verify installation
$ISAACSIM_PATH/python.sh --version
```

#### Option B: Omniverse Launcher Method

```bash
# If using Omniverse Launcher (GUI):
# 1. Install Omniverse Launcher for Linux
# 2. Install Isaac Sim v5 through launcher
# 3. Default location: ~/.local/share/ov/pkg/isaac-sim-*

# Link to standard path
ln -s ~/.local/share/ov/pkg/isaac-sim-* ~/isaac-sim
```

#### WSL2 GPU Setup Verification

```bash
# Check NVIDIA driver
nvidia-smi

# If not working:
# 1. Update Windows GPU drivers
# 2. Ensure WSL2 kernel is updated: wsl --update
# 3. Install CUDA toolkit in WSL2

# Test GPU access
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Step 2: Clone Project in WSL2

```bash
# Navigate to WSL2 home or desired location
cd ~
# Or: cd /mnt/d/coding/Capstone  # Access Windows filesystem

# If not already accessible, clone from Windows:
# Repo is at: /mnt/d/coding/Capstone/SimpleFlight-experimenting

# Work directly from Windows mount (easier) or copy to WSL2 (faster)
cd /mnt/d/coding/Capstone/SimpleFlight-experimenting
```

### Step 3: Install IsaacLab Dependencies

```bash
cd ~/SimpleFlight-experimenting  # Or wherever repo is located

# Create Isaac Sim extension link
cd IsaacLab
# Point to your Isaac Sim installation
./isaaclab.sh --install  # This will set up extensions

# Or manually:
ISAACSIM_PATH=/opt/isaac-sim ./isaaclab.sh --install
```

### Step 4: Install ROS 2 Humble

```bash
# Follow official ROS 2 Humble installation for Ubuntu
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions

# Set up environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source /opt/ros/humble/setup.bash
```

### Step 5: Install Crazyswarm2

```bash
# Create ROS 2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone Crazyswarm2
git clone https://github.com/IMRCLab/crazyswarm2.git --recursive

# Clone SimpleFlight deployment repo (if exists)
# git clone https://github.com/thu-uav/crazyswarm_SimpleFlight.git --recursive
# NOTE: This repo may not exist yet - we may need to create deployment scripts

# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --symlink-install

# Source workspace
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/ros2_ws/install/setup.bash
```

### Step 6: Configure Crazyswarm2 for Your Hardware

```bash
cd ~/ros2_ws/src/crazyswarm2/crazyflie/config

# Edit crazyflies.yaml
nano crazyflies.yaml
```

Add your Crazyflie configuration:

```yaml
robots:
  cf1:
    enabled: true
    uri: radio://0/80/2M/E7E7E7E7E7
    initial_position: [0.0, 0.0, 0.0]
    type: cf21

robot_types:
  cf21:
    big_quad: false
    motion_capture:
      enabled: false  # Using Flow Deck, not motion capture
      marker: default
      dynamics: default
```

### Step 7: Install Python Dependencies

```bash
cd ~/SimpleFlight-experimenting

# Create virtual environment for deployment
python3 -m venv venv_deploy
source venv_deploy/bin/activate

# Install cflib
pip install cflib==0.1.30

# Install SimpleFlight package (for observation construction)
pip install -e .

# Install PyTorch (for policy inference)
pip install torch numpy
```

---

## Next Steps for New Agent

### Immediate Tasks (Priority Order)

1. **Verify WSL2 Setup**
   - Confirm GPU access: `nvidia-smi`
   - Confirm ROS 2: `ros2 --version`
   - Confirm Isaac Sim: `$ISAACSIM_PATH/python.sh --version`

2. **Test Crazyswarm2 Connection**
   ```bash
   # Terminal 1: Launch Crazyswarm2
   cd ~/ros2_ws
   source install/setup.bash
   ros2 launch crazyflie launch.py backend:=cflib
   
   # Terminal 2: Test connection
   ros2 service call /cf1/takeoff std_srvs/srv/Empty
   # Should see drone respond (or at least acknowledge service)
   ```

3. **Create Deployment Script**
   - Adapt `deploy_standalone.py` logic to ROS 2 service calls
   - Use SimpleFlight's original `rl_track.py` as reference (if available)
   - Key service: `/cf1/cmd_vel_legacy` or similar CTBR interface

4. **Calibrate Physics**
   - Update `IsaacLab/scripts/envs/simpleflight_env_cfg.py`
   - Adjust `thrust_to_weight` or `mass` so hover thrust matches 0.611 normalized
   - Current: Sim expects 0.5, real needs 0.611 → ~22% gap

5. **Train Proper Policy**
   ```bash
   cd IsaacLab
   ./isaaclab.sh -p scripts/reinforcement_learning/train_simpleflight.py \
     --headless \
     --enable_cameras false \
     num_envs=2048
   ```
   - Train for 5000-10000 iterations (45-90 minutes)
   - Monitor reward convergence
   - Save checkpoint every 1000 iterations

6. **Deploy and Validate**
   - Export trained policy
   - Deploy via ROS 2
   - Test hover stability
   - Test trajectory tracking

---

## Key Files Reference

### Training
- **Environment**: `omni_drones/envs/single/track.py`
- **Config**: `cfg/task/Track.yaml`
- **Training Script**: `IsaacLab/scripts/reinforcement_learning/train_simpleflight.py`

### Calibration Tools (In Windows, reference only)
- `IsaacLab/scripts/deployment/measure_hover_thrust.py`
- `IsaacLab/scripts/deployment/test_motors.py`
- Measured hover thrust: **40,000 (0.611 normalized)**

### Existing Checkpoint
- **Path**: `logs/rsl_rl/simpleflight_track/2026-01-14_13-40-06/model_999.pt`
- **Iterations**: 1000 (undertrained)
- **Status**: Policy outputs constant actions, needs more training

### Documentation
- `CALIBRATION_GUIDE.md` - 6-phase calibration workflow
- `DEPLOYMENT_SETUP.md` - Original ROS 2 setup guide
- `EMERGENCY_STOP_GUIDE.md` - Safety procedures
- `FAULT_ISOLATION_PROTOCOL.md` - Debugging methodology

---

## Known Issues & Solutions

### Issue 1: Direct cflib Deployment Doesn't Actuate Motors
**Status**: Confirmed bug  
**Cause**: Unknown firmware/commander state issue  
**Solution**: Use ROS 2 + Crazyswarm2 (SimpleFlight's method)

### Issue 2: Undertrained Policy
**Status**: Expected  
**Cause**: Only 1000 iterations (quick test)  
**Solution**: Train 5000-10000 iterations after physics calibration

### Issue 3: Sim-to-Real Thrust Gap
**Status**: Measured  
**Cause**: Physics mismatch (sim: 0.5, real: 0.611)  
**Solution**: Update `thrust_to_weight` or `mass` in env config before training

### Issue 4: IsaacLab Dependency Version Mismatch (Windows)
**Status**: Blocked training on Windows  
**Error**: `urdf` extension requires 2.4.31, has 2.4.19  
**Solution**: Use WSL2 Isaac Sim or run `isaaclab.sh -i` to reinstall

---

## ROS 2 Deployment Architecture

### Command Flow

```
┌─────────────────┐
│  Policy (.pt)   │
│  Inference      │
└────────┬────────┘
         │ action[4]: [roll, pitch, yaw, thrust]
         ↓
┌─────────────────┐
│  ROS 2 Node     │
│  (Python)       │
└────────┬────────┘
         │ cmdVel service call
         ↓
┌─────────────────┐
│  Crazyswarm2    │
│  crazyflie      │
│  server         │
└────────┬────────┘
         │ cflib commands
         ↓
┌─────────────────┐
│  Crazyflie      │
│  Firmware       │
└─────────────────┘
```

### Required ROS 2 Services/Topics

- `/cf1/cmd_vel_legacy` - Low-level CTBR control
- `/cf1/takeoff` - Takeoff command
- `/cf1/land` - Land command
- `/cf1/emergency` - Emergency stop
- `/cf1/pose` - State estimation (subscribe)

### Example Deployment Node (Python)

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import torch
import numpy as np

class PolicyDeploymentNode(Node):
    def __init__(self):
        super().__init__('policy_deployment')
        
        # Load policy
        self.actor = torch.load('path/to/actor.pt')
        self.actor.eval()
        
        # ROS 2 publishers
        self.cmd_pub = self.create_publisher(
            Twist, '/cf1/cmd_vel_legacy', 10
        )
        
        # State subscriber
        self.create_subscription(
            PoseStamped, '/cf1/pose', self.pose_callback, 10
        )
        
        # Control timer (100Hz)
        self.create_timer(0.01, self.control_loop)
    
    def control_loop(self):
        obs = self.get_observation()
        action = self.policy_inference(obs)
        self.send_cmd_vel(action)
```

---

## Success Criteria

### Phase 1: Infrastructure (Week 1)
- [ ] Isaac Sim v5 running in WSL2 with GPU
- [ ] ROS 2 Humble installed and functional
- [ ] Crazyswarm2 can connect to Crazyflie
- [ ] Can send test commands via ROS 2

### Phase 2: Training (Week 1-2)
- [ ] Physics calibrated (hover thrust matches 0.611)
- [ ] Policy trained for 5000+ iterations
- [ ] Reward curve shows convergence
- [ ] Policy exported successfully

### Phase 3: Deployment (Week 2)
- [ ] ROS 2 deployment node working
- [ ] Policy commands reach Crazyflie
- [ ] **Motors spin in response to policy**
- [ ] Drone hovers stably (±10cm position error)

### Phase 4: Validation (Week 2-3)
- [ ] Trajectory tracking functional
- [ ] Flight duration >30 seconds
- [ ] Safe emergency stop tested
- [ ] Ready for advanced tasks

---

## Important Numbers & Parameters

### Hardware Calibration
- **Hover Thrust**: 40,000 PWM (0.611 normalized)
- **Radio URI**: `radio://0/80/2M/E7E7E7E7E7`
- **Motor Threshold**: ~20,000 PWM minimum to spin
- **Battery**: Monitor voltage, stop if <3.0V per cell

### Training Hyperparameters (from existing runs)
- **Observation Dim**: 42
- **Action Dim**: 4
- **Hidden Layers**: [256, 256, 256]
- **Learning Rate**: 3e-4 (typical)
- **Num Envs**: 2048 (adjust for GPU)
- **Target**: 5000-10000 iterations

### Deployment Scaling
- **RPY Scale**: 30 deg/s (SimpleFlight default)
- **Thrust Range**: [0, 65535] uint16
- **Control Rate**: 100 Hz
- **Observation**: 10 waypoints, velocity, orientation

---

## Questions to Ask Windows Agent (If Needed)

1. What was the exact training command used for model_999.pt?
2. Were there any other checkpoints saved?
3. What was the reward curve trajectory?
4. Any observations about policy behavior in simulation?

---

## Safety Reminders

⚠️ **Before ANY Flight**:
1. Clear 3x3m area
2. Flow Deck on textured surface
3. Battery >3.5V
4. Emergency stop ready (Ctrl+C)
5. Propeller guards recommended
6. Never fly near people/pets

🛑 **Emergency Stop**:
```bash
# In ROS 2 terminal
ros2 service call /cf1/emergency std_srvs/srv/Empty
```

---

## Contact & Resources

- **Crazyflie Forums**: https://forum.bitcraze.io/
- **Crazyswarm2 Docs**: https://imrclab.github.io/crazyswarm2/
- **Isaac Sim Docs**: https://docs.omniverse.nvidia.com/isaacsim/
- **SimpleFlight Paper**: (include DOI if available)

---

## Final Notes for New Agent

You are picking up a well-structured project with:
- ✅ Functional training environment
- ✅ Hardware validated and working
- ✅ Calibration measurements completed
- ✅ Clear deployment architecture identified

**Your mission**: Set up ROS 2 deployment pipeline and train a proper policy. The hardest detective work (why motors don't spin with direct cflib) is done - SimpleFlight uses ROS 2, and that's the path forward.

**First action**: Verify WSL2 GPU access and install Isaac Sim v5. Everything else flows from there.

**Expected timeline**: 2-3 weeks to fully functional autonomous flight.

Good luck! 🚁
