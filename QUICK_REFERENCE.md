# SimpleFlight Deployment Quick Reference

## One-Time Setup (Run in WSL2)

```bash
# Copy setup script to WSL2
cd ~
cp /mnt/d/coding/Capstone/SimpleFlight-experimenting/wsl2_setup.sh .
chmod +x wsl2_setup.sh

# Run automated setup
./wsl2_setup.sh
```

## Every Flight Session

### 1. Attach Crazyradio (Windows PowerShell Admin)

```powershell
usbipd attach --wsl --busid 2-4  # Replace 2-4 with your BUSID
```

### 2. Launch ROS 2 Server (WSL2 Terminal 1)

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch crazyflie launch.py backend:=cflib
```

### 3. Run Deployment (WSL2 Terminal 2)

```bash
cd ~/ros2_ws/src/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples

# Figure-8 trajectory
python rl_track.py

# OR arbitrary trajectories
python rl_arbitrary_track.py
```

## Transfer New Models from Windows

```bash
# From WSL2
cp /mnt/d/coding/Capstone/SimpleFlight-experimenting/logs/rsl_rl/simpleflight_track/<DATE>/model_999.pt ~/SimpleFlight/models/my_model.pt

# Update path in deployment script
nano ~/ros2_ws/src/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples/rl_track.py
# Line 68: ckpt_name = "~/SimpleFlight/models/my_model.pt"
```

## Emergency Stop

Press `Ctrl+C` in the deployment terminal (Terminal 2)

## Common Issues

**Crazyradio not found**:
```powershell
# Windows PowerShell Admin
usbipd list
usbipd attach --wsl --busid <BUSID>
```

**ROS 2 import errors**:
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
```

## Training → Deployment Workflow

1. **Train in Windows**: `D:\IsaacSim\python.bat train_simpleflight.py --max_iterations 10000`
2. **Copy to WSL2**: `cp /mnt/d/.../model_*.pt ~/SimpleFlight/models/`
3. **Deploy via ROS 2**: Follow flight session steps above
