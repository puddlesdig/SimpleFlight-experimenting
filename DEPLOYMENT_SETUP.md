# SimpleFlight ROS 2 Deployment Setup (WSL2)

This guide sets up SimpleFlight's official ROS 2 deployment pipeline on WSL2 to deploy trained policies to real Crazyflie 2.1 hardware.

## Prerequisites

- ✅ WSL2 installed and running Ubuntu 24.04
- ✅ Crazyradio 2.0 PA dongle
- ✅ Crazyflie 2.1 with Flow Deck v2
- ✅ Trained policy checkpoint (`deploy.pt` or your custom model)

## Step 1: Install ROS 2 Jazzy in WSL2

Open WSL2 terminal and run:

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install ROS 2 Jazzy
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-jazzy-desktop python3-argcomplete -y

# Install additional dependencies
sudo apt install python3-colcon-common-extensions python3-rosdep -y

# Initialize rosdep
sudo rosdep init
rosdep update

# Add to ~/.bashrc for automatic sourcing
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Step 2: Clone and Build Crazyswarm_SimpleFlight

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone SimpleFlight deployment repo
git clone https://github.com/thu-uav/crazyswarm_SimpleFlight.git --recursive

# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --symlink-install

# Source the workspace
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/ros2_ws/install/setup.bash
```

## Step 3: Install SimpleFlight (Deployment Branch)

```bash
# Clone SimpleFlight training repo (deployment branch)
cd ~/
git clone https://github.com/thu-uav/SimpleFlight.git -b deployment
cd SimpleFlight

# Install SimpleFlight package
pip install -e .
```

## Step 4: Setup USB Passthrough for Crazyradio

WSL2 needs USB passthrough to access the Crazyradio. On **Windows PowerShell (Admin)**:

```powershell
# Install usbipd-win
winget install --interactive --exact dorssel.usbipd-win

# List USB devices
usbipd list

# Find Crazyradio (should show "Bitcraze AB Crazyradio PA USB dongle")
# Note the BUSID (e.g., 2-4)

# Bind the device
usbipd bind --busid 2-4

# Attach to WSL2
usbipd attach --wsl --busid 2-4
```

In **WSL2**, verify the device:

```bash
lsusb | grep Bitcraze
# Should show: Bus 001 Device 00X: ID 1915:7777 Nordic Semiconductor ASA
```

**Note**: After WSL2 restarts, you'll need to re-attach:
```powershell
usbipd attach --wsl --busid 2-4
```

## Step 5: Configure Crazyflie Connection

Edit the Crazyflie configuration:

```bash
cd ~/ros2_ws/src/crazyswarm_SimpleFlight/crazyflie/config
nano crazyflies.yaml
```

Update with your Crazyflie's URI:

```yaml
robots:
  cf1:
    enabled: true
    uri: radio://0/80/2M/E7E7E7E7E7  # Your Crazyflie's URI
    initial_position: [0, 0, 0]
    type: cf21

robot_types:
  cf21:
    big_quad: false
    motion_capture:
      enabled: false  # Set to true if using motion capture
      marker: default
      dynamics: default
```

## Step 6: Copy Trained Model to WSL2

From **Windows**, copy your trained model:

```powershell
# Option 1: Copy pretrained SimpleFlight model
wsl cp /mnt/d/coding/Capstone/SimpleFlight-experimenting/models/deploy.pt ~/SimpleFlight/models/

# Option 2: Copy your custom trained model
wsl cp /mnt/d/coding/Capstone/SimpleFlight-experimenting/logs/rsl_rl/simpleflight_track/2026-01-14_13-40-06/model_999.pt ~/SimpleFlight/models/my_model.pt
```

Or from **WSL2**:

```bash
mkdir -p ~/SimpleFlight/models
cp /mnt/d/coding/Capstone/SimpleFlight-experimenting/models/deploy.pt ~/SimpleFlight/models/
```

## Step 7: Create Deployment Configuration

Create a deploy config at `~/SimpleFlight/config/deploy.yaml`:

```bash
cd ~/SimpleFlight
mkdir -p config
cat > config/deploy.yaml << 'EOF'
seed: 42
algo:
  name: mappo
  network:
    rnn: false
    hidden_units: [256, 256, 256]
    activation: elu
  ppo:
    num_epochs: 8
    num_minibatches: 32
    clip_param: 0.2
    entropy_coef: 0.0
    value_loss_coef: 1.0
    max_grad_norm: 1.0
    gae_lambda: 0.95

env:
  num_envs: 1
  device: cpu
EOF
```

## Step 8: Test Connection

In **WSL2 Terminal 1**, launch the Crazyflie server:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch crazyflie launch.py backend:=cflib
```

Expected output:
```
[INFO] [crazyflie_server]: Connected to radio://0/80/2M/E7E7E7E7E7
```

Press `Ctrl+C` to stop after verifying connection.

## Step 9: Deploy Policy

In **WSL2 Terminal 1**, launch the server:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch crazyflie launch.py backend:=cflib
```

In **WSL2 Terminal 2**, run the deployment script:

```bash
cd ~/ros2_ws/src/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples

# For figure-eight trajectory tracking
python rl_track.py

# OR for arbitrary trajectories (polynomial, pentagram, zigzag)
python rl_arbitrary_track.py
```

## Safety Notes

⚠️ **Before First Flight**:
1. Clear 3x3m flight area
2. Have Flow Deck on textured surface
3. Be ready to press `Ctrl+C` for emergency stop
4. Keep away from walls/obstacles
5. Use low battery alert (motors will stop if battery < 3.0V)

## Troubleshooting

### Crazyradio not detected in WSL2
```bash
# Check USB devices
lsusb

# If not listed, re-attach from Windows PowerShell:
usbipd attach --wsl --busid 2-4
```

### ROS 2 connection timeout
```bash
# Check if server is running
ros2 node list

# Check Crazyflie URI in config
cat ~/ros2_ws/src/crazyswarm_SimpleFlight/crazyflie/config/crazyflies.yaml
```

### Import errors
```bash
# Ensure SimpleFlight is installed
pip install -e ~/SimpleFlight

# Source workspace
source ~/ros2_ws/install/setup.bash
```

### Policy loading fails
```bash
# Verify model exists
ls -lh ~/SimpleFlight/models/deploy.pt

# Check model path in deployment script
# Edit: ~/ros2_ws/src/crazyswarm_SimpleFlight/crazyflife_examples/crazyflie_examples/rl_track.py
# Line 68: ckpt_name = "model/deploy.pt"
```

## Deployment Workflow Summary

1. **Attach Crazyradio** (Windows PowerShell): `usbipd attach --wsl --busid 2-4`
2. **Terminal 1** (WSL2): `ros2 launch crazyflie launch.py backend:=cflib`
3. **Terminal 2** (WSL2): `python rl_track.py` or `python rl_arbitrary_track.py`
4. **Emergency Stop**: Press `Ctrl+C` in Terminal 2

## Next Steps

- Train longer policies (5000-10000 iterations) in IsaacLab on Windows
- Copy checkpoints to WSL2 for deployment testing
- Experiment with different trajectories
- Use motion capture system for precise tracking (optional)

## File Locations

- **Windows**: `D:\coding\Capstone\SimpleFlight-experimenting\`
  - Training: `IsaacLab/scripts/reinforcement_learning/train_simpleflight.py`
  - Models: `logs/rsl_rl/simpleflight_track/*/model_*.pt`

- **WSL2**: `~/`
  - Deployment: `ros2_ws/src/crazyswarm_SimpleFlight/crazyflie_examples/`
  - Models: `SimpleFlight/models/deploy.pt`
  - Config: `SimpleFlight/config/deploy.yaml`
