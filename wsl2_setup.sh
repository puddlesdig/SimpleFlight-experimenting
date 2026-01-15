#!/bin/bash
# Quick setup script for SimpleFlight ROS 2 deployment in WSL2
# Run: bash wsl2_setup.sh

set -e  # Exit on error

echo "================================================"
echo "SimpleFlight ROS 2 Deployment Setup for WSL2"
echo "================================================"
echo ""

# Check if running in WSL2
if ! grep -qi microsoft /proc/version; then
    echo "❌ Error: This script must be run in WSL2"
    exit 1
fi

echo "✓ Running in WSL2"
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install ROS 2 Jazzy
echo ""
echo "🤖 Installing ROS 2 Jazzy..."
if [ ! -d "/opt/ros/jazzy" ]; then
    sudo apt install software-properties-common -y
    sudo add-apt-repository universe -y
    sudo apt update && sudo apt install curl -y
    
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    
    sudo apt update
    sudo apt install ros-jazzy-desktop python3-argcomplete python3-colcon-common-extensions python3-rosdep -y
    
    # Initialize rosdep
    if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
        sudo rosdep init
    fi
    rosdep update
    
    # Add to bashrc
    if ! grep -q "source /opt/ros/jazzy/setup.bash" ~/.bashrc; then
        echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
    fi
    
    echo "✓ ROS 2 Jazzy installed"
else
    echo "✓ ROS 2 Jazzy already installed"
fi

source /opt/ros/jazzy/setup.bash

# Create ROS 2 workspace
echo ""
echo "📁 Setting up ROS 2 workspace..."
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone crazyswarm_SimpleFlight
if [ ! -d "crazyswarm_SimpleFlight" ]; then
    echo "📥 Cloning crazyswarm_SimpleFlight..."
    git clone https://github.com/thu-uav/crazyswarm_SimpleFlight.git --recursive
    echo "✓ Repository cloned"
else
    echo "✓ crazyswarm_SimpleFlight already cloned"
fi

# Install dependencies
echo ""
echo "📦 Installing ROS dependencies..."
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
echo ""
echo "🔨 Building workspace (this may take a few minutes)..."
colcon build --symlink-install

# Source workspace
if ! grep -q "source ~/ros2_ws/install/setup.bash" ~/.bashrc; then
    echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
fi
source ~/ros2_ws/install/setup.bash

echo "✓ Workspace built successfully"

# Clone SimpleFlight (deployment branch)
echo ""
echo "📥 Setting up SimpleFlight..."
if [ ! -d ~/SimpleFlight ]; then
    cd ~/
    git clone https://github.com/thu-uav/SimpleFlight.git -b deployment
    cd SimpleFlight
    pip install -e .
    echo "✓ SimpleFlight installed"
else
    echo "✓ SimpleFlight already installed"
fi

# Create models directory
mkdir -p ~/SimpleFlight/models

# Create deployment config
echo ""
echo "⚙️  Creating deployment configuration..."
mkdir -p ~/SimpleFlight/config
cat > ~/SimpleFlight/config/deploy.yaml << 'EOF'
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

echo "✓ Configuration created"

# Check for Crazyradio
echo ""
echo "🔍 Checking for Crazyradio..."
if lsusb | grep -qi "1915:7777"; then
    echo "✓ Crazyradio detected!"
else
    echo "⚠️  Crazyradio NOT detected"
    echo ""
    echo "To attach Crazyradio from Windows PowerShell (Admin):"
    echo "  1. winget install --interactive --exact dorssel.usbipd-win"
    echo "  2. usbipd list"
    echo "  3. usbipd bind --busid <BUSID>"
    echo "  4. usbipd attach --wsl --busid <BUSID>"
fi

# Copy pretrained model if available
echo ""
echo "📋 Checking for pretrained models..."
if [ -f "/mnt/d/coding/Capstone/SimpleFlight-experimenting/models/deploy.pt" ]; then
    cp /mnt/d/coding/Capstone/SimpleFlight-experimenting/models/deploy.pt ~/SimpleFlight/models/
    echo "✓ Pretrained model copied to ~/SimpleFlight/models/deploy.pt"
fi

# Summary
echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Attach Crazyradio (Windows PowerShell Admin):"
echo "   usbipd attach --wsl --busid <BUSID>"
echo ""
echo "2. Update Crazyflie URI:"
echo "   nano ~/ros2_ws/src/crazyswarm_SimpleFlight/crazyflie/config/crazyflies.yaml"
echo ""
echo "3. Launch ROS 2 server (Terminal 1):"
echo "   ros2 launch crazyflie launch.py backend:=cflib"
echo ""
echo "4. Run deployment (Terminal 2):"
echo "   cd ~/ros2_ws/src/crazyswarm_SimpleFlight/crazyflie_examples/crazyflie_examples"
echo "   python rl_track.py"
echo ""
echo "📖 Full guide: /mnt/d/coding/Capstone/SimpleFlight-experimenting/DEPLOYMENT_SETUP.md"
echo ""
