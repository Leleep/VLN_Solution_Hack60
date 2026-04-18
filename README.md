# LM-Nav: Vision-Language Navigation in Gazebo Simulation

A Vision-Language Navigation (VLN) pipeline that uses CLIP + LLM to navigate a TurtleBot3 robot through a simulated house based on natural language instructions.

## Architecture

```
User Instruction → LLM Landmark Extractor → CLIP Scorer → Graph Search → Nav2 Execution
       ↓                    ↓                    ↓              ↓              ↓
 "Go to kitchen"     ["kitchen",          Score nodes     Find optimal    Drive robot
                      "refrigerator"]     with CLIP       path through    via Nav2
                                          similarity      topological     actions
                                                          graph
```

### Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| **LLM Extractor** | `lmnav/llm_extractor.py` | Extracts landmarks from natural language (Ollama/spaCy/OpenAI) |
| **CLIP Scorer** | `lmnav/clip_scorer.py` | Scores graph node images against landmark descriptions |
| **Graph Search** | `lmnav/graph_search.py` | Finds optimal path through topological graph |
| **Visualizer** | `lmnav/visualizer.py` | Generates walk visualization images |
| **House Explorer** | `scripts/explore_house.py` | Drives robot through house, captures node images |
| **Pipeline Runner** | `scripts/run_pipeline.py` | Runs the full VLN pipeline (offline) |
| **Walk Executor** | `scripts/execute_walk.py` | Drives the robot along the planned path |
| **Ego View** | `scripts/ego_view.py` | First-person camera view from the robot |

---

## Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 (native or via WSL2/Distrobox)
- **GPU**: NVIDIA GPU with driver support (tested on RTX 4050)
- **RAM**: 8GB+ recommended
- **Disk**: ~5GB for ROS2 + dependencies

### Software Dependencies
- **ROS2 Humble** (full desktop install)
- **Gazebo Classic 11** (comes with ROS2 Humble desktop)
- **Nav2** (ROS2 navigation stack)
- **TurtleBot3 packages**
- **Conda** (Miniconda/Anaconda)
- **Ollama** (optional, for LLM landmark extraction)

---

## Installation Guide

### Step 1: Install ROS2 Humble

> If using Distrobox/WSL2, run these inside the Ubuntu container.

```bash
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS2 repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble Desktop
sudo apt update
sudo apt install ros-humble-desktop -y
```

### Step 2: Install Nav2 & TurtleBot3 Packages

```bash
sudo apt install -y \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-description \
    ros-humble-turtlebot3-navigation2 \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-cv-bridge \
    python3-colcon-common-extensions
```

### Step 3: Create Conda Environment

```bash
conda create -n dl_env python=3.10 -y
conda activate dl_env

# Install Python dependencies
cd VLN_Solution_Hack60
pip install -r requirements.txt

# Download spaCy model (for NLP fallback)
python -m spacy download en_core_web_sm
```

### Step 4: Install Ollama (Optional — for LLM landmark extraction)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &        # Start in background
ollama pull llama3    # Download the model
```

> If you don't install Ollama, set `llm_backend: "spacy"` in `config/pipeline_config.yaml`.

### Step 5: Clone the AWS Small House World

> This should already be included in the repository under `aws-robomaker-small-house-world/`.

If missing:
```bash
cd VLN_Solution_Hack60
git clone https://github.com/aws-robotics/aws-robomaker-small-house-world.git
```

---

## Running the Pipeline

The pipeline has **3 phases**, each in a separate terminal.

### Phase 0: Pre-flight Check

Make sure you're inside the Ubuntu environment (Distrobox/WSL2) and activate the conda env:

```bash
# If using Distrobox:
distrobox enter ubuntu

# Activate conda & ROS2:
conda activate dl_env
source /opt/ros/humble/setup.bash

# Navigate to project:
cd /path/to/VLN_Solution_Hack60
```

### Phase 1: Launch Simulation (Terminal 1)

```bash
bash launch_sim.sh
```

This single command handles everything:
- ✅ Sets NVIDIA GPU rendering environment variables
- ✅ Sets `GAZEBO_PLUGIN_PATH` and `GAZEBO_MODEL_PATH`
- ✅ Sets `TURTLEBOT3_MODEL=waffle`
- ✅ Launches Gazebo + Nav2 + RViz
- ✅ Waits for Gazebo to be ready → spawns the robot (anti-topple model)
- ✅ Tunes Nav2 costmap for indoor navigation (reduced inflation)

**Wait** until you see:
```
🎉 Simulation is fully running!
   Robot: TurtleBot3 Waffle at (-2.0, -0.5)
   Costmap: tuned for indoor doorways
```

### Phase 2: Explore the House (Terminal 2)

```bash
conda activate dl_env
source /opt/ros/humble/setup.bash
cd /path/to/VLN_Solution_Hack60

python scripts/explore_house.py
```

This drives the robot through **55 predefined waypoints** covering:
- Hallway, Living Room, Kitchen, Bathroom, Bedroom, Fitness Room

At each waypoint, it captures a first-person image and records the pose. Output:
```
data/aws_house_graph/
├── node_000.png ... node_054.png    # First-person images
└── poses.json                       # Robot poses at each node
```

> ⏱️ Takes ~10-15 minutes for a full exploration.

### Phase 3: Run the VLN Pipeline (Terminal 2 — after exploration)

```bash
python scripts/run_pipeline.py -i "Go to the kitchen and find the refrigerator"
```

This runs **offline** (no robot movement):
1. Extracts landmarks from the instruction using LLM/spaCy
2. Scores all node images with CLIP against each landmark
3. Finds the optimal path through the topological graph
4. Saves the planned walk to `output/planned_walk.json`

### Phase 4: Execute the Walk (Terminal 3)

```bash
conda activate dl_env
source /opt/ros/humble/setup.bash
cd /path/to/VLN_Solution_Hack60

python scripts/execute_walk.py
```

This reads `output/planned_walk.json` and drives the robot along the planned path using Nav2.

---

## Environment Variables Reference

These are automatically set by `launch_sim.sh`, but documented here for manual use:

```bash
# ── ROS2 ──
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle

# ── Gazebo Plugin & Model Paths (CRITICAL) ──
export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:${GAZEBO_PLUGIN_PATH}
export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models:$(pwd)/aws-robomaker-small-house-world/models:${GAZEBO_MODEL_PATH}

# ── NVIDIA GPU Rendering (for laptop hybrid GPU setups) ──
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MESA_GL_VERSION_OVERRIDE=3.3
export OGRE_RTT_MODE=FBO
```

---

## Nav2 Costmap Tuning

The default Nav2 costmap parameters are too conservative for indoor navigation. `launch_sim.sh` automatically applies these tuned values:

| Parameter | Default | Tuned | Why |
|-----------|---------|-------|-----|
| `inflation_radius` | 0.55m | **0.15m** | Allows passage through doorways |
| `cost_scaling_factor` | 3.0 | **15.0** | Cost drops off faster from walls |
| `robot_radius` | 0.22m | **0.12m** | TurtleBot3 fits through tight spaces |

To adjust manually (live, without restart):
```bash
ros2 param set /local_costmap/local_costmap inflation_layer.inflation_radius 0.15
ros2 param set /global_costmap/global_costmap inflation_layer.inflation_radius 0.15
ros2 param set /local_costmap/local_costmap inflation_layer.cost_scaling_factor 15.0
ros2 param set /global_costmap/global_costmap inflation_layer.cost_scaling_factor 15.0
```

---

## Robot Model: Anti-Topple Mod

The file `waffle_stable.model` is a modified TurtleBot3 Waffle SDF with anti-topple physics:

| Property | Original | Modified | Effect |
|----------|----------|----------|--------|
| Mass | 1.0 kg | **20.0 kg** | Too heavy to push over |
| Center of mass (z) | 0.048m | **0.005m** | Very low center of gravity |
| Roll inertia (ixx) | 0.001 | **1.0** | 1000x resistance to roll |
| Pitch inertia (iyy) | 0.001 | **1.0** | 1000x resistance to pitch |

---

## Project Structure

```
VLN_Solution_Hack60/
├── launch_sim.sh                    # One-command simulation launcher
├── waffle_stable.model              # Anti-topple robot SDF
├── pipeline_config.yaml             # Quick config (backend selector)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── config/
│   └── pipeline_config.yaml         # Full pipeline configuration
│
├── lmnav/                           # Core pipeline modules
│   ├── __init__.py
│   ├── llm_extractor.py             # LLM landmark extraction
│   ├── clip_scorer.py               # CLIP image-text scoring
│   ├── graph_search.py              # Topological graph search
│   ├── visualizer.py                # Walk visualization
│   ├── adapter.py                   # Base environment adapter
│   ├── aws_house_adapter.py         # AWS house specific adapter
│   └── mp3d_adapter.py              # Matterport3D adapter
│
├── scripts/                         # Executable scripts
│   ├── explore_house.py             # Robot house exploration
│   ├── run_pipeline.py              # VLN pipeline (offline)
│   ├── execute_walk.py              # Execute planned walk
│   ├── ego_view.py                  # First-person camera view
│   └── generate_test_data.py        # Test data generator
│
├── aws-robomaker-small-house-world/ # Gazebo world + maps
│   ├── worlds/small_house.world
│   ├── models/                      # House furniture models
│   └── maps/turtlebot3_waffle_pi/
│       ├── map.yaml
│       └── map.pgm
│
├── data/
│   └── aws_house_graph/             # Generated by explore_house.py
│       ├── node_*.png               # First-person images
│       └── poses.json               # Node poses
│
└── output/                          # Generated by run_pipeline.py
    ├── planned_walk.json
    ├── execution_report.json
    └── walk_visualization.png
```

---

## Troubleshooting

### Robot not spawning in Gazebo
- **Cause**: Race condition — `spawn_entity` runs before Gazebo is ready.
- **Fix**: Use `launch_sim.sh` which handles this automatically.

### Robot topples over
- **Cause**: Original waffle.model has low mass/inertia.
- **Fix**: `launch_sim.sh` uses `waffle_stable.model` with anti-topple physics.

### Robot gets stuck at doorways
- **Cause**: Default Nav2 inflation radius (0.55m) is too large.
- **Fix**: `launch_sim.sh` auto-tunes to 0.15m. For live adjustment:
  ```bash
  ros2 param set /local_costmap/local_costmap inflation_layer.inflation_radius 0.15
  ros2 param set /global_costmap/global_costmap inflation_layer.inflation_radius 0.15
  ```

### Gazebo renders with Mesa (CPU) instead of NVIDIA GPU
- **Cause**: Hybrid GPU laptop not routing to NVIDIA.
- **Fix**: `launch_sim.sh` sets `__NV_PRIME_RENDER_OFFLOAD=1` and related vars.
- **Verify**: In Gazebo, check the rendering engine in the bottom status bar.

### `GAZEBO_PLUGIN_PATH` not set
- **Symptoms**: Robot spawns but doesn't publish `/odom`, `/scan`, etc.
- **Fix**: `launch_sim.sh` exports `GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib`.

### Ollama not running
- **Symptoms**: `run_pipeline.py` fails with connection error.
- **Fix**: Start Ollama: `ollama serve &`
- **Alternative**: Use spaCy backend: set `llm_backend: "spacy"` in `config/pipeline_config.yaml`.

---

## Quick Reference Commands

```bash
# === Kill everything ===
killall gzserver gzclient 2>/dev/null; pkill -f ros2

# === Delete and respawn robot (without restarting Gazebo) ===
ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity '{name: "turtlebot3_waffle"}'
sleep 2
ros2 run gazebo_ros spawn_entity.py -entity turtlebot3_waffle \
    -file $(pwd)/waffle_stable.model -x -2.0 -y -0.5 -z 0.01

# === Check robot status ===
ros2 topic info /odom                          # Should show 1 publisher
ros2 topic echo /odom --once | head -15        # Check position

# === List Gazebo models ===
ros2 service call /get_model_list gazebo_msgs/srv/GetModelList '{}'

# === Tune costmap live ===
ros2 param set /local_costmap/local_costmap inflation_layer.inflation_radius 0.15
ros2 param set /global_costmap/global_costmap inflation_layer.inflation_radius 0.15
```
