#!/bin/bash
# ============================================================
# LM-Nav Gazebo Simulation Launcher
# ============================================================
# This script launches the full simulation stack:
#   1. Sets all required environment variables
#   2. Launches Gazebo + Nav2 (without robot spawn)
#   3. Waits for Gazebo to be ready
#   4. Spawns the TurtleBot3 waffle robot
#
# Usage:
#   cd /home/anurag/coding/dl_hackathon/VLN_Solution_Hack60
#   bash launch_sim.sh
# ============================================================

set -e

# ── Ensure ROS2 is sourced ──
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/humble/setup.bash
fi

# ── Force NVIDIA GPU rendering (RTX 4050) ──
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MESA_GL_VERSION_OVERRIDE=3.3
export OGRE_RTT_MODE=FBO

# ── TurtleBot3 + Gazebo paths (CRITICAL) ──
export TURTLEBOT3_MODEL=waffle
export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:${GAZEBO_PLUGIN_PATH}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models:${SCRIPT_DIR}/aws-robomaker-small-house-world/models:${GAZEBO_MODEL_PATH}

WORLD_FILE="${SCRIPT_DIR}/aws-robomaker-small-house-world/worlds/small_house.world"
MAP_FILE="${SCRIPT_DIR}/aws-robomaker-small-house-world/maps/turtlebot3_waffle_pi/map.yaml"
ROBOT_SDF="${SCRIPT_DIR}/waffle_stable.model"

echo "============================================================"
echo "🚀 LM-Nav Gazebo Simulation Launcher"
echo "============================================================"
echo "  World:  ${WORLD_FILE}"
echo "  Map:    ${MAP_FILE}"
echo "  Robot:  ${ROBOT_SDF}"
echo "  GPU:    NVIDIA (forced)"
echo "============================================================"

# ── Kill any leftover processes ──
echo "🧹 Cleaning up old processes..."
killall -q gzserver gzclient 2>/dev/null || true
sleep 1

# ── Launch Nav2 + Gazebo in background ──
echo "🌍 Launching Gazebo + Nav2..."
ros2 launch nav2_bringup tb3_simulation_launch.py \
    headless:=False \
    use_rviz:=True \
    world:=${WORLD_FILE} \
    map:=${MAP_FILE} &

LAUNCH_PID=$!

# ── Wait for Gazebo spawn_entity service to be available ──
echo "⏳ Waiting for Gazebo to be fully ready..."
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ros2 service list 2>/dev/null | grep -q "/spawn_entity"; then
        echo "✅ Gazebo is ready! (waited ${WAITED}s)"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo "   ...waiting (${WAITED}s / ${MAX_WAIT}s)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "❌ Gazebo did not start in time!"
    kill $LAUNCH_PID 2>/dev/null
    exit 1
fi

# ── Extra wait for Gazebo to fully initialize ──
echo "⏳ Giving Gazebo 5s to finish loading models..."
sleep 5

# ── Check if robot was already spawned by the launch file ──
MODELS=$(ros2 service call /get_model_list gazebo_msgs/srv/GetModelList '{}' 2>/dev/null)
if echo "$MODELS" | grep -q "turtlebot3_waffle"; then
    echo "✅ TurtleBot3 already spawned by launch file!"
else
    echo "🤖 Spawning TurtleBot3 waffle at (-2.0, -0.5)..."
    for try in 1 2 3; do
        ros2 run gazebo_ros spawn_entity.py \
            -entity turtlebot3_waffle \
            -file ${ROBOT_SDF} \
            -x -2.0 -y -0.5 -z 0.01
        
        sleep 3
        # Verify it actually spawned
        CHECK=$(ros2 service call /get_model_list gazebo_msgs/srv/GetModelList '{}' 2>/dev/null)
        if echo "$CHECK" | grep -q "turtlebot3_waffle"; then
            echo "✅ TurtleBot3 spawned successfully on attempt ${try}!"
            break
        else
            echo "⚠️  Spawn attempt ${try} failed, retrying..."
        fi
    done
fi

# ── Ensure emergency spawn is executable ──
chmod +x emergency_spawn.sh 2>/dev/null || true

sleep 2
ODOM_PUB=$(ros2 topic info /odom 2>/dev/null | grep "Publisher count" | awk '{print $3}')
if [ "$ODOM_PUB" -ge 1 ] 2>/dev/null; then
    echo "✅ /odom publishing!"
else
    echo "⚠️  Robot may not be publishing odom. Check Gazebo window."
fi

# ── Tune Nav2 costmap for indoor navigation ──
# Default inflation_radius=0.55m blocks narrow doorways in the AWS house.
# Reduce it so the robot can pass through doorways and tight spaces.
echo "🔧 Tuning Nav2 costmap for indoor navigation..."
ros2 param set /local_costmap/local_costmap inflation_layer.inflation_radius 0.15 2>/dev/null || true
ros2 param set /global_costmap/global_costmap inflation_layer.inflation_radius 0.15 2>/dev/null || true
ros2 param set /local_costmap/local_costmap inflation_layer.cost_scaling_factor 15.0 2>/dev/null || true
ros2 param set /global_costmap/global_costmap inflation_layer.cost_scaling_factor 15.0 2>/dev/null || true
ros2 param set /local_costmap/local_costmap robot_radius 0.12 2>/dev/null || true
ros2 param set /global_costmap/global_costmap robot_radius 0.12 2>/dev/null || true
echo "✅ Costmap tuned: inflation=0.15m, scaling=15.0, radius=0.12m"

echo ""
echo "============================================================"
echo "🎉 Simulation is fully running!"
echo "   Robot: TurtleBot3 Waffle at (-2.0, -0.5)"
echo "   Costmap: tuned for indoor doorways"
echo ""
echo "   Next steps:"
echo "   Terminal 2: python scripts/run_pipeline.py -i \"Go to the kitchen...\""
echo "   Terminal 3: python scripts/execute_walk.py"
echo "============================================================"

# ── Keep running (forward the launch process) ──
wait $LAUNCH_PID
