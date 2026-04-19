#!/bin/bash
# ============================================================
# Emergency TurtleBot3 Spawner
# ============================================================
# Run this script if the robot didn't spawn in Gazebo automatically.
# Usage:
#   bash emergency_spawn.sh
# ============================================================

source /opt/ros/humble/setup.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_SDF="${SCRIPT_DIR}/waffle_stable.model"

echo "🤖 Force spawning TurtleBot3 waffle at (-2.0, -0.5)..."

# Delete any existing ghost entity first just in case
ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "{name: 'turtlebot3_waffle'}" 2>/dev/null || true

# Spawn the robot
ros2 run gazebo_ros spawn_entity.py \
    -entity turtlebot3_waffle \
    -file ${ROBOT_SDF} \
    -x -2.0 -y -0.5 -z 0.01

echo "✅ Spawn command sent. Check Gazebo window!"
