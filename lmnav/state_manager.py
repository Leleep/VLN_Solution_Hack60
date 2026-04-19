"""
State Manager — Robot State Persistence Between Prompts
========================================================
Tracks where the robot is across multiple pipeline runs so that
Dijkstra doesn't always restart from node 0.

Priority order for determining the start node:
  1. CLI override (--start-node)
  2. Live AMCL pose (/amcl_pose topic)
  3. Saved state file (output/robot_state.json)
  4. Default: node 0

Why /amcl_pose and not /odom?
  /odom is in the odom frame which drifts over time.
  /amcl_pose is in the map frame — the same frame as all graph coordinates.
"""

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def get_robot_pose_amcl(timeout_sec: float = 3.0) -> Optional[Tuple[float, float, float]]:
    """
    Try to read the robot's current pose from /amcl_pose.

    Returns:
        (x, y, theta) in map frame, or None if ROS2 is not available
        or no message is received within the timeout.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseWithCovarianceStamped
        import math
    except ImportError:
        return None

    pose_result = [None]

    class _PoseReader(Node):
        def __init__(self):
            super().__init__("_state_manager_pose_reader")
            self._sub = self.create_subscription(
                PoseWithCovarianceStamped,
                "/amcl_pose",
                self._callback,
                10,
            )

        def _callback(self, msg):
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (ori.w * ori.z + ori.x * ori.y),
                1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z),
            )
            pose_result[0] = (pos.x, pos.y, yaw)

    # Try to initialize rclpy if not already initialized
    need_init = False
    if not rclpy.ok():
        try:
            rclpy.init()
            need_init = True
        except RuntimeError:
            pass  # Already initialized

    node = _PoseReader()
    deadline = time.time() + timeout_sec

    try:
        while time.time() < deadline and pose_result[0] is None:
            rclpy.spin_once(node, timeout_sec=0.1)
    except Exception:
        pass
    finally:
        node.destroy_node()
        if need_init:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    return pose_result[0]


def find_nearest_node(
    x: float, y: float, poses_data: list
) -> Tuple[int, float]:
    """
    Find the graph node nearest to (x, y) in the map frame.

    Args:
        x, y: World coordinates (map frame, meters)
        poses_data: List of dicts with 'id', 'x', 'y' keys

    Returns:
        (node_id, distance_m) — ID and Euclidean distance to the nearest node
    """
    best_id = poses_data[0]["id"]
    best_dist = float("inf")

    for p in poses_data:
        dist = np.sqrt((p["x"] - x) ** 2 + (p["y"] - y) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_id = p["id"]

    return best_id, best_dist


def load_state(state_path: str) -> Optional[dict]:
    """
    Load saved robot state from a JSON file.

    Returns:
        Dict with 'node_id', 'x', 'y', 'theta' or None if file doesn't exist.
    """
    state_path = Path(state_path)
    if not state_path.exists():
        return None

    try:
        with open(state_path, "r") as f:
            data = json.load(f)
        # Validate required fields
        if all(k in data for k in ("node_id", "x", "y", "theta")):
            return data
        return None
    except (json.JSONDecodeError, KeyError):
        return None


def save_state(
    state_path: str,
    node_id: int,
    x: float,
    y: float,
    theta: float,
    is_off_graph: bool = False,
    return_pose: dict = None,
) -> None:
    """
    Save the robot's current state after a walk completes.

    Args:
        state_path: Path to write robot_state.json
        node_id: Last successfully reached node ID
        x, y, theta: Actual robot pose (from AMCL, in map frame)
        is_off_graph: True if robot drove off-graph to approach an object
        return_pose: Dict {x, y, theta} of the graph node to return to
    """
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "node_id": node_id,
        "x": float(x),
        "y": float(y),
        "theta": float(theta),
        "timestamp": time.time(),
        "is_off_graph": is_off_graph,
        "return_pose": return_pose,
    }

    with open(state_path, "w") as f:
        json.dump(data, f, indent=2)

    off_str = " (OFF-GRAPH)" if is_off_graph else ""
    print(f"💾 Robot state saved: node={node_id}, pos=({x:.2f}, {y:.2f}){off_str}")


def determine_start_node(
    poses_data: list,
    state_path: str = "output/robot_state.json",
    override_node: Optional[int] = None,
    spawn_x: float = None,
    spawn_y: float = None,
) -> Tuple[int, str]:
    """
    Determine the best start node for the pipeline.

    Priority:
      1. CLI override (--start-node)
      2. Live AMCL pose → nearest node
      3. Saved state file → saved node
      4. Nearest node to configured spawn position
      5. Node 0 (absolute last resort)

    Args:
        poses_data: List of pose dicts from poses.json
        state_path: Path to robot_state.json
        override_node: If set, use this node (manual CLI override)
        spawn_x: Robot spawn X in map frame (from pipeline_config.yaml)
        spawn_y: Robot spawn Y in map frame (from pipeline_config.yaml)

    Returns:
        (node_id, reason_string) — the selected start node and why
    """
    # 1. Manual override
    if override_node is not None:
        valid_ids = {p["id"] for p in poses_data}
        if override_node in valid_ids:
            return override_node, f"manual override (--start-node {override_node})"
        else:
            print(
                f"⚠️  Override node {override_node} not found in graph. "
                f"Valid IDs: {sorted(valid_ids)[:10]}..."
            )

    # 2. Try live AMCL pose
    print("📡 Checking for live AMCL pose...")
    amcl_pose = get_robot_pose_amcl(timeout_sec=3.0)
    if amcl_pose is not None:
        x, y, theta = amcl_pose
        node_id, dist = find_nearest_node(x, y, poses_data)
        reason = (
            f"nearest to live AMCL pose ({x:.2f}, {y:.2f}), "
            f"distance={dist:.2f}m"
        )
        return node_id, reason

    # 3. Try saved state file
    print("📄 Checking for saved robot state...")
    state = load_state(state_path)
    if state is not None:
        node_id = state["node_id"]
        x, y = state["x"], state["y"]
        reason = f"saved state from previous walk (node {node_id} at ({x:.2f}, {y:.2f}))"
        return node_id, reason

    # 4. Use node nearest to configured spawn position
    if spawn_x is not None and spawn_y is not None:
        node_id, dist = find_nearest_node(spawn_x, spawn_y, poses_data)
        reason = (
            f"nearest to spawn ({spawn_x:.2f}, {spawn_y:.2f}), "
            f"distance={dist:.2f}m (no AMCL, no saved state)"
        )
        return node_id, reason

    # 5. Absolute last resort — node 0
    default_id = poses_data[0]["id"] if poses_data else 0
    return default_id, "node 0 fallback (no AMCL, no saved state, no spawn config)"
