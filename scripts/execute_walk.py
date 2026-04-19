#!/usr/bin/env python3
"""
Nav2 Walk Executor
==================
Executes a planned walk in Gazebo by sending sequential Nav2 goals.

Reads the planned walk JSON from the pipeline output and drives
the TurtleBot3 to each waypoint using Nav2 NavigateToPose.

This replaces ViNG (the Visual Navigation Model) from the LM-Nav paper.
The "brains" (CLIP + LLM + Dijkstra) do the planning; Nav2 does the "legs".

Usage:
  # 1. Ensure Gazebo + Nav2 are running
  # 2. Run the pipeline first to generate the walk:
  python scripts/run_pipeline.py -i "Go to the kitchen..."
  # 3. Execute the walk:
  python scripts/execute_walk.py
  # or with a specific walk file:
  python scripts/execute_walk.py --walk-file output/planned_walk.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
    from geometry_msgs.msg import PoseStamped
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("⚠️  ROS2 not available. This script requires ROS2 Humble.")
    print("   Source your ROS2 workspace: source /opt/ros/humble/setup.bash")


class WalkExecutor(Node):
    """
    ROS2 node that executes a planned walk via Nav2.
    
    Adapted from vln_agent.py:VLNNavigator — same Nav2 action client pattern.
    """

    def __init__(self):
        super().__init__("walk_executor")
        self._action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        # NavigateThroughPoses for smooth batch transit (no spinning/wobbling)
        self._through_poses_client = ActionClient(
            self, NavigateThroughPoses, "navigate_through_poses"
        )
        
        # Odom subscriber for reading robot's current position
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self._latest_odom = None
        self._latest_amcl = None
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )
        # AMCL gives the map-frame pose (not drifting like odom)
        self._amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_callback, 10
        )
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )

        # Load transit edges (pre-computed obstacle-free paths between node pairs)
        self._transit_edges = {}
        transit_path = project_root / "data" / "aws_house_graph" / "transit_edges.json"
        if transit_path.exists():
            with open(transit_path) as f:
                td = json.load(f)
            self._transit_edges = td.get("edges", {})
            self._blocked_edges = set(td.get("blocked", []))
            n_blocked = len(self._blocked_edges)
            n_transit = sum(len(v) for v in self._transit_edges.values())
            self.get_logger().info(
                f"🗃️  Transit graph loaded: {len(self._transit_edges)} edges, "
                f"{n_transit} transit waypoints, {n_blocked} blocked"
            )
        else:
            self._blocked_edges = set()
            self.get_logger().warn(
                f"⚠️  No transit_edges.json found at {transit_path}. "
                f"Run 'python scripts/build_nav_graph.py' for obstacle-free navigation."
            )

        self.get_logger().info("🤖 Walk Executor initialized.")

    def _odom_callback(self, msg):
        self._latest_odom = msg

    def _amcl_callback(self, msg):
        self._latest_amcl = msg

    def _check_tf_health(self, timeout_sec: float = 8.0) -> bool:
        """
        Fast-fail TF health check: verify 'odom -> base_link' exists within
        timeout_sec seconds. If missing, Gazebo's diff-drive plugin has crashed;
        AMCL will NEVER localize without odom, so we exit immediately with
        clear instructions instead of hanging for 3 minutes.
        """
        try:
            from tf2_ros import Buffer, TransformListener
            import rclpy.duration
        except ImportError:
            return True  # tf2_ros not available, proceed optimistically

        tf_buffer   = Buffer()
        _tf_listener = TransformListener(tf_buffer, self)  # noqa: F841

        self.get_logger().info("🔎 Checking TF tree health (odom → base_link)...")
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            try:
                tf_buffer.lookup_transform(
                    "odom", "base_link",
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.3),
                )
                self.get_logger().info("   ✅ TF: odom → base_link OK")
                return True
            except Exception:
                pass  # keep waiting

        # Diagnose which key frames are absent
        missing = []
        for frame in ("odom", "base_link", "base_footprint", "base_scan"):
            try:
                tf_buffer.lookup_transform(frame, frame, rclpy.time.Time())
            except Exception:
                missing.append(frame)

        self.get_logger().error(
            f"\n{'='*58}\n"
            f"❌  TF HEALTH CHECK FAILED — missing frames: {missing}\n"
            f"{'='*58}\n"
            f"ROOT CAUSE: Gazebo's differential-drive / robot_state_publisher\n"
            f"  plugin stopped publishing transforms.\n\n"
            f"  This commonly happens when:\n"
            f"    • Two navigation sessions are run back-to-back without\n"
            f"      letting the simulation reset between them.\n"
            f"    • A previous execute_walk.py was killed mid-navigation.\n"
            f"    • Gazebo ran for too long and the heartbeat drifted.\n\n"
            f"  ── FIX ──────────────────────────────────────────────────\n"
            f"  Step 1 – kill all stale ROS / Gazebo processes:\n"
            f"    pkill -9 -f 'gzserver|gzclient|gazebo'\n"
            f"    pkill -9 -f 'rviz2|nav2|amcl|robot_state'\n"
            f"  Step 2 – wait ~5 s, then relaunch (Term 1):\n"
            f"    bash launch_sim.sh\n"
            f"  Step 3 – once Nav2 is ready, re-run (Term 3):\n"
            f"    python scripts/execute_walk.py\n"
            f"{'='*58}"
        )
        return False

    def wait_for_nav2(self, timeout_sec: float = 120.0) -> bool:
        """Wait for TF + AMCL to be healthy, then confirm Nav2 accepts goals."""
        self.get_logger().info("⏳ Waiting for Nav2 action server...")
        if not self._action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("❌ Nav2 action server not available!")
            return False

        # ── 0. TF health check — fail fast if Gazebo is broken ───────────────
        if not self._check_tf_health(timeout_sec=8.0):
            return False

        # ── 1. Get robot position for the initial-pose hint ──────────────────
        self.get_logger().info("📡 Waiting for odometry...")
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_odom is not None:
                break

        if self._latest_odom:
            pos = self._latest_odom.pose.pose.position
            ori = self._latest_odom.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (ori.w * ori.z + ori.x * ori.y),
                1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z),
            )
            x, y = pos.x, pos.y
        elif self._latest_amcl is not None:
            # Re-use a pose from a previous AMCL callback in this session
            pos = self._latest_amcl.pose.pose.position
            ori = self._latest_amcl.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (ori.w * ori.z + ori.x * ori.y),
                1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z),
            )
            x, y = pos.x, pos.y
            self.get_logger().info("   ℹ️  No fresh odom — using last AMCL pose")
        else:
            x, y, yaw = -2.0, -0.5, 0.0
            self.get_logger().warn("   ⚠️  No odom/AMCL data — using spawn defaults")

        self.get_logger().info(f"📍 Robot position: ({x:.2f}, {y:.2f}, θ={yaw:.2f})")

        # ── 2. Publish initial pose and wait for AMCL to accept goals ─────────
        from geometry_msgs.msg import PoseWithCovarianceStamped
        max_attempts = 15   # was 30 — fail faster if something is truly broken
        for attempt in range(max_attempts):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
            msg.pose.covariance[0]  = 0.25
            msg.pose.covariance[7]  = 0.25
            msg.pose.covariance[35] = 0.06
            for _ in range(5):
                self._initial_pose_pub.publish(msg)
                time.sleep(0.1)

            for _ in range(20):
                rclpy.spin_once(self, timeout_sec=0.1)

            test_goal = NavigateToPose.Goal()
            test_goal.pose.header.frame_id = "map"
            test_goal.pose.header.stamp = self.get_clock().now().to_msg()
            test_goal.pose.pose.position.x = x
            test_goal.pose.pose.position.y = y
            test_goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
            test_goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

            send_future = self._action_client.send_goal_async(test_goal)
            rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)

            if send_future.done() and send_future.result() is not None:
                goal_handle = send_future.result()
                if goal_handle.accepted:
                    self.get_logger().info(f"✅ AMCL localized! (attempt {attempt+1})")
                    cancel_future = goal_handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=3.0)
                    for _ in range(30):
                        rclpy.spin_once(self, timeout_sec=0.1)
                    return True

            self.get_logger().info(
                f"   ⏳ AMCL not ready (attempt {attempt+1}/{max_attempts})..."
            )

        self.get_logger().error(
            f"❌ AMCL did not localize after {max_attempts} attempts.\n"
            f"   TF tree was healthy, so AMCL may just need more time.\n"
            f"   Tip: run   ros2 run tf2_tools view_frames   to inspect."
        )
        return False

    def _make_pose_stamped(self, x: float, y: float, theta: float) -> PoseStamped:
        """Create a PoseStamped message from x, y, theta."""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(theta / 2.0)
        pose.pose.orientation.w = math.cos(theta / 2.0)
        return pose

    def navigate_to_pose(self, x: float, y: float, theta: float = 0.0,
                         tolerance_override: float = None) -> bool:
        """
        Send a single Nav2 goal and wait for completion.

        Args:
            x: Target x position (meters, in 'map' frame)
            y: Target y position (meters, in 'map' frame)
            theta: Target heading (radians)
            tolerance_override: If set, use this xy goal tolerance (m) instead
                                of the default. Transit waypoints use 0.4m for
                                smooth flow-through navigation.

        Returns:
            True if goal reached, False otherwise
        """
        self.get_logger().info("⏳ Waiting for Nav2 Action Server...")
        self._action_client.wait_for_server()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._make_pose_stamped(x, y, theta)

        tol_str = f" (tol={tolerance_override:.1f}m)" if tolerance_override else ""
        self.get_logger().info(f"🚀 Navigating to ({x:.2f}, {y:.2f}, θ={theta:.2f}){tol_str}...")

        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("❌ Nav2 rejected the goal (path blocked or invalid!)")
            return False

        self.get_logger().info("🛣️  Path accepted! Robot is driving...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        # STATUS 4 = SUCCEEDED, STATUS 6 = ABORTED
        status = result_future.result().status
        if status == 4:
            self.get_logger().info("✅ Goal reached!")
            return True
        else:
            self.get_logger().error(
                f"⚠️  Nav2 finished with status {status}. "
                f"Robot may be stuck or path is blocked."
            )
            return False

    def navigate_through_poses(self, pose_list: list) -> bool:
        """
        Send a batch of poses via NavigateThroughPoses.

        The robot flows smoothly through the list using RemovePassedGoals —
        it does NOT fully stop or rotate at each intermediate point.
        Only orientation of the LAST pose is enforced.

        Args:
            pose_list: List of dicts with 'x', 'y' keys.
                       Orientation is set to identity (no yaw enforcement).

        Returns:
            True if all poses reached, False otherwise
        """
        if not pose_list:
            return True

        self.get_logger().info(
            f"🛤️  Navigating through {len(pose_list)} poses (batch, no spinning)..."
        )
        self._through_poses_client.wait_for_server(timeout_sec=10.0)

        goal_msg = NavigateThroughPoses.Goal()
        for p in pose_list:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = float(p["x"])
            ps.pose.position.y = float(p["y"])
            ps.pose.position.z = 0.0
            # Identity quaternion — do NOT enforce any heading at transit points
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = 0.0
            ps.pose.orientation.w = 1.0
            goal_msg.poses.append(ps)

        send_future = self._through_poses_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(
                "⚠️  NavigateThroughPoses rejected — falling back to single goals"
            )
            return False

        self.get_logger().info("🛣️  Batch path accepted! Robot flowing through...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        status = result_future.result().status
        if status == 4:
            self.get_logger().info("✅ Batch navigation complete!")
            return True
        else:
            self.get_logger().warn(
                f"⚠️  Batch navigation finished with status {status}."
            )
            return False

    def _get_transit_waypoints(self, from_node_id: int, to_node_id: int):
        """
        Look up A*-computed transit waypoints between two nodes.
        Tries both key orderings (A-B and B-A).
        Returns list of {x, y} dicts, or [] if no transit path (direct ok).
        """
        key_fwd = f"{from_node_id}-{to_node_id}"
        key_rev = f"{to_node_id}-{from_node_id}"
        # Blocked edge — warn but fall back to direct
        if key_fwd in self._blocked_edges or key_rev in self._blocked_edges:
            self.get_logger().warn(
                f"⚠️  Edge {from_node_id}→{to_node_id} is blocked in transit graph. "
                f"Attempting direct navigation..."
            )
            return []
        if key_fwd in self._transit_edges:
            return self._transit_edges[key_fwd]
        if key_rev in self._transit_edges:
            # Reverse the waypoints for the reverse direction
            return list(reversed(self._transit_edges[key_rev]))
        return []  # No precomputed path — use direct navigation

    def _approach_landmark(self, wp: dict) -> bool:
        """
        After reaching a landmark node, drive TOWARD the actual object.

        Uses the approach_goal computed by the pipeline (CLIP patch heatmap +
        depth unprojection). Falls back to driving 0.8m in the best capture
        direction if no approach_goal is available.

        Returns True if approach succeeded.
        """
        approach_goal = wp.get("approach_goal")

        if approach_goal is not None:
            goal_x = approach_goal["x"]
            goal_y = approach_goal["y"]
            goal_theta = approach_goal["theta"]
            source = wp.get("approach_source", "computed")
            self.get_logger().info(
                f"🎯 Approaching object via {source}: ({goal_x:.2f}, {goal_y:.2f})"
            )
        else:
            # Fallback: drive 0.8m in the best-angle capture direction
            best_theta = wp.get("best_clip_theta", wp.get("theta", 0.0))
            goal_x = wp["x"] + 0.8 * math.cos(best_theta)
            goal_y = wp["y"] + 0.8 * math.sin(best_theta)
            goal_theta = best_theta
            self.get_logger().info(
                f"🎯 Approaching object via bearing fallback "
                f"(θ={best_theta:.2f}): ({goal_x:.2f}, {goal_y:.2f})"
            )

        return self.navigate_to_pose(goal_x, goal_y, goal_theta)

    def execute_walk(self, walk_data: dict) -> dict:
        """
        Execute the full planned walk.

        Uses NavigateThroughPoses for transit+intermediate waypoints (smooth,
        no spinning) and NavigateToPose only for the final landmark waypoint.
        After reaching a landmark, executes an approach phase to drive toward
        the actual object.
        
        Args:
            walk_data: Dict from planned_walk.json with 'waypoints' list
            
        Returns:
            Execution report with success/failure per waypoint
        """
        waypoints = walk_data["waypoints"]
        landmarks = walk_data.get("landmarks", [])
        landmark_nodes = set(walk_data.get("landmark_nodes", []))

        self.get_logger().info(
            f"\n{'='*50}\n"
            f"🧭 Executing LM-Nav Walk\n"
            f"   Instruction: \"{walk_data.get('instruction', 'N/A')}\"\n"
            f"   Landmarks: {landmarks}\n"
            f"   Waypoints: {len(waypoints)}\n"
            f"{'='*50}\n"
        )

        # ── Fix 5: Return to graph if robot is off-graph from last approach ──
        from lmnav import state_manager
        state_path = str(project_root / "output" / "robot_state.json")
        saved_state = state_manager.load_state(state_path)
        if saved_state and saved_state.get("is_off_graph") and saved_state.get("return_pose"):
            rp = saved_state["return_pose"]
            self.get_logger().info(
                f"↩️  Robot is off-graph from last approach. "
                f"Returning to node at ({rp['x']:.2f}, {rp['y']:.2f}) first..."
            )
            self.navigate_to_pose(rp["x"], rp["y"], rp.get("theta", 0.0))
            # Clear the off-graph flag
            state_manager.save_state(
                state_path=state_path,
                node_id=saved_state["node_id"],
                x=rp["x"], y=rp["y"], theta=rp.get("theta", 0.0),
                is_off_graph=False, return_pose=None,
            )
            self.get_logger().info("✅ Back on graph. Executing new walk.")

        results = []
        landmark_idx = 0
        start_time = time.time()
        last_success_idx = -1  # Track last successfully reached waypoint
        approached = False     # Track if we did an approach

        for i, wp in enumerate(waypoints):
            node_id = wp["node_id"]
            x, y, theta = wp["x"], wp["y"], wp["theta"]

            # Check if this waypoint is a landmark node
            is_landmark = wp.get("is_landmark", node_id in landmark_nodes)
            landmark_label = ""
            if is_landmark and landmark_idx < len(landmarks):
                landmark_label = f" 🏷️  LANDMARK: \"{landmarks[landmark_idx]}\""

            self.get_logger().info(
                f"\n{'─'*40}\n"
                f"📍 Waypoint {i+1}/{len(waypoints)}: "
                f"Node {node_id} ({x:.2f}, {y:.2f}){landmark_label}\n"
                f"{'─'*40}"
            )

            # ── Collect transit waypoints ─────────────────────────────────────
            prev_node = waypoints[i - 1]["node_id"] if i > 0 else None
            transit_wps = []
            if prev_node is not None and prev_node != node_id:
                transit_wps = self._get_transit_waypoints(prev_node, node_id)

            if not is_landmark:
                # ── INTERMEDIATE NODE: batch transit + node via NavigateThroughPoses ──
                # Combine transit waypoints + the node itself into one smooth batch
                batch_poses = list(transit_wps)  # list of {x, y}
                batch_poses.append({"x": x, "y": y})

                if len(batch_poses) > 1:
                    batch_ok = self.navigate_through_poses(batch_poses)
                    if not batch_ok:
                        # Fallback: try direct NavigateToPose with identity orientation
                        self.get_logger().info(
                            "   ↪ Batch failed, trying direct navigation..."
                        )
                        batch_ok = self.navigate_to_pose(x, y, theta=0.0)
                    success = batch_ok
                else:
                    # Just one point → use NavigateThroughPoses still (identity quat)
                    success = self.navigate_through_poses(batch_poses)
                    if not success:
                        success = self.navigate_to_pose(x, y, theta=0.0)
            else:
                # ── LANDMARK NODE: transit via batch, final node via NavigateToPose ──
                if transit_wps:
                    self.get_logger().info(
                        f"   📍 Flowing through {len(transit_wps)} transit waypoints..."
                    )
                    self.navigate_through_poses(transit_wps)

                # Navigate to exact landmark pose (with orientation!)
                success = self.navigate_to_pose(x, y, theta)

                # ── APPROACH PHASE: drive toward the actual object ────────────
                if success:
                    approach_ok = self._approach_landmark(wp)
                    if approach_ok:
                        approached = True
                        self.get_logger().info("✅ Approached landmark!")
                    else:
                        self.get_logger().warn(
                            "⚠️  Approach failed — robot stays at graph node."
                        )

                if is_landmark and landmark_idx < len(landmarks):
                    landmark_idx += 1

            results.append({
                "node_id": node_id,
                "x": x,
                "y": y,
                "success": success,
                "is_landmark": is_landmark,
            })

            if success:
                last_success_idx = i

            # Brief pause between waypoints
            time.sleep(0.1)

        elapsed = time.time() - start_time
        successes = sum(1 for r in results if r["success"])

        self.get_logger().info(
            f"\n{'='*50}\n"
            f"🎉 Walk Execution Complete!\n"
            f"   Waypoints reached: {successes}/{len(waypoints)}\n"
            f"   Total time: {elapsed:.1f}s\n"
            f"   Distance planned: {walk_data.get('walk_distance_m', 'N/A')}m\n"
            f"{'='*50}"
        )

        return {
            "total_waypoints": len(waypoints),
            "successful": successes,
            "failed": len(waypoints) - successes,
            "elapsed_seconds": elapsed,
            "results": results,
            "last_success_idx": last_success_idx,
            "approached": approached,
        }


def main():
    if not HAS_ROS:
        print("❌ ROS2 is required. Source your workspace and try again.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Execute a planned LM-Nav walk via Nav2"
    )
    parser.add_argument(
        "--walk-file", "-w",
        type=str,
        default=None,
        help="Path to planned_walk.json (default: output/planned_walk.json)",
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Increase robot speed (0.5 m/s instead of default 0.26 m/s)",
    )
    args = parser.parse_args()

    # Find walk file
    if args.walk_file:
        walk_path = Path(args.walk_file)
    else:
        project_root = Path(__file__).resolve().parent.parent
        walk_path = project_root / "output" / "planned_walk.json"

    if not walk_path.exists():
        print(f"❌ Walk file not found: {walk_path}")
        print("   Run the pipeline first:")
        print('   python scripts/run_pipeline.py -i "Go to the kitchen..."')
        sys.exit(1)

    # Load walk data
    with open(walk_path, "r") as f:
        walk_data = json.load(f)

    print(f"📄 Loaded walk from: {walk_path}")
    print(f"   Instruction: \"{walk_data.get('instruction', 'N/A')}\"")
    print(f"   Waypoints: {len(walk_data['waypoints'])}")

    # Execute
    rclpy.init()
    executor = WalkExecutor()

    try:
        # Ensure AMCL is localized before starting
        if not executor.wait_for_nav2():
            print("❌ Nav2 not ready. Exiting.")
            sys.exit(1)

        # Increase robot speed if --fast flag is set
        if args.fast:
            import subprocess
            print("🏎️  Fast mode: increasing max velocity to 0.5 m/s...")
            try:
                subprocess.run(
                    ["ros2", "param", "set", "/controller_server",
                     "FollowPath.max_vel_x", "0.5"],
                    capture_output=True, timeout=5
                )
                subprocess.run(
                    ["ros2", "param", "set", "/controller_server",
                     "FollowPath.max_speed_xy", "0.5"],
                    capture_output=True, timeout=5
                )
                print("   ✅ Speed increased!")
            except Exception as e:
                print(f"   ⚠️  Could not set speed params: {e}")
                print("   Continuing with default speed.")

        report = executor.execute_walk(walk_data)

        # ── Save robot state for next pipeline run ──
        last_idx = report.get("last_success_idx", -1)
        if last_idx >= 0:
            from lmnav import state_manager

            last_wp = walk_data["waypoints"][last_idx]

            # Try to get actual AMCL pose (more accurate than planned coords)
            actual_x, actual_y, actual_theta = last_wp["x"], last_wp["y"], last_wp.get("theta", 0.0)
            if executor._latest_amcl is not None:
                pos = executor._latest_amcl.pose.pose.position
                ori = executor._latest_amcl.pose.pose.orientation
                actual_x = pos.x
                actual_y = pos.y
                actual_theta = math.atan2(
                    2.0 * (ori.w * ori.z + ori.x * ori.y),
                    1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z),
                )
                print(f"📍 Using AMCL pose for state: ({actual_x:.2f}, {actual_y:.2f})")
            else:
                print(f"📍 Using planned coords for state (no AMCL): ({actual_x:.2f}, {actual_y:.2f})")

            state_path = str(project_root / "output" / "robot_state.json")

            # If we approached a landmark, mark the robot as off-graph
            if report.get("approached", False):
                state_manager.save_state(
                    state_path=state_path,
                    node_id=last_wp["node_id"],
                    x=actual_x, y=actual_y, theta=actual_theta,
                    is_off_graph=True,
                    return_pose={
                        "x": last_wp["x"],
                        "y": last_wp["y"],
                        "theta": last_wp.get("theta", 0.0),
                    },
                )
            else:
                state_manager.save_state(
                    state_path=state_path,
                    node_id=last_wp["node_id"],
                    x=actual_x, y=actual_y, theta=actual_theta,
                )
        else:
            print("⚠️  No waypoints reached — robot state not saved.")

        # Save execution report
        report_path = walk_path.parent / "execution_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Execution report saved to: {report_path}")

    except KeyboardInterrupt:
        print("\n🛑 Execution cancelled by user.")
    finally:
        executor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
