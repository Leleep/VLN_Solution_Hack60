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

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
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
        
        # Odom subscriber for reading robot's current position
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self._latest_odom = None
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.get_logger().info("🤖 Walk Executor initialized.")

    def _odom_callback(self, msg):
        self._latest_odom = msg

    def wait_for_nav2(self, timeout_sec: float = 120.0) -> bool:
        """Wait for AMCL to localize and Nav2 to accept goals."""
        self.get_logger().info("⏳ Waiting for Nav2 action server...")
        if not self._action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("❌ Nav2 action server not available!")
            return False

        # Wait for odom
        self.get_logger().info("📡 Waiting for odometry...")
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_odom is not None:
                break

        # Get current position from odom
        if self._latest_odom:
            pos = self._latest_odom.pose.pose.position
            ori = self._latest_odom.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (ori.w * ori.z + ori.x * ori.y),
                1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
            )
            x, y = pos.x, pos.y
        else:
            x, y, yaw = -2.0, -0.5, 0.0

        self.get_logger().info(f"📍 Robot odom: ({x:.2f}, {y:.2f}, θ={yaw:.2f})")

        # Repeatedly publish initial pose until AMCL localizes
        from geometry_msgs.msg import PoseWithCovarianceStamped
        max_attempts = 30
        for attempt in range(max_attempts):
            # Publish initial pose
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
            msg.pose.covariance[0] = 0.25
            msg.pose.covariance[7] = 0.25
            msg.pose.covariance[35] = 0.06
            for _ in range(5):
                self._initial_pose_pub.publish(msg)
                time.sleep(0.1)

            # Wait for AMCL to process
            for _ in range(20):
                rclpy.spin_once(self, timeout_sec=0.1)

            # Test if Nav2 accepts goals
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
                    # Brief pause for costmaps
                    for _ in range(30):
                        rclpy.spin_once(self, timeout_sec=0.1)
                    return True

            self.get_logger().info(f"   ⏳ AMCL not ready (attempt {attempt+1}/{max_attempts})...")

        self.get_logger().error("❌ AMCL did not localize in time!")
        return False

    def navigate_to_pose(self, x: float, y: float, theta: float = 0.0) -> bool:
        """
        Send a single Nav2 goal and wait for completion.
        
        Args:
            x: Target x position (meters, in 'map' frame)
            y: Target y position (meters, in 'map' frame)
            theta: Target heading (radians)
            
        Returns:
            True if goal reached, False otherwise
        """
        self.get_logger().info("⏳ Waiting for Nav2 Action Server...")
        self._action_client.wait_for_server()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta (yaw) to quaternion
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        self.get_logger().info(f"🚀 Navigating to ({x:.2f}, {y:.2f}, θ={theta:.2f})...")

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

    def execute_walk(self, walk_data: dict) -> dict:
        """
        Execute the full planned walk.
        
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

        results = []
        landmark_idx = 0
        start_time = time.time()

        for i, wp in enumerate(waypoints):
            node_id = wp["node_id"]
            x, y, theta = wp["x"], wp["y"], wp["theta"]

            # Check if this waypoint is a landmark node
            is_landmark = node_id in landmark_nodes
            landmark_label = ""
            if is_landmark and landmark_idx < len(landmarks):
                landmark_label = f" 🏷️  LANDMARK: \"{landmarks[landmark_idx]}\""
                landmark_idx += 1

            self.get_logger().info(
                f"\n{'─'*40}\n"
                f"📍 Waypoint {i+1}/{len(waypoints)}: "
                f"Node {node_id} ({x:.2f}, {y:.2f}){landmark_label}\n"
                f"{'─'*40}"
            )

            success = self.navigate_to_pose(x, y, theta)
            results.append({
                "node_id": node_id,
                "x": x,
                "y": y,
                "success": success,
                "is_landmark": is_landmark,
            })

            if not success:
                self.get_logger().warn(
                    f"⚠️  Failed to reach node {node_id}. Continuing to next waypoint..."
                )

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
