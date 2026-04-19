#!/usr/bin/env python3
"""
Automated House Exploration — Graph Builder
============================================
ROS2 node that drives the TurtleBot3 through the AWS Small House World,
capturing RGB images + poses at predefined waypoints.

This is the LM-Nav "exploration phase" — the robot traverses the environment
to build the topological graph that CLIP and Dijkstra DP operate on.

Usage:
  # 1. Launch Gazebo with the house world + TurtleBot3 + Nav2
  # 2. Run this script:
  python scripts/explore_house.py

Output:
  data/aws_house_graph/
  ├── node_000.png ... node_029.png
  └── poses.json
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ROS2 imports — these are available when ROS2 is sourced
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
    from sensor_msgs.msg import Image as ROSImage
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("⚠️  ROS2 not available. This script requires ROS2 Humble.")
    print("   Source your ROS2 workspace: source /opt/ros/humble/setup.bash")

try:
    from PIL import Image as PILImage
except ImportError:
    print("⚠️  Pillow not installed. Run: pip install Pillow")
    sys.exit(1)


# ─── Predefined Exploration Waypoints ────────────────────────────────────────
# Dense whole-house coverage with multiple viewing angles per room.
# Every coordinate based on verified-reachable positions from prior runs.
#
# House layout (from navigation data + map):
#   Bedroom:     x ∈ [-6, -3],  y ∈ [0, 2]
#   Hallway:     x ∈ [-2, 4],   y ∈ [-0.5, 0]
#   Living Room: x ∈ [0, 3],    y ∈ [-0.5, -2]    (sofa at ~y=-2.5, AVOID)
#   Kitchen:     x ∈ [5, 8.5],  y ∈ [-0.5, -3.5]
#   Bathroom:    x ∈ [-3, -1],  y ∈ [0.5, 2]      (north of hallway)
#
# theta: 0=east, π/2=north, π=west, -π/2=south
# Total: 55 waypoints

EXPLORATION_WAYPOINTS = [
    # ═══ Phase 1: Hallway — full corridor with multiple angles ═══
    {"x": -1.5, "y": -0.3, "theta":  0.0,    "label": "hallway_spawn"},
    {"x": -1.0, "y": -0.3, "theta":  1.57,   "label": "hallway_west_north"},    # look north
    {"x": -1.0, "y": -0.3, "theta": -1.57,   "label": "hallway_west_south"},    # look south
    {"x":  0.0, "y": -0.3, "theta":  0.0,    "label": "hallway_mid"},
    {"x":  0.5, "y": -0.3, "theta": -1.57,   "label": "hallway_center"},        # look into living room
    {"x":  1.5, "y": -0.3, "theta":  0.0,    "label": "hallway_east_mid"},
    {"x":  2.0, "y": -0.3, "theta":  0.0,    "label": "hallway_east"},
    {"x":  3.0, "y": -0.3, "theta":  0.0,    "label": "hallway_far_east"},

    # ═══ Phase 2: Living Room — open areas, multi-angle views ═══
    {"x":  1.5, "y": -0.8, "theta": -1.57,   "label": "living_room_entry"},     # looking south into room
    {"x":  0.8, "y": -1.2, "theta":  0.0,    "label": "near_sofa_east"},        # facing TV
    {"x":  0.8, "y": -1.2, "theta":  3.14,   "label": "near_sofa_west"},        # facing sofa
    {"x":  1.5, "y": -1.2, "theta": -1.57,   "label": "living_room_center"},
    {"x":  2.2, "y": -1.0, "theta":  3.14,   "label": "facing_tv"},             # face west toward TV
    {"x":  1.5, "y": -1.8, "theta":  3.14,   "label": "living_room_south"},     # deepest safe point
    {"x":  2.5, "y": -1.2, "theta":  0.0,    "label": "living_room_east"},
    {"x":  1.0, "y": -1.8, "theta":  1.57,   "label": "near_coffee_table"},

    # ═══ Phase 3: Transition to Kitchen ═══
    {"x":  3.5, "y": -0.3, "theta":  0.0,    "label": "hallway_to_kitchen"},
    {"x":  4.5, "y": -0.4, "theta":  0.0,    "label": "kitchen_hallway"},
    {"x":  5.0, "y": -0.4, "theta":  0.0,    "label": "kitchen_entrance"},

    # ═══ Phase 4: Kitchen — comprehensive coverage ═══
    {"x":  6.0, "y": -0.4, "theta":  0.0,    "label": "kitchen_counter"},
    {"x":  6.0, "y": -0.4, "theta": -1.57,   "label": "kitchen_counter_south"}, # look into kitchen
    {"x":  7.0, "y": -0.6, "theta":  1.57,   "label": "facing_fridge"},
    {"x":  7.3, "y": -1.0, "theta":  3.14,   "label": "fridge_area"},
    {"x":  6.4, "y": -1.5, "theta":  1.57,   "label": "kitchen_center_north"},
    {"x":  6.4, "y": -1.5, "theta": -1.57,   "label": "kitchen_center_south"},
    {"x":  7.8, "y": -1.6, "theta":  0.0,    "label": "kitchen_far_east"},
    {"x":  7.0, "y": -2.0, "theta":  1.57,   "label": "kitchen_table"},
    {"x":  5.6, "y": -1.9, "theta":  0.0,    "label": "kitchen_doorway"},
    {"x":  7.3, "y": -2.3, "theta": -1.57,   "label": "kitchen_east"},
    {"x":  6.5, "y": -3.3, "theta": -1.57,   "label": "kitchen_south"},
    {"x":  5.5, "y": -3.0, "theta":  3.14,   "label": "kitchen_sw_corner"},

    # ═══ Phase 5: Back through hallway ═══
    {"x":  3.5, "y": -0.3, "theta":  3.14,   "label": "back_to_hallway"},
    {"x":  0.5, "y": -0.3, "theta":  3.14,   "label": "hallway_return"},

    # ═══ Phase 6: Bathroom / North rooms ═══
    {"x": -0.5, "y":  0.0, "theta":  1.57,   "label": "north_hallway"},
    {"x": -1.0, "y":  0.5, "theta":  1.57,   "label": "bathroom_entrance"},
    {"x": -1.5, "y":  1.0, "theta":  3.14,   "label": "bathroom_left"},
    {"x": -2.0, "y":  1.0, "theta":  1.57,   "label": "bathroom_center"},

    # ═══ Phase 7: Bedroom — full coverage ═══
    {"x": -3.3, "y":  0.4, "theta":  3.14,   "label": "bedroom_entrance"},
    {"x": -3.5, "y":  0.4, "theta":  1.57,   "label": "bedroom_door_north"},    # look at bed
    {"x": -4.5, "y":  0.6, "theta":  3.14,   "label": "bedroom_center"},
    {"x": -4.5, "y":  0.6, "theta":  1.57,   "label": "facing_bed"},            # look north at bed
    {"x": -5.4, "y":  0.6, "theta":  1.57,   "label": "bedroom_far"},
    {"x": -5.4, "y":  0.6, "theta":  3.14,   "label": "facing_wardrobe"},       # look west at wardrobe
    {"x": -4.5, "y":  1.5, "theta": -1.57,   "label": "bedroom_window"},
    {"x": -3.5, "y":  1.5, "theta":  0.0,    "label": "bedroom_nightstand"},

    # ═══ Phase 8: Exercise / Fitness Room (SW of bedroom — blue ball area) ═══
    {"x": -5.5, "y": -0.5, "theta": -1.57,   "label": "fitness_entrance"},
    {"x": -6.0, "y": -1.5, "theta":  3.14,   "label": "fitness_west"},
    {"x": -5.0, "y": -1.5, "theta": -1.57,   "label": "fitness_center"},
    {"x": -5.5, "y": -2.5, "theta":  0.0,    "label": "near_exercise_ball"},
    {"x": -4.5, "y": -2.0, "theta":  3.14,   "label": "fitness_east"},

    # ═══ Phase 9: Southeast Room (below living room / dining area) ═══
    {"x":  2.0, "y": -2.5, "theta": -1.57,   "label": "se_room_entrance"},
    {"x":  3.0, "y": -3.5, "theta":  0.0,    "label": "se_room_center"},
    {"x":  4.0, "y": -3.5, "theta":  0.0,    "label": "se_room_east"},
    {"x":  3.0, "y": -4.5, "theta": -1.57,   "label": "se_room_south"},
    {"x":  2.0, "y": -4.0, "theta":  3.14,   "label": "se_room_west"},
]


class HouseExplorer(Node):
    """ROS2 node that drives to waypoints and captures images + poses."""

    def __init__(self, output_dir: str):
        super().__init__("house_explorer")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Nav2 action client for driving to waypoints
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Publisher for initial pose (AMCL needs this to localize)
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            "/initialpose",
            10,
        )

        # Image subscriber (no cv_bridge — avoids numpy ABI issues)
        self._latest_image = None
        self._image_sub = self.create_subscription(
            ROSImage,
            "/intel_realsense_r200_depth/image_raw",  # TurtleBot3 Waffle RealSense RGB
            self._image_callback,
            10,
        )

        # Depth image subscriber (for depth-gated capture)
        self._latest_depth = None
        self._depth_sub = self.create_subscription(
            ROSImage,
            "/intel_realsense_r200_depth/depth/image_raw",  # 16-bit depth, mm
            self._depth_callback,
            10,
        )

        # Camera intrinsics subscriber (for depth unprojection in approach phase)
        from sensor_msgs.msg import CameraInfo
        self._camera_intrinsics = None
        self._cam_info_sub = self.create_subscription(
            CameraInfo,
            "/intel_realsense_r200_depth/depth/camera_info",
            self._cam_info_callback,
            10,
        )

        # Odometry subscriber for ground-truth pose
        self._latest_odom = None
        self._odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_callback,
            10,
        )

        self.get_logger().info(f"🏠 House Explorer initialized. Output: {self.output_dir}")

    def _image_callback(self, msg: ROSImage):
        """Store the latest camera image (direct numpy, no cv_bridge)."""
        try:
            # Convert ROS Image to numpy array directly
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == "rgb8":
                img_array = img_array.reshape((msg.height, msg.width, 3))
            elif msg.encoding == "bgr8":
                img_array = img_array.reshape((msg.height, msg.width, 3))
                img_array = img_array[:, :, ::-1]  # BGR → RGB
            elif msg.encoding == "rgba8":
                img_array = img_array.reshape((msg.height, msg.width, 4))[:, :, :3]
            else:
                self.get_logger().warn(f"Unknown encoding: {msg.encoding}, trying rgb8")
                img_array = img_array.reshape((msg.height, msg.width, 3))
            self._latest_image = img_array.copy()
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def _odom_callback(self, msg: Odometry):
        """Store the latest odometry."""
        self._latest_odom = msg

    def _depth_callback(self, msg: ROSImage):
        """Store the latest depth image."""
        try:
            if msg.encoding == "16UC1":
                depth_array = np.frombuffer(msg.data, dtype=np.uint16)
                depth_array = depth_array.reshape((msg.height, msg.width))
            elif msg.encoding == "32FC1":
                depth_array = np.frombuffer(msg.data, dtype=np.float32)
                depth_array = depth_array.reshape((msg.height, msg.width))
            else:
                return  # Unknown encoding
            self._latest_depth = depth_array.copy()
            self._depth_encoding = msg.encoding
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth: {e}")

    def _cam_info_callback(self, msg):
        """Store the camera intrinsics (once)."""
        if self._camera_intrinsics is None:
            self._camera_intrinsics = {
                "fx": msg.k[0],
                "fy": msg.k[4],
                "cx": msg.k[2],
                "cy": msg.k[5],
                "width": msg.width,
                "height": msg.height,
            }

    def _get_current_pose(self):
        """Extract (x, y, theta) from latest odometry."""
        if self._latest_odom is None:
            return None
        pos = self._latest_odom.pose.pose.position
        ori = self._latest_odom.pose.pose.orientation
        # Convert quaternion to yaw
        siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        return (pos.x, pos.y, theta)

    def publish_initial_pose(self, x: float = -2.0, y: float = -0.5, theta: float = 0.0):
        """
        Publish initial pose for AMCL localization.
        
        The TurtleBot3 spawns at approximately (-2.0, -0.5) in the AWS House World.
        AMCL needs to know where the robot is to localize on the pre-built map.
        """
        from geometry_msgs.msg import PoseWithCovarianceStamped
        
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        # Set covariance (small uncertainty)
        msg.pose.covariance[0] = 0.25  # x variance
        msg.pose.covariance[7] = 0.25  # y variance
        msg.pose.covariance[35] = 0.06  # yaw variance

        self.get_logger().info(f"📌 Publishing initial pose: ({x:.2f}, {y:.2f}, θ={theta:.2f})")
        # Publish several times to make sure AMCL picks it up
        for _ in range(5):
            self._initial_pose_pub.publish(msg)
            time.sleep(0.1)
        
        # Give AMCL time to process
        self.get_logger().info("⏳ Waiting 3s for AMCL to localize...")
        time.sleep(3.0)

    def wait_for_nav2_ready(self, timeout_sec: float = 120.0):
        """
        Wait for Nav2 to be fully initialized and ready to accept goals.
        
        Sequence:
          1. Wait for action server
          2. Wait for odometry
          3. Repeatedly publish initial pose until AMCL creates the map frame
          4. Verify by sending a test goal
        """
        self.get_logger().info("⏳ Waiting for Nav2 to be fully ready...")
        
        # Wait for action server
        if not self._nav_client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error(
                f"❌ Nav2 action server not available after {timeout_sec}s! "
                f"Make sure Nav2 is launched with a pre-built map."
            )
            return False

        self.get_logger().info("✅ Nav2 action server found.")

        # Wait for odometry
        self.get_logger().info("📡 Waiting for odometry data...")
        for _ in range(100):  # up to 10 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_odom is not None:
                break
        
        if self._latest_odom is None:
            self.get_logger().error("❌ No odometry received after 10s!")
            return False

        # Get the robot's actual position
        current_pose = self._get_current_pose()
        x, y, theta = current_pose if current_pose else (-2.0, -0.5, 0.0)
        self.get_logger().info(
            f"📍 Robot's current odom: ({x:.2f}, {y:.2f}, θ={theta:.2f})"
        )

        # Repeatedly publish initial pose until AMCL processes it
        # AMCL needs time to start up — the map frame won't exist until it does
        self.get_logger().info(
            "📌 Publishing initial pose repeatedly until AMCL localizes...\n"
            "   (This may take 15-30 seconds after Gazebo launch)"
        )
        
        max_attempts = 30  # try for up to 30 * 2 = 60 seconds
        amcl_ready = False
        
        for attempt in range(max_attempts):
            # Publish initial pose
            self.publish_initial_pose(x, y, theta)
            
            # Spin and wait a bit
            for _ in range(20):  # 2 seconds
                rclpy.spin_once(self, timeout_sec=0.1)
            
            # Try a test goal to see if Nav2 accepts it
            # (Nav2 won't accept goals until AMCL has the map frame)
            import math as _math
            test_goal = NavigateToPose.Goal()
            test_goal.pose.header.frame_id = "map"
            test_goal.pose.header.stamp = self.get_clock().now().to_msg()
            test_goal.pose.pose.position.x = x
            test_goal.pose.pose.position.y = y
            test_goal.pose.pose.orientation.z = _math.sin(theta / 2.0)
            test_goal.pose.pose.orientation.w = _math.cos(theta / 2.0)
            
            send_future = self._nav_client.send_goal_async(test_goal)
            rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
            
            if send_future.done() and send_future.result() is not None:
                goal_handle = send_future.result()
                if goal_handle.accepted:
                    self.get_logger().info(
                        f"✅ AMCL localized! Nav2 accepting goals (attempt {attempt+1})"
                    )
                    # Cancel the test goal
                    cancel_future = goal_handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=3.0)
                    amcl_ready = True
                    break
            
            self.get_logger().info(
                f"   ⏳ AMCL not ready yet (attempt {attempt+1}/{max_attempts})..."
            )
        
        if not amcl_ready:
            self.get_logger().error(
                "❌ AMCL did not localize after 60s. "
                "Try setting the initial pose manually in RViz2."
            )
            return False

        # Give costmaps a moment to update after AMCL is ready
        self.get_logger().info("⏳ Waiting 3s for costmaps to stabilize...")
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("✅ Nav2 is fully ready! Starting exploration.")
        return True

    def navigate_to(self, x: float, y: float, theta: float, retries: int = 2) -> bool:
        """Send a Nav2 goal and wait for completion, with retries."""
        for attempt in range(retries + 1):
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = "map"
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = float(x)
            goal_msg.pose.pose.position.y = float(y)
            goal_msg.pose.pose.position.z = 0.0

            # Convert theta to quaternion (yaw only)
            goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
            goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

            if attempt > 0:
                self.get_logger().info(f"🔄 Retry {attempt}/{retries}...")
                time.sleep(2.0)  # wait before retry

            self.get_logger().info(f"🚀 Navigating to ({x:.2f}, {y:.2f}, θ={theta:.2f})...")
            send_goal_future = self._nav_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future)

            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().warn(f"⚠️  Nav2 rejected goal (attempt {attempt+1}/{retries+1})")
                if attempt < retries:
                    # Spin to process any pending messages
                    for _ in range(20):
                        rclpy.spin_once(self, timeout_sec=0.1)
                continue

            self.get_logger().info("🛣️  Goal accepted, robot is driving...")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)

            status = result_future.result().status
            if status == 4:  # SUCCEEDED
                self.get_logger().info("✅ Reached waypoint!")
                return True
            else:
                self.get_logger().warn(f"⚠️  Nav2 finished with status {status}")
                if attempt < retries:
                    continue
                return False

        self.get_logger().error(f"❌ Failed after {retries+1} attempts")
        return False

    def capture_node(self, node_id: int) -> dict:
        """Capture current image + pose + depth and save to disk."""
        # Wait for fresh data
        for _ in range(30):  # wait up to 3 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_image is not None and self._latest_odom is not None:
                break

        if self._latest_image is None:
            self.get_logger().error("❌ No camera image received!")
            return None

        # Save image
        img_path = self.output_dir / f"node_{node_id:03d}.png"
        pil_image = PILImage.fromarray(self._latest_image)
        pil_image.save(str(img_path))

        # Get pose
        pose = self._get_current_pose()
        if pose is None:
            self.get_logger().error("❌ No odometry received!")
            return None

        node_data = {
            "id": node_id,
            "x": pose[0],
            "y": pose[1],
            "theta": pose[2],
        }

        # Save depth image as .npy for approach phase (depth unprojection)
        if self._latest_depth is not None:
            depth = self._latest_depth.copy()

            # Convert to float32 meters if needed (16UC1 is in millimeters)
            if hasattr(self, '_depth_encoding') and self._depth_encoding == "16UC1":
                depth_m = depth.astype(np.float32) / 1000.0
            else:
                depth_m = depth.astype(np.float32)

            depth_path = self.output_dir / f"node_{node_id:03d}_depth.npy"
            np.save(str(depth_path), depth_m)

            # Also compute median depth metadata
            h, w = depth.shape
            center = depth[h//4:3*h//4, w//4:3*w//4]
            # Filter invalid readings (0 = no return, >6000mm = too far)
            valid = center[(center > 0) & (center < 6000)]
            if len(valid) > 0:
                median_depth_mm = float(np.median(valid))
                node_data["median_depth_m"] = round(median_depth_mm / 1000.0, 3)
                node_data["depth_source"] = "realsense"
            else:
                node_data["median_depth_m"] = None
                node_data["depth_source"] = "no_valid_readings"
        else:
            node_data["median_depth_m"] = None
            node_data["depth_source"] = "no_depth_topic"

        self.get_logger().info(
            f"📸 Captured node {node_id}: "
            f"({pose[0]:.2f}, {pose[1]:.2f}, θ={pose[2]:.2f}) → {img_path.name}"
            f" depth={node_data.get('median_depth_m', 'N/A')}m"
        )
        return node_data

    def run_exploration(self, waypoints: list) -> list:
        """
        Drive to all waypoints, capture images + poses.

        Optimized: waypoints at the same (x, y) are grouped so the robot
        navigates to each physical position ONCE, then rotates in place for
        each angular capture. This prevents redundant re-navigations that
        cause the robot to get stuck on repeated small goals.
        """
        if not self.wait_for_nav2_ready():
            self.get_logger().error(
                "❌ Nav2 is not ready. Make sure you launched with a pre-built map."
            )
            return []

        # ── Save camera intrinsics (once) for depth unprojection ──────────
        # Wait briefly for CameraInfo to arrive
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._camera_intrinsics is not None:
                break

        if self._camera_intrinsics is not None:
            intrinsics_path = self.output_dir / "camera_intrinsics.json"
            with open(intrinsics_path, "w") as f:
                json.dump(self._camera_intrinsics, f, indent=2)
            self.get_logger().info(
                f"📷 Camera intrinsics saved: {intrinsics_path}\n"
                f"   fx={self._camera_intrinsics['fx']:.1f}, "
                f"fy={self._camera_intrinsics['fy']:.1f}, "
                f"cx={self._camera_intrinsics['cx']:.1f}, "
                f"cy={self._camera_intrinsics['cy']:.1f}"
            )
        else:
            self.get_logger().warn(
                "⚠️  Could not capture camera intrinsics. "
                "Depth-based approach goals will not work."
            )

        # ── Group waypoints by unique (x, y) position ──────────────────────
        from collections import OrderedDict
        position_groups: OrderedDict = OrderedDict()
        for wp in waypoints:
            key = (round(wp["x"], 2), round(wp["y"], 2))
            if key not in position_groups:
                position_groups[key] = []
            position_groups[key].append(wp)

        n_unique   = len(position_groups)
        n_total    = len(waypoints)
        self.get_logger().info(
            f"🗺️  Starting exploration: {n_total} waypoints at "
            f"{n_unique} unique positions"
        )

        all_poses   = []
        failed_count = 0
        node_idx    = 0

        for pos_idx, ((px, py), group) in enumerate(position_groups.items()):
            self.get_logger().info(
                f"\n{'='*50}\n"
                f"📍 Position {pos_idx+1}/{n_unique}: ({px:.2f}, {py:.2f})  "
                f"→  {len(group)} view(s)\n"
                f"{'='*50}"
            )

            # Navigate to this position ONCE (use first angle in the group)
            first_wp  = group[0]
            success   = self.navigate_to(px, py, first_wp["theta"], retries=2)
            if not success:
                failed_count += 1
                self.get_logger().warn(
                    f"⚠️  Could not reach position ({px:.2f}, {py:.2f}). "
                    f"Capturing from current position."
                )

            time.sleep(0.8)  # stabilize after arrival

            # Capture each angular view at this position
            for angle_wp in group:
                # Rotate in place if angle differs from current orientation
                if angle_wp is not first_wp or len(group) == 1:
                    rotate_ok = self.navigate_to(px, py, angle_wp["theta"], retries=1)
                    if not rotate_ok:
                        self.get_logger().warn(
                            f"⚠️  Rotation to θ={angle_wp['theta']:.2f} failed. "
                            f"Capturing current view."
                        )
                    time.sleep(0.5)

                node_data = self.capture_node(node_idx)
                if node_data is not None:
                    node_data["label"]    = angle_wp.get("label", f"auto_{node_idx:03d}")
                    node_data["target_x"] = angle_wp["x"]
                    node_data["target_y"] = angle_wp["y"]
                    node_data["reached"]  = success
                    all_poses.append(node_data)

                node_idx += 1

        # Save poses to JSON
        poses_file = self.output_dir / "poses.json"
        with open(poses_file, "w") as f:
            json.dump(all_poses, f, indent=2)

        self.get_logger().info(
            f"\n🎉 Exploration complete!\n"
            f"   Captured {len(all_poses)}/{n_total} nodes\n"
            f"   Successfully reached: {n_unique - failed_count}/{n_unique} positions\n"
            f"   Data saved to: {self.output_dir}"
        )
        return all_poses


def load_waypoints(project_root: Path, use_legacy: bool = False) -> list:
    """
    Load exploration waypoints.

    Priority:
      1. chamber_nodes.json (auto-generated by generate_waypoints.py)
      2. Legacy hardcoded EXPLORATION_WAYPOINTS (if --legacy-waypoints flag)
    """
    chamber_path = project_root / "data" / "aws_house_graph" / "chamber_nodes.json"

    if not use_legacy and chamber_path.exists():
        print(f"📄 Loading auto-generated waypoints from: {chamber_path}")
        with open(chamber_path, "r") as f:
            waypoints = json.load(f)
        print(f"   Loaded {len(waypoints)} waypoints")
        return waypoints

    if not use_legacy and not chamber_path.exists():
        print(
            f"⚠️  chamber_nodes.json not found at {chamber_path}\n"
            f"   Run 'python scripts/generate_waypoints.py' first, or use\n"
            f"   '--legacy-waypoints' to use the hardcoded waypoint list."
        )
        print("\n🔄 Falling back to legacy waypoints...")

    print(f"📋 Using legacy hardcoded waypoints: {len(EXPLORATION_WAYPOINTS)} waypoints")
    return EXPLORATION_WAYPOINTS


def main():
    """Run the automated house exploration."""
    if not HAS_ROS:
        print("❌ ROS2 is required. Source your workspace and try again.")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Automated House Exploration")
    parser.add_argument(
        "--legacy-waypoints", action="store_true",
        help="Use hardcoded waypoints instead of chamber_nodes.json",
    )
    parser.add_argument(
        "--single-node", type=int, default=None,
        help="Explore only a single waypoint (for testing)",
    )
    args = parser.parse_args()

    # Determine output directory
    script_dir = Path(__file__).resolve().parent.parent
    output_dir = script_dir / "data" / "aws_house_graph"

    # Load waypoints
    waypoints = load_waypoints(script_dir, use_legacy=args.legacy_waypoints)

    # Single-node mode (for testing)
    if args.single_node is not None:
        idx = args.single_node
        if idx < len(waypoints):
            waypoints = [waypoints[idx]]
            print(f"🧪 Single-node mode: testing waypoint {idx}")
        else:
            print(f"❌ Node {idx} out of range (0-{len(waypoints)-1})")
            sys.exit(1)

    rclpy.init()
    explorer = HouseExplorer(str(output_dir))

    try:
        poses = explorer.run_exploration(waypoints)
        print(f"\n✅ Exploration finished. {len(poses)} nodes captured.")
        print(f"   Images: {output_dir}/node_*.png")
        print(f"   Poses:  {output_dir}/poses.json")
    except KeyboardInterrupt:
        print("\n🛑 Exploration cancelled by user.")
    finally:
        explorer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
