#!/usr/bin/env python3
"""
Egocentric View — See through the robot's camera
=================================================
Opens an OpenCV window showing the TurtleBot3's camera feed in real-time.

Run this in a separate terminal alongside execute_walk.py to see
the robot's first-person perspective as it navigates.

Controls:
    q     — Quit
    s     — Save screenshot
    f     — Toggle fullscreen

Usage:
    python scripts/ego_view.py
    python scripts/ego_view.py --topic /intel_realsense_r200_depth/image_raw
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    HAS_ROS = True
except ImportError:
    HAS_ROS = False
    print("⚠️  ROS2 not available.")

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  OpenCV not available. Install: pip install opencv-python")


class EgoViewer(Node):
    """ROS2 node that subscribes to camera and displays in OpenCV window."""

    def __init__(self, topic: str, save_dir: str = "output"):
        super().__init__("ego_viewer")
        self._latest_frame = None
        self._frame_count = 0
        self._screenshot_count = 0
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._fullscreen = False
        self._window_name = "🤖 Robot Egocentric View"

        self._sub = self.create_subscription(
            Image, topic, self._image_callback, 10
        )
        self.get_logger().info(f"📹 Subscribing to: {topic}")
        self.get_logger().info("   Controls: [q]uit  [s]creenshot  [f]ullscreen")

    def _image_callback(self, msg):
        """Convert ROS Image to OpenCV format."""
        try:
            # Handle different encodings
            if msg.encoding in ("rgb8", "bgr8"):
                dtype = np.uint8
                channels = 3
            elif msg.encoding in ("rgba8", "bgra8"):
                dtype = np.uint8
                channels = 4
            elif msg.encoding == "mono8":
                dtype = np.uint8
                channels = 1
            elif msg.encoding in ("16UC1", "32FC1"):
                # Depth image — normalize for display
                if msg.encoding == "16UC1":
                    dtype = np.uint16
                else:
                    dtype = np.float32
                channels = 1
            else:
                dtype = np.uint8
                channels = 3

            # Decode image
            img = np.frombuffer(msg.data, dtype=dtype)
            img = img.reshape(msg.height, msg.width, channels) if channels > 1 else img.reshape(msg.height, msg.width)

            # Convert to BGR for OpenCV display
            if msg.encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "rgba8":
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif msg.encoding == "bgra8":
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif msg.encoding in ("16UC1",):
                # Normalize depth for visualization
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            elif msg.encoding == "32FC1":
                img = np.nan_to_num(img, nan=0.0)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

            self._latest_frame = img
            self._frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Frame decode error: {e}")

    def save_screenshot(self):
        """Save current frame as PNG."""
        if self._latest_frame is not None:
            self._screenshot_count += 1
            path = self._save_dir / f"ego_screenshot_{self._screenshot_count:03d}.png"
            cv2.imwrite(str(path), self._latest_frame)
            self.get_logger().info(f"📸 Screenshot saved: {path}")

    def run(self):
        """Main display loop."""
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, 800, 600)

        print("\n" + "=" * 50)
        print("🤖 Egocentric View — Robot Camera Feed")
        print("=" * 50)
        print("   [q] Quit   [s] Screenshot   [f] Fullscreen")
        print("=" * 50 + "\n")

        last_status = time.time()

        while rclpy.ok():
            # Process ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.01)

            # Display frame
            if self._latest_frame is not None:
                # Add HUD overlay
                display = self._latest_frame.copy()
                h, w = display.shape[:2]

                # Semi-transparent status bar at bottom
                overlay = display.copy()
                cv2.rectangle(overlay, (0, h - 35), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

                # Status text
                cv2.putText(
                    display,
                    f"LM-Nav Ego View | Frame {self._frame_count} | [q]uit [s]creenshot [f]ullscreen",
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                )

                cv2.imshow(self._window_name, display)
            else:
                # Waiting screen
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank, "Waiting for camera feed...",
                    (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2,
                )
                cv2.imshow(self._window_name, blank)

                # Status update every 3 seconds
                if time.time() - last_status > 3.0:
                    self.get_logger().info("📡 Waiting for camera frames...")
                    last_status = time.time()

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot()
            elif key == ord('f'):
                self._fullscreen = not self._fullscreen
                if self._fullscreen:
                    cv2.setWindowProperty(
                        self._window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                else:
                    cv2.setWindowProperty(
                        self._window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL,
                    )

        cv2.destroyAllWindows()


def main():
    if not HAS_ROS:
        print("❌ ROS2 required.")
        sys.exit(1)
    if not HAS_CV2:
        print("❌ OpenCV required: pip install opencv-python")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Egocentric View — see through the robot's camera"
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default="/intel_realsense_r200_depth/image_raw",
        help="Camera topic (default: /intel_realsense_r200_depth/image_raw)",
    )
    parser.add_argument(
        "--depth", "-d",
        action="store_true",
        help="Show depth camera instead of RGB",
    )
    args = parser.parse_args()

    topic = args.topic
    if args.depth:
        topic = "/intel_realsense_r200_depth/depth/image_raw"

    rclpy.init()
    viewer = EgoViewer(topic)

    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\n🛑 Viewer closed.")
    finally:
        viewer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
