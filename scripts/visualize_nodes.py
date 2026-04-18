#!/usr/bin/env python3
"""
Visualize Exploration Nodes on House Map
=========================================
Overlays all exploration waypoints and/or captured poses on the
occupancy grid map (map.pgm) of the AWS Small House.

Usage:
    python scripts/visualize_nodes.py                    # Show planned waypoints
    python scripts/visualize_nodes.py --poses             # Overlay captured poses too
    python scripts/visualize_nodes.py --save map_viz.png  # Save to file
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("❌ Pillow required: pip install Pillow")
    sys.exit(1)

# ── Import waypoints without triggering ROS2 imports ──
def _load_waypoints():
    """Load EXPLORATION_WAYPOINTS from explore_house.py without importing the module."""
    wp_file = Path(__file__).resolve().parent / "explore_house.py"
    source = wp_file.read_text()
    # Extract just the EXPLORATION_WAYPOINTS list
    ns = {}
    # Find the start of the list and extract it
    start = source.find("EXPLORATION_WAYPOINTS = [")
    if start == -1:
        raise RuntimeError("Cannot find EXPLORATION_WAYPOINTS in explore_house.py")
    # Find matching closing bracket
    depth = 0
    end = start
    for i, ch in enumerate(source[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    snippet = source[start:end]
    exec(snippet, ns)
    return ns["EXPLORATION_WAYPOINTS"]

EXPLORATION_WAYPOINTS = _load_waypoints()


# ── Map parameters (from map.yaml) ──
MAP_RESOLUTION = 0.05          # meters per pixel
MAP_ORIGIN_X = -12.5           # world x at pixel (0, height)
MAP_ORIGIN_Y = -12.5           # world y at pixel (0, height)


def world_to_pixel(x, y, map_height):
    """Convert world coordinates (meters) to pixel coordinates on the map."""
    px = int((x - MAP_ORIGIN_X) / MAP_RESOLUTION)
    py = int(map_height - (y - MAP_ORIGIN_Y) / MAP_RESOLUTION)
    return px, py


# ── Room color coding ──
ROOM_COLORS = {
    "hallway":    "#3498db",   # blue
    "living":     "#e74c3c",   # red
    "kitchen":    "#2ecc71",   # green
    "bathroom":   "#9b59b6",   # purple
    "bedroom":    "#f39c12",   # orange
    "fitness":    "#1abc9c",   # teal
    "se_room":    "#e91e63",   # pink
    "back_to":    "#95a5a6",   # grey (transit)
    "north":      "#9b59b6",   # purple (bathroom access)
    "facing":     "#e67e22",   # dark orange
    "near":       "#e74c3c",   # red (living room)
}


def get_color(label):
    """Get color based on waypoint label."""
    for key, color in ROOM_COLORS.items():
        if key in label:
            return color
    return "#34495e"  # default dark grey


def main():
    parser = argparse.ArgumentParser(description="Visualize exploration nodes on house map")
    parser.add_argument("--poses", action="store_true", help="Also show captured poses from poses.json")
    parser.add_argument("--save", type=str, default=None, help="Save to file instead of showing")
    parser.add_argument("--no-labels", action="store_true", help="Don't show node labels (cleaner)")
    args = parser.parse_args()

    # ── Load map ──
    script_dir = Path(__file__).resolve().parent.parent
    map_path = script_dir / "aws-robomaker-small-house-world" / "maps" / "turtlebot3_waffle_pi" / "map.pgm"

    if not map_path.exists():
        print(f"❌ Map not found: {map_path}")
        sys.exit(1)

    map_img = np.array(Image.open(map_path))
    map_height, map_width = map_img.shape[:2]
    print(f"📍 Map loaded: {map_width}x{map_height} pixels, "
          f"resolution={MAP_RESOLUTION}m/px")

    # ── Create figure ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(map_img, cmap="gray", origin="upper")
    ax.set_title("LM-Nav Exploration Nodes — AWS Small House", fontsize=16, fontweight="bold")

    # ── Plot planned waypoints ──
    arrow_len = 12  # pixels
    for idx, wp in enumerate(EXPLORATION_WAYPOINTS):
        px, py = world_to_pixel(wp["x"], wp["y"], map_height)
        color = get_color(wp["label"])

        # Draw node circle
        ax.plot(px, py, "o", color=color, markersize=8, markeredgecolor="white",
                markeredgewidth=1.5, zorder=5)

        # Draw orientation arrow
        dx = arrow_len * math.cos(wp["theta"])
        dy = -arrow_len * math.sin(wp["theta"])  # negative because pixel y is inverted
        ax.annotate("", xy=(px + dx, py + dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    zorder=4)

        # Node ID label
        if not args.no_labels:
            ax.annotate(f"{idx}", (px + 5, py - 5), fontsize=6, color=color,
                        fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor=color, alpha=0.8))

    # ── Room label annotations (placed at room centers) ──
    room_labels = [
        (-1.0, -0.3, "Hallway"),
        (1.5, -1.2, "Living\nRoom"),
        (6.5, -1.5, "Kitchen"),
        (-1.5, 1.0, "Bathroom"),
        (-4.5, 0.6, "Bedroom"),
        (-5.5, -1.5, "Fitness\nRoom"),
        (3.0, -3.5, "SE Room"),
    ]
    for rx, ry, rlabel in room_labels:
        rpx, rpy = world_to_pixel(rx, ry, map_height)
        ax.annotate(rlabel, (rpx, rpy), fontsize=11, fontweight="bold",
                    color="white", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#2c3e50",
                              edgecolor="white", alpha=0.85),
                    zorder=7)

    # ── Overlay captured poses (actual robot positions) ──
    if args.poses:
        poses_path = script_dir / "data" / "aws_house_graph" / "poses.json"
        if poses_path.exists():
            with open(poses_path) as f:
                poses = json.load(f)
            print(f"📊 Loaded {len(poses)} captured poses")

            for pose in poses:
                px, py = world_to_pixel(pose["x"], pose["y"], map_height)
                reached = pose.get("reached", True)

                marker = "^" if reached else "x"
                mcolor = "#00ff00" if reached else "#ff0000"
                msize = 6 if reached else 8

                ax.plot(px, py, marker, color=mcolor, markersize=msize,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=8)

                # Draw line from planned to actual position
                if "target_x" in pose and "target_y" in pose:
                    tpx, tpy = world_to_pixel(pose["target_x"], pose["target_y"], map_height)
                    ax.plot([tpx, px], [tpy, py], "-", color=mcolor, alpha=0.4, lw=0.8, zorder=3)
        else:
            print(f"⚠️  No poses.json found at {poses_path}")

    # ── Legend ──
    legend_items = [
        mpatches.Patch(color="#3498db", label="Hallway"),
        mpatches.Patch(color="#e74c3c", label="Living Room"),
        mpatches.Patch(color="#2ecc71", label="Kitchen"),
        mpatches.Patch(color="#9b59b6", label="Bathroom"),
        mpatches.Patch(color="#f39c12", label="Bedroom"),
        mpatches.Patch(color="#1abc9c", label="Fitness Room"),
        mpatches.Patch(color="#e91e63", label="SE Room"),
    ]
    if args.poses:
        legend_items.extend([
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#00ff00",
                       markersize=8, label="Reached (actual)"),
            plt.Line2D([0], [0], marker="x", color="w", markerfacecolor="#ff0000",
                       markersize=8, label="Failed (actual)"),
        ])

    ax.legend(handles=legend_items, loc="upper right", fontsize=9,
              framealpha=0.9, fancybox=True)

    # ── Stats ──
    n_waypoints = len(EXPLORATION_WAYPOINTS)
    ax.text(0.02, 0.02, f"Total waypoints: {n_waypoints}",
            transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Pixels (1 px = 0.05m)")
    ax.set_ylabel("Pixels")
    plt.tight_layout()

    # ── Save or show ──
    if args.save:
        save_path = script_dir / args.save
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"✅ Saved to: {save_path}")
    else:
        plt.show()
        print("✅ Plot displayed.")


if __name__ == "__main__":
    main()
