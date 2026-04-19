#!/usr/bin/env python3
"""
Visualize Exploration Nodes on House Map
=========================================
Loads actual captured poses from poses.json and shows them overlaid on
the occupancy grid map. Also draws transit-edge paths if transit_edges.json
is available.

Usage:
    python scripts/visualize_nodes.py                   # Show all nodes on map
    python scripts/visualize_nodes.py --transit         # Also draw transit paths
    python scripts/visualize_nodes.py --save map_viz.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    print("❌ Pillow required: pip install Pillow")
    sys.exit(1)

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lmnav.map_utils import load_map, world_to_pixel

# ── Map configuration ─────────────────────────────────────────────────────────
MAP_YAML = (
    PROJECT_ROOT
    / "aws-robomaker-small-house-world"
    / "maps"
    / "turtlebot3_waffle_pi"
    / "map.yaml"
)
GRAPH_DIR = PROJECT_ROOT / "data" / "aws_house_graph"


def main():
    parser = argparse.ArgumentParser(description="Visualize exploration nodes on house map")
    parser.add_argument("--transit",   action="store_true",
                        help="Draw A* transit paths between nodes")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide node ID labels")
    parser.add_argument("--save",      type=str, default=None,
                        help="Save to file instead of showing interactively")
    args = parser.parse_args()

    # ── Validate paths ────────────────────────────────────────────────────────
    if not MAP_YAML.exists():
        print(f"❌ Map not found: {MAP_YAML}")
        sys.exit(1)

    poses_path = GRAPH_DIR / "poses.json"
    if not poses_path.exists():
        print(f"❌ poses.json not found: {poses_path}")
        print("   Run 'python scripts/explore_house.py' first.")
        sys.exit(1)

    # ── Load map ──────────────────────────────────────────────────────────────
    map_data = load_map(str(MAP_YAML))
    map_img  = np.array(PILImage.open(
        str(MAP_YAML).replace(".yaml", ".pgm")
    ))
    print(f"📍 Map: {map_data.width}×{map_data.height} px, "
          f"{map_data.resolution} m/px, "
          f"origin ({map_data.origin_x:.1f}, {map_data.origin_y:.1f})")

    # ── Load poses ────────────────────────────────────────────────────────────
    with open(poses_path) as f:
        poses = json.load(f)
    print(f"📊 Loaded {len(poses)} captured poses")

    # ── Load transit edges (optional) ─────────────────────────────────────────
    transit_edges = {}
    blocked_edges = []
    transit_path  = GRAPH_DIR / "transit_edges.json"
    if args.transit and transit_path.exists():
        with open(transit_path) as f:
            td = json.load(f)
        transit_edges = td.get("edges", {})
        blocked_edges = td.get("blocked", [])
        n_transit = sum(len(v) for v in transit_edges.values())
        print(f"🗺️  Transit graph: {len(transit_edges)} edges, "
              f"{n_transit} transit waypoints, "
              f"{len(blocked_edges)} blocked")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.imshow(map_img, cmap="gray", origin="upper", alpha=0.85)
    ax.set_title(
        f"LM-Nav Exploration — AWS Small House\n"
        f"{len(poses)} nodes at {len(set((round(p['x'],1),round(p['y'],1)) for p in poses))} "
        f"unique positions",
        fontsize=14, fontweight="bold"
    )

    # ── Draw transit paths ────────────────────────────────────────────────────
    if args.transit:
        # Gather node positions by ID
        pose_by_id = {p["id"]: p for p in poses}

        for key, wps in transit_edges.items():
            a_id, b_id = int(key.split("-")[0]), int(key.split("-")[1])
            pa = pose_by_id.get(a_id)
            pb = pose_by_id.get(b_id)
            if pa is None or pb is None:
                continue

            full_path = [{"x": pa["x"], "y": pa["y"]}] + wps + [{"x": pb["x"], "y": pb["y"]}]
            xs = [world_to_pixel(w["x"], w["y"], map_data)[0] for w in full_path]
            ys = [world_to_pixel(w["x"], w["y"], map_data)[1] for w in full_path]
            ax.plot(xs, ys, "-", color="deepskyblue", alpha=0.55, lw=1.5, zorder=2)

            for wp in wps:
                px, py = world_to_pixel(wp["x"], wp["y"], map_data)
                ax.plot(px, py, ".", color="deepskyblue", markersize=4, zorder=3)

        for key in blocked_edges:
            a_id, b_id = int(key.split("-")[0]), int(key.split("-")[1])
            pa = pose_by_id.get(a_id)
            pb = pose_by_id.get(b_id)
            if pa and pb:
                ax.plot(
                    *zip(world_to_pixel(pa["x"], pa["y"], map_data),
                         world_to_pixel(pb["x"], pb["y"], map_data)),
                    "--", color="red", alpha=0.8, lw=1.5, zorder=2
                )

    # ── Group poses by unique position ────────────────────────────────────────
    from collections import defaultdict
    groups = defaultdict(list)
    for p in poses:
        key = (round(p["x"], 1), round(p["y"], 1))
        groups[key].append(p)

    arrow_len = 14  # pixels

    # ── Draw each unique position ─────────────────────────────────────────────
    position_cm = plt.cm.get_cmap("tab20", len(groups))
    for color_idx, ((px_world, py_world), group) in enumerate(groups.items()):
        color = position_cm(color_idx)
        representative = group[0]
        px, py = world_to_pixel(representative["x"], representative["y"], map_data)

        # Draw position circle
        ax.plot(px, py, "o", color=color, markersize=11,
                markeredgecolor="white", markeredgewidth=1.8, zorder=5)

        # Draw orientation arrows for each angle
        for p in group:
            theta = p.get("theta", 0)
            dx_arr =  arrow_len * math.cos(theta)
            dy_arr = -arrow_len * math.sin(theta)  # y-axis flipped in image
            apx, apy = world_to_pixel(p["x"], p["y"], map_data)
            ax.annotate("", xy=(apx + dx_arr, apy + dy_arr), xytext=(apx, apy),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.8),
                        zorder=4)

        # Node ID label (show the first node's ID + count of angles)
        if not args.no_labels:
            label = str(representative["id"])
            if len(group) > 1:
                label += f"+{len(group)-1}"
            ax.annotate(label, (px + 5, py - 5), fontsize=6, color="white",
                        fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a1a2e",
                                  edgecolor=color, alpha=0.85))

        # Mark failed nodes with an X
        if not representative.get("reached", True):
            ax.plot(px, py, "x", color="red", markersize=12,
                    markeredgewidth=2.5, zorder=7)

    # ── Room label annotations ─────────────────────────────────────────────────
    ROOM_LABELS = [
        (-1.0, -0.3, "Hallway"),
        (1.5,  -1.2, "Living Room"),
        (6.5,  -1.5, "Kitchen"),
        (-1.5,  1.0, "Bathroom"),
        (-4.5,  0.6, "Bedroom"),
        (-5.5, -1.5, "Fitness Room"),
        (3.0,  -3.5, "SE Room"),
    ]
    for rx, ry, rlabel in ROOM_LABELS:
        rpx, rpy = world_to_pixel(rx, ry, map_data)
        if 0 <= rpx < map_data.width and 0 <= rpy < map_data.height:
            ax.annotate(rlabel, (rpx, rpy), fontsize=11, fontweight="bold",
                        color="white", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2c3e50",
                                  edgecolor="white", alpha=0.80),
                        zorder=8)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
               markersize=10, label=f"Reached node ({len(poses)} total)"),
        Line2D([0], [0], marker="x", color="red", markersize=8,
               markeredgewidth=2, lw=0, label="Failed navigation"),
    ]
    if args.transit:
        legend_items += [
            Line2D([0], [0], color="deepskyblue", lw=2, label="Transit path (A*)"),
            Line2D([0], [0], marker=".", color="deepskyblue", markersize=6,
                   lw=0, label="Transit waypoint"),
            Line2D([0], [0], color="red", lw=2, ls="--", label="Blocked edge"),
        ]

    ax.legend(handles=legend_items, loc="upper right", fontsize=9, framealpha=0.92)

    # ── Stats box ─────────────────────────────────────────────────────────────
    n_unique = len(groups)
    n_reached = sum(1 for p in poses if p.get("reached", True))
    stats = (f"Captured: {len(poses)}  |  Unique positions: {n_unique}  |  "
             f"Reached: {n_reached}/{len(poses)}")
    if args.transit:
        n_transit = sum(len(v) for v in transit_edges.values())
        stats += f"  |  Transit wpts: {n_transit}"

    ax.text(0.01, 0.01, stats, transform=ax.transAxes,
            fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    ax.set_xlabel("Pixels (1 px = 0.05 m)")
    ax.set_ylabel("Pixels")
    plt.tight_layout()

    # ── Output ────────────────────────────────────────────────────────────────
    if args.save:
        save_path = PROJECT_ROOT / args.save
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"✅ Saved to: {save_path}")
    else:
        plt.show()
        print("✅ Plot displayed.")


if __name__ == "__main__":
    main()
