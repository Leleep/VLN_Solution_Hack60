#!/usr/bin/env python3
"""
Build Navigation Graph — Obstacle-Free Transit Edge Pre-computation
====================================================================
For every connected node pair in the exploration graph, runs A* on the
occupancy grid to find an obstacle-free path, then stores intermediate
"transit" waypoints in transit_edges.json.

During execution (execute_walk.py), the robot follows these transit
waypoints instead of navigating directly between nodes, preventing it
from getting stuck crossing walls or narrow doorways.

Also writes transit_graph_debug.png for visual inspection.

Usage:
    python scripts/build_nav_graph.py

Output:
    data/aws_house_graph/transit_edges.json
    data/transit_graph_debug.png
"""

import json
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lmnav.map_utils import load_map, world_to_pixel
from lmnav.nav_graph_builder import build_transit_edges


def main():
    print("=" * 60)
    print("🗺️  Navigation Graph Builder — Transit Edge Pre-computation")
    print("=" * 60)

    # ── Load config ───────────────────────────────────────────────────────────
    try:
        import yaml
        with open(PROJECT_ROOT / "config" / "pipeline_config.yaml") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Could not load config: {e}")
        sys.exit(1)

    map_yaml = (
        PROJECT_ROOT
        / "aws-robomaker-small-house-world"
        / "maps"
        / "turtlebot3_waffle_pi"
        / "map.yaml"
    )
    if not map_yaml.exists():
        print(f"❌ Map not found: {map_yaml}")
        sys.exit(1)

    poses_path = (
        PROJECT_ROOT
        / config["environment"]["graph_data_dir"]
        / "poses.json"
    )
    if not poses_path.exists():
        print(f"❌ poses.json not found: {poses_path}. Run explore_house.py first.")
        sys.exit(1)

    output_path = poses_path.parent / "transit_edges.json"
    debug_path  = PROJECT_ROOT / "data" / "transit_graph_debug.png"

    wp_cfg       = config.get("waypoints", {})
    edge_thresh  = config["graph"]["pose_edge_threshold_m"]
    robot_radius = config.get("robot", {}).get("radius_m", 0.30)
    spacing_m    = wp_cfg.get("transit_spacing_m", 0.40)

    # ── Load map ──────────────────────────────────────────────────────────────
    print("\n📍 Loading map...")
    map_data = load_map(str(map_yaml))
    print(f"   {map_data.width}×{map_data.height} px  |  "
          f"{map_data.resolution} m/px  |  "
          f"origin ({map_data.origin_x:.1f}, {map_data.origin_y:.1f})")

    # ── Load poses ────────────────────────────────────────────────────────────
    with open(poses_path) as f:
        poses_data = json.load(f)

    # Deduplicate: keep one representative per unique (x,y) position
    # so that 3-angle triplets don't generate 9 redundant edge entries
    seen_xy = {}
    compact_poses = []
    for p in poses_data:
        key = (round(p["x"], 2), round(p["y"], 2))
        if key not in seen_xy:
            seen_xy[key] = True
            compact_poses.append(p)

    print(f"   Loaded {len(poses_data)} nodes → {len(compact_poses)} unique positions")

    # ── Run A* for all connected pairs ────────────────────────────────────────
    print(f"\n🔍 Computing transit paths...")
    print(f"   Edge threshold: {edge_thresh}m | Robot radius: {robot_radius}m | "
          f"Spacing: {spacing_m}m")

    transit_result = build_transit_edges(
        map_data=map_data,
        poses_data=compact_poses,
        edge_threshold_m=edge_thresh,
        robot_radius_m=robot_radius,
        transit_spacing_m=spacing_m,
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(transit_result, f, indent=2)

    edges    = transit_result["edges"]
    blocked  = transit_result["blocked"]
    n_edges  = len(edges)
    n_blocked= len(blocked)
    n_transit= sum(len(v) for v in edges.values())
    print(f"\n💾 Saved transit_edges.json:")
    print(f"   Connected edges: {n_edges}")
    print(f"   Blocked edges:   {n_blocked}")
    print(f"   Transit waypts:  {n_transit}")

    # ── Debug visualization ───────────────────────────────────────────────────
    print("\n📊 Generating debug visualization...")
    map_img = np.array(__import__("PIL").Image.open(
        str(map_yaml).replace(".yaml", ".pgm")
    ))

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(map_img, cmap="gray", origin="upper", alpha=0.8)
    ax.set_title("Transit Graph — A* obstacle-free paths between nodes", fontsize=14)

    # Draw transit paths
    for key, wps in edges.items():
        a_id, b_id = key.split("-")
        pa = next((p for p in compact_poses if str(p["id"]) == a_id), None)
        pb = next((p for p in compact_poses if str(p["id"]) == b_id), None)
        if pa is None or pb is None:
            continue

        if len(wps) == 0:
            # Short direct edge — thin gray line
            pxa, pya = world_to_pixel(pa["x"], pa["y"], map_data)
            pxb, pyb = world_to_pixel(pb["x"], pb["y"], map_data)
            ax.plot([pxa, pxb], [pya, pyb], "-", color="gray", alpha=0.4, lw=1.0, zorder=2)
            continue

        # Build full path: start → transit → end
        full_path = [{"x": pa["x"], "y": pa["y"]}] + wps + [{"x": pb["x"], "y": pb["y"]}]
        xs = [world_to_pixel(w["x"], w["y"], map_data)[0] for w in full_path]
        ys = [world_to_pixel(w["x"], w["y"], map_data)[1] for w in full_path]
        ax.plot(xs, ys, "-", color="cyan", alpha=0.5, lw=1.2, zorder=2)

        # Transit nodes
        for wp in wps:
            px, py = world_to_pixel(wp["x"], wp["y"], map_data)
            ax.plot(px, py, ".", color="cyan", markersize=3, zorder=3)

    # Draw blocked edges
    for key in blocked:
        a_id, b_id = key.split("-")
        pa = next((p for p in compact_poses if str(p["id"]) == a_id), None)
        pb = next((p for p in compact_poses if str(p["id"]) == b_id), None)
        if pa and pb:
            pxa, pya = world_to_pixel(pa["x"], pa["y"], map_data)
            pxb, pyb = world_to_pixel(pb["x"], pb["y"], map_data)
            ax.plot([pxa, pxb], [pya, pyb], "--", color="red", alpha=0.8, lw=1.5, zorder=2)

    # Draw main nodes
    for p in compact_poses:
        px, py = world_to_pixel(p["x"], p["y"], map_data)
        ax.plot(px, py, "o", color="lime", markersize=8,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.annotate(str(p["id"]), (px + 4, py - 4), fontsize=6,
                    color="white", fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#222", alpha=0.7))

    # Legend
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], color="cyan",  lw=2, label="Transit path (A*)"),
        Line2D([0], [0], color="red",   lw=2, ls="--", label="Blocked edge"),
        Line2D([0], [0], marker="o", color="lime", markersize=8,
               lw=0, label="Graph node", markeredgecolor="white"),
        Line2D([0], [0], marker=".", color="cyan", markersize=6,
               lw=0, label="Transit waypoint"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.9)

    stats_txt = (f"Nodes: {len(compact_poses)}  |  "
                 f"Edges: {n_edges}  |  "
                 f"Blocked: {n_blocked}  |  "
                 f"Transit wpts: {n_transit}")
    ax.text(0.01, 0.01, stats_txt, transform=ax.transAxes,
            fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    debug_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(debug_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"   Debug image: {debug_path}")
    print()
    print("=" * 60)
    print("✅ Done! Transit graph built.")
    print("=" * 60)
    print(f"\nNext: run 'python scripts/execute_walk.py' — it will automatically")
    print(f"      use transit paths from: {output_path.name}")


if __name__ == "__main__":
    main()
