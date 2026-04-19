#!/usr/bin/env python3
"""
Automatic Waypoint Generator — Distance Transform Local Maxima
================================================================
Generates exploration waypoints from the occupancy grid map using
the medial axis / distance transform approach.

Algorithm:
  1. Load map.pgm → threshold to binary free-space mask
  2. Erode by robot radius + safety buffer
  3. Euclidean distance transform (each pixel = distance to nearest wall)
  4. Find local maxima → geometric room centers with maximum clearance
  5. Filter by minimum clearance and minimum inter-node spacing
  6. Convert to world coordinates
  7. Generate multiple viewing angles per position

Usage:
  python scripts/generate_waypoints.py
  python scripts/generate_waypoints.py --config config/pipeline_config.yaml
  python scripts/generate_waypoints.py --map-yaml path/to/map.yaml

Output:
  data/aws_house_graph/chamber_nodes.json
  data/waypoint_debug.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    from scipy import ndimage
except ImportError:
    print("❌ scipy is required: pip install scipy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available — debug image will be skipped")

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lmnav.map_utils import (
    load_map,
    world_to_pixel,
    pixel_to_world,
    is_free,
    validate_spawn,
    meters_to_pixels,
)


def load_config(config_path: str = None) -> dict:
    """Load pipeline config, falling back to defaults."""
    defaults = {
        "robot": {"spawn_x": -2.0, "spawn_y": -0.5, "spawn_theta": 0.0},
        "waypoints": {
            "min_clearance_m": 0.50,
            "min_node_spacing_m": 1.20,
            "local_max_window_m": 0.80,
            "capture_angles_deg": [0, 120, 240],
        },
        "map": {
            "yaml_path": "aws-robomaker-small-house-world/maps/turtlebot3_waffle_pi/map.yaml",
        },
    }

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        # Merge — config values override defaults
        for section in defaults:
            if section in cfg:
                defaults[section].update(cfg[section])

    return defaults


def generate_waypoints(
    map_yaml_path: str,
    min_clearance_m: float = 0.50,
    min_node_spacing_m: float = 1.20,
    local_max_window_m: float = 0.80,
    robot_radius_m: float = 0.30,
    capture_angles_deg: list = None,
) -> list:
    """
    Generate exploration waypoints from the occupancy grid map.

    Args:
        map_yaml_path: Path to map.yaml
        min_clearance_m: Minimum distance-transform value for a valid node (meters)
        min_node_spacing_m: Minimum distance between any two nodes (meters)
        local_max_window_m: Suppression window for local maxima detection (meters)
        robot_radius_m: Robot inscribed radius + safety buffer (meters)
        capture_angles_deg: List of rotation angles (degrees) per position

    Returns:
        List of waypoint dicts: [{"x", "y", "theta", "label", "clearance_m"}, ...]
    """
    if capture_angles_deg is None:
        capture_angles_deg = [0, 120, 240]

    # ── Step 1: Load map ──
    print("━" * 50)
    print("Step 1: Loading occupancy grid map...")
    print("━" * 50)

    map_data = load_map(map_yaml_path)
    free_mask = map_data.free_mask
    print(f"   Map: {map_data.width}×{map_data.height} pixels")
    print(f"   Resolution: {map_data.resolution} m/px")
    print(f"   Origin: ({map_data.origin_x}, {map_data.origin_y})")
    print(f"   Free cells: {np.sum(free_mask)} / {free_mask.size} "
          f"({100 * np.sum(free_mask) / free_mask.size:.1f}%)")

    # ── Step 2: Erode by robot radius ──
    print("\nStep 2: Eroding free space by robot radius...")
    erode_px = meters_to_pixels(robot_radius_m, map_data)
    print(f"   Robot radius + buffer: {robot_radius_m}m = {erode_px}px")

    # Create structuring element (disk)
    struct = ndimage.generate_binary_structure(2, 1)  # cross
    eroded = ndimage.binary_erosion(free_mask, iterations=erode_px)
    print(f"   Eroded free cells: {np.sum(eroded)} "
          f"(removed {np.sum(free_mask) - np.sum(eroded)} near walls)")

    # ── Step 3: Distance Transform ──
    print("\nStep 3: Computing Euclidean distance transform...")
    dist_transform = ndimage.distance_transform_edt(eroded)
    max_dist_px = np.max(dist_transform)
    max_dist_m = max_dist_px * map_data.resolution
    print(f"   Max clearance: {max_dist_m:.2f}m ({max_dist_px:.0f}px)")

    # ── Step 4: Find local maxima ──
    print("\nStep 4: Finding local maxima (room centers)...")
    window_px = meters_to_pixels(local_max_window_m, map_data)
    # Ensure window is odd
    if window_px % 2 == 0:
        window_px += 1
    print(f"   Suppression window: {local_max_window_m}m = {window_px}px")

    # Local maximum filter
    local_max = ndimage.maximum_filter(dist_transform, size=window_px)
    # A pixel is a local maximum if its value equals the filtered value
    # AND it's above the minimum clearance threshold
    min_clearance_px = min_clearance_m / map_data.resolution
    maxima_mask = (
        (dist_transform == local_max) &
        (dist_transform > min_clearance_px) &
        (eroded)  # Must be in eroded free space
    )

    # Extract maxima positions
    maxima_rows, maxima_cols = np.where(maxima_mask)
    maxima_clearance = dist_transform[maxima_rows, maxima_cols]
    print(f"   Raw local maxima: {len(maxima_rows)}")

    # ── Step 5: Filter by minimum spacing ──
    print("\nStep 5: Filtering by minimum node spacing...")
    min_spacing_px = min_node_spacing_m / map_data.resolution

    # Sort by clearance (highest first — keep the best located ones)
    sort_idx = np.argsort(-maxima_clearance)
    maxima_rows = maxima_rows[sort_idx]
    maxima_cols = maxima_cols[sort_idx]
    maxima_clearance = maxima_clearance[sort_idx]

    # Greedy selection: keep a maximum if no previously kept maximum is too close
    kept_indices = []
    kept_positions = []

    for i in range(len(maxima_rows)):
        pos = np.array([maxima_cols[i], maxima_rows[i]], dtype=np.float64)
        too_close = False
        for kept_pos in kept_positions:
            if np.linalg.norm(pos - kept_pos) < min_spacing_px:
                too_close = True
                break
        if not too_close:
            kept_indices.append(i)
            kept_positions.append(pos)

    print(f"   After spacing filter ({min_node_spacing_m}m): {len(kept_indices)} positions")

    # ── Step 6: Convert to world coordinates ──
    print("\nStep 6: Converting to world coordinates...")
    waypoints = []
    node_counter = 0

    for idx in kept_indices:
        col = int(maxima_cols[idx])
        row = int(maxima_rows[idx])
        clearance_m = float(maxima_clearance[idx]) * map_data.resolution

        wx, wy = pixel_to_world(col, row, map_data)

        # Generate multiple viewing angles
        for angle_deg in capture_angles_deg:
            theta = math.radians(angle_deg)
            waypoints.append({
                "x": round(wx, 3),
                "y": round(wy, 3),
                "theta": round(theta, 4),
                "label": f"auto_{node_counter:03d}",
                "clearance_m": round(clearance_m, 3),
                "parent_position": f"({wx:.2f}, {wy:.2f})",
            })
            node_counter += 1

    print(f"   Total waypoints: {len(waypoints)} "
          f"({len(kept_indices)} positions × {len(capture_angles_deg)} angles)")

    return waypoints


def generate_debug_image(
    map_yaml_path: str,
    waypoints: list,
    output_path: str,
    spawn_x: float = -2.0,
    spawn_y: float = -0.5,
):
    """Generate a debug visualization of the waypoint generation."""
    if not HAS_MATPLOTLIB:
        print("⚠️  Skipping debug image (matplotlib not available)")
        return

    map_data = load_map(map_yaml_path)
    eroded = ndimage.binary_erosion(map_data.free_mask, iterations=6)
    dist_transform = ndimage.distance_transform_edt(eroded)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: Distance transform heatmap
    ax1 = axes[0]
    im = ax1.imshow(dist_transform, cmap="inferno", origin="upper")
    ax1.set_title("Distance Transform (Euclidean)", fontsize=14)
    plt.colorbar(im, ax=ax1, label="Distance to wall (pixels)")

    # Right: Map with waypoints
    ax2 = axes[1]
    ax2.imshow(map_data.image, cmap="gray", origin="upper")
    ax2.set_title("Generated Waypoints on Map", fontsize=14)

    # Plot unique positions (not per-angle)
    seen_positions = set()
    for wp in waypoints:
        pos_key = wp["parent_position"]
        if pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)

        col, row = world_to_pixel(wp["x"], wp["y"], map_data)
        clearance = wp["clearance_m"]

        # Color by clearance
        color = plt.cm.viridis(min(clearance / 3.0, 1.0))
        ax2.plot(col, row, "o", color=color, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        # Draw all viewing angle arrows for this position
        for wp2 in waypoints:
            if wp2["parent_position"] == pos_key:
                dx = 8 * math.cos(wp2["theta"])
                dy = -8 * math.sin(wp2["theta"])
                ax2.annotate("", xy=(col + dx, row + dy), xytext=(col, row),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                            zorder=4)

    # Plot spawn position
    spawn_col, spawn_row = world_to_pixel(spawn_x, spawn_y, map_data)
    ax2.plot(spawn_col, spawn_row, "*", color="red", markersize=15,
             markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax2.annotate("SPAWN", (spawn_col, spawn_row), fontsize=9,
                 color="red", fontweight="bold",
                 xytext=(5, -10), textcoords="offset points", zorder=6)

    # Stats
    n_positions = len(seen_positions)
    n_total = len(waypoints)
    ax2.text(0.02, 0.98,
             f"Positions: {n_positions}\nTotal nodes: {n_total}",
             transform=ax2.transAxes, fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Debug image saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate exploration waypoints from occupancy grid map"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to pipeline_config.yaml",
    )
    parser.add_argument(
        "--map-yaml", type=str, default=None,
        help="Path to map.yaml (overrides config)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for chamber_nodes.json",
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config or str(project_root / "config" / "pipeline_config.yaml")
    config = load_config(config_path)

    # Resolve map path
    map_yaml_path = args.map_yaml
    if not map_yaml_path:
        map_yaml_path = str(project_root / config["map"]["yaml_path"])

    # Resolve output path
    output_dir = project_root / "data" / "aws_house_graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(output_dir / "chamber_nodes.json")

    spawn_x = config["robot"]["spawn_x"]
    spawn_y = config["robot"]["spawn_y"]
    wp_cfg = config["waypoints"]

    print("=" * 60)
    print("🗺️  LM-Nav Waypoint Generator")
    print("=" * 60)
    print(f"   Map: {map_yaml_path}")
    print(f"   Config: {config_path}")
    print(f"   Output: {output_path}")
    print()

    # ── Validate spawn ──
    map_data = load_map(map_yaml_path)
    validate_spawn(spawn_x, spawn_y, map_data)
    print()

    # ── Generate waypoints ──
    waypoints = generate_waypoints(
        map_yaml_path=map_yaml_path,
        min_clearance_m=wp_cfg["min_clearance_m"],
        min_node_spacing_m=wp_cfg["min_node_spacing_m"],
        local_max_window_m=wp_cfg["local_max_window_m"],
        capture_angles_deg=wp_cfg.get("capture_angles_deg", [0, 120, 240]),
    )

    if not waypoints:
        print("❌ No waypoints generated! Check map and config parameters.")
        sys.exit(1)

    # ── Save ──
    with open(output_path, "w") as f:
        json.dump(waypoints, f, indent=2)
    print(f"\n💾 Saved {len(waypoints)} waypoints to: {output_path}")

    # ── Debug image ──
    debug_path = str(project_root / "data" / "waypoint_debug.png")
    generate_debug_image(
        map_yaml_path=map_yaml_path,
        waypoints=waypoints,
        output_path=debug_path,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )

    print()
    print("=" * 60)
    print("✅ Waypoint generation complete!")
    print("=" * 60)
    print(f"   Waypoints: {len(waypoints)}")
    print(f"   File: {output_path}")
    print(f"   Debug: {debug_path}")
    print()
    print("Next: run 'python scripts/explore_house.py' to collect images at these waypoints.")


if __name__ == "__main__":
    main()
