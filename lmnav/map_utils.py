"""
Map Utilities — Coordinate Transforms & Occupancy Queries
==========================================================
Handles the critical pixel↔world coordinate conversions and
occupancy grid queries used throughout the pipeline.

Coordinate Chain:
    Gazebo world frame ←→ ROS map frame ←→ map.pgm pixel coordinates

The key subtlety: map.pgm row 0 is at the TOP of the image (image convention),
but in the ROS map frame Y increases UPWARD. The Y-axis must be flipped.

Given map.yaml fields:
    origin: [ox, oy, otheta]
    resolution: r  (meters per pixel)

And image dimensions: width × height (pixels)

    Pixel (col, row) → World (x, y):
        x = ox + col * r
        y = oy + (height - 1 - row) * r     ← row is flipped

    World (x, y) → Pixel (col, row):
        col = round((x - ox) / r)
        row = round(height - 1 - (y - oy) / r)  ← y is flipped back
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")


@dataclass
class MapData:
    """Container for all map-related data."""
    image: np.ndarray          # Raw PGM pixel values (0-255)
    resolution: float          # Meters per pixel
    origin_x: float            # World X at pixel column 0
    origin_y: float            # World Y at pixel row (height-1) [bottom of image]
    origin_theta: float        # Map rotation (almost always 0.0)
    negate: bool               # If True, pixel values are inverted
    occupied_thresh: float     # Occupancy probability threshold
    free_thresh: float         # Free-space probability threshold
    width: int                 # Image width in pixels
    height: int                # Image height in pixels

    @property
    def free_mask(self) -> np.ndarray:
        """
        Binary mask: True where the map is free space, False elsewhere.

        Handles the `negate` field from map.yaml:
        - negate=0 (default): white (255) = free, black (0) = occupied
        - negate=1: black (0) = free, white (255) = occupied

        Thresholding logic (from ROS map_server docs):
        - Probability p = (255 - pixel) / 255  when negate=0
        - Probability p = pixel / 255          when negate=1
        - Free if p < free_thresh
        - Occupied if p > occupied_thresh
        - Unknown otherwise → treated as occupied for safety
        """
        if self.negate:
            prob = self.image.astype(np.float64) / 255.0
        else:
            prob = (255 - self.image.astype(np.float64)) / 255.0

        # Free where probability is below free_thresh
        return prob < self.free_thresh


def load_map(yaml_path: str) -> MapData:
    """
    Load an occupancy grid map from a map.yaml + map.pgm pair.

    Args:
        yaml_path: Path to the map.yaml file

    Returns:
        MapData with all map parameters and the occupancy image
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Map YAML not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    # Load the PGM image
    pgm_path = yaml_path.parent / meta["image"]
    if not pgm_path.exists():
        raise FileNotFoundError(f"Map PGM not found: {pgm_path}")

    img = np.array(Image.open(pgm_path))
    h, w = img.shape[:2]

    origin = meta["origin"]
    if len(origin) >= 3 and abs(origin[2]) > 1e-6:
        raise ValueError(
            f"Non-zero map origin rotation ({origin[2]} rad) is not supported. "
            f"Generate a map with theta=0 in the origin."
        )

    return MapData(
        image=img,
        resolution=meta["resolution"],
        origin_x=origin[0],
        origin_y=origin[1],
        origin_theta=origin[2] if len(origin) >= 3 else 0.0,
        negate=bool(meta.get("negate", 0)),
        occupied_thresh=meta.get("occupied_thresh", 0.65),
        free_thresh=meta.get("free_thresh", 0.196),
        width=w,
        height=h,
    )


def world_to_pixel(x: float, y: float, map_data: MapData) -> Tuple[int, int]:
    """
    Convert world coordinates (meters) to pixel coordinates (col, row).

    Returns:
        (col, row) — integer pixel coordinates. col is horizontal (X),
                     row is vertical (Y, top-of-image = row 0).
    """
    col = round((x - map_data.origin_x) / map_data.resolution)
    row = round(map_data.height - 1 - (y - map_data.origin_y) / map_data.resolution)
    return int(col), int(row)


def pixel_to_world(col: int, row: int, map_data: MapData) -> Tuple[float, float]:
    """
    Convert pixel coordinates (col, row) to world coordinates (x, y).

    Args:
        col: Pixel column (horizontal, 0 = left)
        row: Pixel row (vertical, 0 = top)

    Returns:
        (x, y) — world coordinates in meters (map frame)
    """
    x = map_data.origin_x + col * map_data.resolution
    y = map_data.origin_y + (map_data.height - 1 - row) * map_data.resolution
    return x, y


def is_in_bounds(col: int, row: int, map_data: MapData) -> bool:
    """Check if pixel coordinates are within the map image bounds."""
    return 0 <= col < map_data.width and 0 <= row < map_data.height


def is_free(x: float, y: float, map_data: MapData) -> bool:
    """
    Check if a world coordinate is in free space on the occupancy grid.

    Returns:
        True if the position is free (navigable), False if occupied/unknown/OOB.
    """
    col, row = world_to_pixel(x, y, map_data)
    if not is_in_bounds(col, row, map_data):
        return False
    return bool(map_data.free_mask[row, col])


def validate_spawn(
    spawn_x: float, spawn_y: float, map_data: MapData
) -> bool:
    """
    Validate that the robot's spawn position is in free space.

    Prints diagnostic info and returns True/False.
    """
    col, row = world_to_pixel(spawn_x, spawn_y, map_data)
    in_bounds = is_in_bounds(col, row, map_data)
    free = is_free(spawn_x, spawn_y, map_data) if in_bounds else False

    pixel_val = map_data.image[row, col] if in_bounds else -1

    print(f"🔍 Spawn validation:")
    print(f"   World:  ({spawn_x:.2f}, {spawn_y:.2f})")
    print(f"   Pixel:  ({col}, {row})")
    print(f"   Bounds: {'✅ in bounds' if in_bounds else '❌ OUT OF BOUNDS'}")
    if in_bounds:
        print(f"   Value:  {pixel_val} (negate={map_data.negate})")
        print(f"   Free:   {'✅ free space' if free else '❌ OCCUPIED/UNKNOWN'}")

    if not free:
        print(
            f"\n⚠️  WARNING: Spawn position ({spawn_x}, {spawn_y}) maps to "
            f"non-free space in the occupancy grid! Navigation will likely fail.\n"
            f"   Check that map.yaml origin and the Gazebo spawn pose are consistent."
        )

    return free


def meters_to_pixels(meters: float, map_data: MapData) -> int:
    """Convert a distance in meters to pixels using the map resolution."""
    return max(1, round(meters / map_data.resolution))


def load_poses(poses_path: str) -> list:
    """Load poses.json and return the list of pose dicts."""
    with open(poses_path, "r") as f:
        return json.load(f)


def find_nearest_node(
    x: float, y: float, poses_data: list
) -> Tuple[int, float]:
    """
    Find the graph node nearest to (x, y).

    Args:
        x, y: World coordinates (map frame)
        poses_data: List of dicts with 'id', 'x', 'y' keys

    Returns:
        (node_id, distance_m) — ID and distance to the nearest node
    """
    best_id = poses_data[0]["id"]
    best_dist = float("inf")

    for p in poses_data:
        dist = np.sqrt((p["x"] - x) ** 2 + (p["y"] - y) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_id = p["id"]

    return best_id, best_dist
