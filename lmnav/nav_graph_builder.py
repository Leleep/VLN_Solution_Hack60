"""
Navigation Graph Builder — A* Transit Path Generator
=====================================================
Pre-computes obstacle-free transit paths between every connected node pair
using A* on the occupancy grid. During execution, these replace direct
Nav2 long-range goals, preventing the robot from getting stuck crossing
walls or narrow doorways.

Architecture:
  Planning graph (Dijkstra DP)  ←  45 capture nodes, no change
  Execution transit edges        ←  this module: A* paths between pairs
"""

import heapq
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lmnav.map_utils import MapData, pixel_to_world, world_to_pixel


# ─── A* on Occupancy Grid ─────────────────────────────────────────────────────

def astar(
    free_mask: np.ndarray,
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    A* on a binary occupancy grid.
    free_mask: bool array of shape (H, W), True = navigable
    start_rc / end_rc: (row, col) integer positions
    Returns list of (row, col) from start to end, or None if unreachable.
    """
    H, W = free_mask.shape
    sr, sc = start_rc
    er, ec = end_rc

    if not (0 <= sr < H and 0 <= sc < W and 0 <= er < H and 0 <= ec < W):
        return None
    if not free_mask[sr, sc] or not free_mask[er, ec]:
        return None

    # g_scores as numpy array — much faster than dict on 500×500 grid
    g = np.full((H, W), np.inf, dtype=np.float32)
    g[sr, sc] = 0.0

    # Parent: encode (r, c) as r*W + c
    parent = np.full((H, W), -1, dtype=np.int32)

    heur = math.sqrt((er - sr) ** 2 + (ec - sc) ** 2)
    open_heap = [(heur, sr, sc)]

    # 8-connected moves: (dr, dc, cost)
    MOVES = [
        (-1, -1, 1.4142), (-1,  0, 1.0), (-1,  1, 1.4142),
        ( 0, -1, 1.0),                    ( 0,  1, 1.0),
        ( 1, -1, 1.4142), ( 1,  0, 1.0), ( 1,  1, 1.4142),
    ]

    while open_heap:
        f, r, c = heapq.heappop(open_heap)

        if r == er and c == ec:
            # Reconstruct path
            path = []
            cur_r, cur_c = er, ec
            while not (cur_r == sr and cur_c == sc):
                path.append((cur_r, cur_c))
                p = parent[cur_r, cur_c]
                cur_r, cur_c = divmod(p, W)
            path.append((sr, sc))
            path.reverse()
            return path

        cur_g = g[r, c]
        # Ignore stale heap entries
        if f > cur_g + math.sqrt((er - r) ** 2 + (ec - c) ** 2) + 1e-4:
            continue

        for dr, dc, cost in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not free_mask[nr, nc]:
                continue
            new_g = cur_g + cost
            if new_g < g[nr, nc]:
                g[nr, nc] = new_g
                parent[nr, nc] = r * W + c
                f_new = new_g + math.sqrt((er - nr) ** 2 + (ec - nc) ** 2)
                heapq.heappush(open_heap, (f_new, nr, nc))

    return None  # unreachable


# ─── Path Smoothing (line-of-sight reduction) ───────────────────────────────

def _line_of_sight(
    free_mask: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
) -> bool:
    """Rasterised line-of-sight check between two grid cells."""
    r1, c1 = p1
    r2, c2 = p2
    n = max(abs(r2 - r1), abs(c2 - c1))
    if n == 0:
        return True
    rs = np.round(np.linspace(r1, r2, n + 1)).astype(int)
    cs = np.round(np.linspace(c1, c2, n + 1)).astype(int)
    return bool(np.all(free_mask[rs, cs]))


def smooth_path(
    path: List[Tuple[int, int]],
    free_mask: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Greedy line-of-sight reduction: keep only turns where direct LOS is blocked.
    Reduces 500-point A* paths to 5-10 key vertices.
    """
    if len(path) <= 2:
        return path

    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Find the furthest reachable point with clear LOS
        j = len(path) - 1
        while j > i + 1:
            if _line_of_sight(free_mask, path[i], path[j]):
                break
            j -= 1
        result.append(path[j])
        i = j
    return result


# ─── World-Space Resampling (correct interpolating approach) ─────────────────

def sample_path_world(
    path_rc: List[Tuple[int, int]],
    spacing_m: float,
    map_data: MapData,
) -> List[Dict]:
    """
    Resample a pixel path to world-space waypoints at exact spacing_m intervals.
    Uses linear interpolation along each segment so short smoothed paths (2 pts)
    still generate interior transit waypoints.

    Returns list of {"x": ..., "y": ...} — includes start and end.
    """
    if len(path_rc) < 2:
        x, y = pixel_to_world(path_rc[0][1], path_rc[0][0], map_data)
        return [{"x": round(x, 3), "y": round(y, 3)}]

    spacing_px = spacing_m / map_data.resolution
    result = []

    def _add(r: float, c: float):
        x, y = pixel_to_world(int(round(c)), int(round(r)), map_data)
        result.append({"x": round(x, 3), "y": round(y, 3)})

    # Always include start
    _add(*path_rc[0])

    accum = 0.0  # distance accumulated since last emitted waypoint
    prev_r, prev_c = float(path_rc[0][0]), float(path_rc[0][1])

    for curr in path_rc[1:]:
        cr, cc = float(curr[0]), float(curr[1])
        seg_len = math.sqrt((cr - prev_r) ** 2 + (cc - prev_c) ** 2)
        if seg_len == 0:
            continue

        # How far along this segment until we first hit a sample point
        dist_to_next = spacing_px - accum

        while dist_to_next <= seg_len:
            # Interpolate exactly
            t = dist_to_next / seg_len
            itp_r = prev_r + t * (cr - prev_r)
            itp_c = prev_c + t * (cc - prev_c)
            _add(itp_r, itp_c)
            dist_to_next += spacing_px
            seg_len -= (dist_to_next - spacing_px)
            prev_r += t * (cr - prev_r)
            prev_c += t * (cc - prev_c)
            dist_to_next = spacing_px  # reset for next point
            # Recompute remaining segment
            seg_len = math.sqrt((cr - prev_r) ** 2 + (cc - prev_c) ** 2)
            dist_to_next = spacing_px
            accum = 0.0

        accum += seg_len
        prev_r, prev_c = cr, cc

    # Always include exact endpoint
    x_end, y_end = pixel_to_world(path_rc[-1][1], path_rc[-1][0], map_data)
    end_pt = {"x": round(x_end, 3), "y": round(y_end, 3)}
    if not result or result[-1] != end_pt:
        result.append(end_pt)

    return result


def _resample_segment(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    spacing_m: float,
) -> List[Tuple[float, float]]:
    """
    Simple linear resampler in world-space for a single segment.
    Returns list of (x, y) points at spacing_m intervals, NOT including p1.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    l = math.sqrt(dx * dx + dy * dy)
    if l < 1e-6:
        return []
    n = max(1, int(l / spacing_m))
    pts = []
    for k in range(1, n + 1):
        t = k / n
        pts.append((p1[0] + t * dx, p1[1] + t * dy))
    return pts


def sample_smooth_path_world(
    smoothed_path: List[Tuple[int, int]],
    spacing_m: float,
    map_data: MapData,
) -> List[Dict]:
    """
    World-space resampling of a smoothed (key-vertices) pixel path.
    Converts each key vertex to world coords, then resamples the polyline.
    """
    # Convert pixels to world
    world_pts = []
    for (r, c) in smoothed_path:
        x, y = pixel_to_world(c, r, map_data)
        world_pts.append((x, y))

    if len(world_pts) < 2:
        return [{"x": round(world_pts[0][0], 3), "y": round(world_pts[0][1], 3)}]

    result = [{"x": round(world_pts[0][0], 3), "y": round(world_pts[0][1], 3)}]

    for i in range(len(world_pts) - 1):
        pts = _resample_segment(world_pts[i], world_pts[i + 1], spacing_m)
        for x, y in pts:
            result.append({"x": round(x, 3), "y": round(y, 3)})

    # Deduplicate adjacent duplicates
    deduped = [result[0]]
    for pt in result[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)

    return deduped


# ─── Main Builder ────────────────────────────────────────────────────────────

def build_transit_edges(
    map_data: MapData,
    poses_data: List[Dict],
    edge_threshold_m: float = 3.0,
    robot_radius_m: float = 0.30,
    transit_spacing_m: float = 0.40,
) -> Dict:
    """
    Build transit waypoints for every connected node pair.

    Returns a dict:
    {
        "edges": {
            "A-B": [{"x":..,"y":..}, …],  # transit waypoints (may be empty = direct short path)
        },
        "blocked": ["C-D", ...]           # edges with no viable A* path
    }
    """
    from scipy import ndimage

    # Erode the free mask by robot radius
    free_mask = map_data.free_mask.astype(bool)
    erode_px      = max(1, round(robot_radius_m        / map_data.resolution))
    erode_half_px = max(1, round(robot_radius_m * 0.5  / map_data.resolution))

    eroded       = ndimage.binary_erosion(free_mask, iterations=erode_px)
    eroded_half  = ndimage.binary_erosion(free_mask, iterations=erode_half_px)

    n = len(poses_data)
    edges: Dict[str, List[Dict]] = {}
    blocked: List[str] = []

    total_pairs = 0
    found_paths = 0
    t0 = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            pi = poses_data[i]
            pj = poses_data[j]

            dx = pi["x"] - pj["x"]
            dy = pi["y"] - pj["y"]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > edge_threshold_m:
                continue

            total_pairs += 1
            key = f"{pi['id']}-{pj['id']}"

            si = world_to_pixel(pi["x"], pi["y"], map_data)  # (col, row)
            sj = world_to_pixel(pj["x"], pj["y"], map_data)

            # A* with full erosion
            path = astar(eroded, (si[1], si[0]), (sj[1], sj[0]))

            # Fall back to half-erosion in narrow corridors
            if path is None:
                path = astar(eroded_half, (si[1], si[0]), (sj[1], sj[0]))

            if path is None:
                blocked.append(key)
                continue

            found_paths += 1

            # Smooth path (remove redundant intermediate pixels)
            smoothed = smooth_path(path, eroded_half)

            # Resample at regular world-space intervals
            waypoints = sample_smooth_path_world(smoothed, transit_spacing_m, map_data)

            # Strip exact start and end (they are the node positions themselves)
            transit_only = waypoints[1:-1]
            edges[key] = transit_only

    elapsed = time.time() - t0
    n_transit = sum(len(v) for v in edges.values())
    print(f"   Transit edges built: {found_paths}/{total_pairs} paths found, "
          f"{len(blocked)} blocked, {n_transit} transit waypoints  [{elapsed:.1f}s]")
    if blocked:
        print(f"   ⚠️  Blocked (no clear path): {blocked}")

    return {"edges": edges, "blocked": blocked}
