#!/usr/bin/env python3
"""
LM-Nav Full Pipeline
====================
End-to-end: instruction → landmarks → CLIP scoring → Dijkstra walk → visualization.

This runs entirely offline — no Gazebo needed (uses pre-collected exploration data).

Usage:
  python scripts/run_pipeline.py \
      --instruction "Go to the kitchen and find the refrigerator" \
      --start-node 0

  python scripts/run_pipeline.py \
      --instruction "Walk to the bedroom and find the bed" \
      --config config/pipeline_config.yaml

Output:
  - Planned walk (printed to console)
  - Visualization saved to output/walk_visualization.png
  - Walk waypoints saved to output/planned_walk.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lmnav.aws_house_adapter import AWSHouseAdapter
from lmnav.clip_scorer import CLIPScorer
from lmnav.llm_extractor import extract_landmarks
from lmnav.graph_search import (
    find_optimal_route,
    get_walk_nodes,
    get_landmark_assignments,
    compute_walk_distance,
)
from lmnav.visualizer import visualize_walk
from lmnav import state_manager


def load_config(config_path: str) -> dict:
    """Load pipeline configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _compute_approach_goal(
    clip_scorer, landmark_text, best_node_idx, node_ids,
    poses, graph_data_dir, output_dir,
):
    """
    Compute where the robot should drive to approach the actual object.

    Uses CLIP patch heatmap (16×16 per-patch similarity) to find WHERE in
    the image the landmark appears, then depth unprojection to get the
    world-frame (x, y) coordinate. Falls back to bearing-only approach if
    depth data is unavailable.

    Returns:
        dict with 'approach_goal', 'approach_source', 'best_clip_theta'
    """
    best_node_id = node_ids[best_node_idx]
    robot_pose = poses[best_node_idx]  # [x, y, theta]
    rx, ry, r_theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])

    result = {
        "approach_goal": None,
        "approach_source": "fallback_bearing",
        "best_clip_theta": r_theta,
    }

    # Try depth-based approach first
    depth_path = Path(graph_data_dir) / f"node_{best_node_id:03d}_depth.npy"
    intrinsics_path = Path(graph_data_dir) / "camera_intrinsics.json"
    rgb_path = Path(graph_data_dir) / f"node_{best_node_id:03d}.png"

    if not depth_path.exists() or not intrinsics_path.exists():
        print(f"   ⚠️  No depth data for node {best_node_id} — using bearing fallback")
        return result

    try:
        # Stage B1: CLIP patch heatmap
        px, py, heatmap = clip_scorer.compute_patch_heatmap(
            str(rgb_path), landmark_text
        )

        if px is None:
            print("   ⚠️  Low heatmap confidence — using bearing fallback")
            return result

        # Save debug heatmap overlay
        _save_heatmap_debug(str(rgb_path), heatmap, px, py,
                           str(output_dir / "approach_heatmap_debug.jpg"))

        # Stage B2: Depth unprojection
        with open(intrinsics_path) as f:
            intrinsics = json.load(f)

        depth = np.load(str(depth_path))

        # Search a small window around (px, py) for a valid depth reading
        w = 5
        h_img, w_img = depth.shape
        region = depth[
            max(0, py - w): min(h_img, py + w),
            max(0, px - w): min(w_img, px + w)
        ]
        valid = region[(region > 0.1) & (region < 10.0) & np.isfinite(region)]

        if len(valid) == 0:
            print("   ⚠️  No valid depth at heatmap centroid — using bearing fallback")
            return result

        d = float(np.median(valid))

        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]

        # Unproject pixel → camera optical frame (Z=forward, X=right, Y=down)
        X_opt = (px - cx) * d / fx
        Y_opt = (py - cy) * d / fy
        Z_opt = d

        # Optical → camera_link (X=forward, Y=left, Z=up)
        X_cam = Z_opt
        Y_cam = -X_opt

        # Camera-to-base_link offset from waffle_stable.model camera_joint:
        # <pose>0.064 -0.065 0.34 0 0 0</pose>
        CAM_X_OFFSET = 0.064
        CAM_Y_OFFSET = -0.065

        X_base = X_cam + CAM_X_OFFSET
        Y_base = Y_cam + CAM_Y_OFFSET

        # base_link → map frame
        cos_t = np.cos(r_theta)
        sin_t = np.sin(r_theta)
        obj_x = rx + X_base * cos_t - Y_base * sin_t
        obj_y = ry + X_base * sin_t + Y_base * cos_t

        # Compute approach goal (standoff 0.6m from object)
        dx = obj_x - rx
        dy = obj_y - ry
        dist = np.hypot(dx, dy)
        standoff = 0.6

        if dist < standoff:
            # Already close — just face it
            a_x, a_y = obj_x, obj_y
        else:
            a_x = obj_x - standoff * (dx / dist)
            a_y = obj_y - standoff * (dy / dist)
        a_theta = float(np.arctan2(dy, dx))

        result["approach_goal"] = {
            "x": round(float(a_x), 4),
            "y": round(float(a_y), 4),
            "theta": round(a_theta, 4),
        }
        result["approach_source"] = "clip_patch_depth"
        print(f"   🎯 Approach goal computed: ({a_x:.2f}, {a_y:.2f}) "
              f"(obj at {obj_x:.2f}, {obj_y:.2f}, depth={d:.2f}m)")

    except Exception as e:
        print(f"   ⚠️  Approach computation failed ({e}) — using bearing fallback")

    return result


def _save_heatmap_debug(rgb_path, heatmap, px, py, output_path):
    """Save a debug image: RGB with heatmap overlay and centroid marker."""
    try:
        from PIL import Image as PILImage, ImageDraw
        img = PILImage.open(rgb_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        # Draw crosshair at centroid
        r = 8
        draw.ellipse([px - r, py - r, px + r, py + r], outline="red", width=3)
        draw.line([px - r*2, py, px + r*2, py], fill="red", width=2)
        draw.line([px, py - r*2, px, py + r*2], fill="red", width=2)
        img.save(output_path)
        print(f"   📊 Heatmap debug saved: {output_path}")
    except Exception as e:
        print(f"   ⚠️  Could not save heatmap debug: {e}")


def run_pipeline(
    instruction: str,
    start_node: int = None,
    config_path: str = None,
):
    """
    Run the full LM-Nav pipeline.
    
    Adapted from lm_nav/pipeline.py:full_pipeline() — same structure:
      1. Extract landmarks (LLM)
      2. Build CLIP similarity matrix
      3. Find optimal route (Dijkstra DP)
      4. Visualize result
    """
    # ─── Load Config ───
    if config_path is None:
        config_path = str(project_root / "config" / "pipeline_config.yaml")
    config = load_config(config_path)

    print("=" * 60)
    print("🧭 LM-Nav Pipeline — AWS Small House World")
    print("=" * 60)
    print(f"📝 Instruction: \"{instruction}\"")
    print(f"🏁 Start node: {start_node}")
    print()

    # ─── Step 1: Load Environment ───
    print("━" * 40)
    print("Step 1: Loading environment graph...")
    print("━" * 40)

    graph_data_dir = str(project_root / config["environment"]["graph_data_dir"])
    threshold = config["graph"]["pose_edge_threshold_m"]
    adapter = AWSHouseAdapter(graph_data_dir, pose_edge_threshold_m=threshold)
    graph = adapter.get_graph()
    node_ids = adapter.get_node_ids()
    images = adapter.get_all_images()
    poses = adapter.get_all_poses()

    print(f"   ✅ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"   ✅ Node IDs: {node_ids[:5]}... (showing first 5)")
    print()

    # ─── Determine Start Node ───
    print("━" * 40)
    print("Start node selection...")
    print("━" * 40)

    # Load poses data for state_manager
    poses_file = Path(graph_data_dir) / "poses.json"
    import json as _json
    with open(poses_file, "r") as f:
        poses_data = _json.load(f)

    state_path = str(project_root / "output" / "robot_state.json")
    spawn_x = config.get("robot", {}).get("spawn_x", None)
    spawn_y = config.get("robot", {}).get("spawn_y", None)
    selected_node, reason = state_manager.determine_start_node(
        poses_data=poses_data,
        state_path=state_path,
        override_node=start_node,
        spawn_x=spawn_x,
        spawn_y=spawn_y,
    )
    start_node = selected_node

    print(f"   🏁 Start node: {start_node} — {reason}")
    print()

    # ─── Step 2: Extract Landmarks ───
    print("━" * 40)
    print("Step 2: Extracting landmarks (LLM)...")
    print("━" * 40)

    llm_backend = config["models"]["llm_backend"]
    llm_model = config["models"]["llm_model"]
    ollama_host = config["models"]["ollama_host"]

    landmarks = extract_landmarks(
        instruction,
        backend=llm_backend,
        model=llm_model,
        ollama_host=ollama_host,
    )

    if not landmarks:
        print("❌ No landmarks extracted! Trying spaCy fallback...")
        landmarks = extract_landmarks(instruction, backend="spacy")

    if not landmarks:
        print("❌ Could not extract any landmarks. Exiting.")
        return None

    print(f"   ✅ Landmarks: {landmarks}")
    print()

    # ─── Step 3: CLIP Scoring ───
    print("━" * 40)
    print("Step 3: Computing CLIP similarities...")
    print("━" * 40)

    clip_scorer = CLIPScorer(
        model_name=config["models"]["clip_model"],
        pretrained=config["models"]["clip_pretrained"],
        prompt_template=config["search"]["clip_prompt"],
    )
    # Use RAW cosine similarity — NOT softmax.
    # With raw scores + near-zero alpha, the Dijkstra DP correctly picks
    # the node CLIP actually thinks matches, regardless of distance.
    similarity_matrix = clip_scorer.score(images, landmarks)

    # ── Build position groups ──────────────────────────────────────────
    pos_groups = defaultdict(list)  # (x_rounded, y_rounded) → [indices]
    for idx, nid in enumerate(node_ids):
        x_r = round(float(poses[idx, 0]), 2)
        y_r = round(float(poses[idx, 1]), 2)
        pos_groups[(x_r, y_r)].append(idx)

    # ── Fix 3: Record best raw node per position BEFORE fusion ─────────
    # This tells us which camera angle actually saw the landmark best.
    # After fusion all angles get the same score, so we'd lose this info.
    best_raw_node_per_position = {}  # (x_rounded, y_rounded) → node_index
    for (px, py), indices in pos_groups.items():
        for j in range(similarity_matrix.shape[1]):
            best_idx = max(indices, key=lambda i: similarity_matrix[i, j])
            # Store per landmark j
            key = (px, py, j)
            best_raw_node_per_position[key] = best_idx

    # ── Multi-view fusion ──────────────────────────────────────────────
    # Nodes at the same (x, y) are just different camera angles. If ANY
    # angle at a position sees the target, ALL angles at that position
    # should benefit — the robot can rotate to face it once it arrives.
    n_fused = 0
    for (px, py), indices in pos_groups.items():
        if len(indices) <= 1:
            continue
        for j in range(similarity_matrix.shape[1]):
            max_score = max(similarity_matrix[i, j] for i in indices)
            for i in indices:
                if similarity_matrix[i, j] < max_score:
                    n_fused += 1
                    similarity_matrix[i, j] = max_score

    if n_fused > 0:
        print(f"   🔀 Multi-view fusion: boosted {n_fused} node scores "
              f"across {len([g for g in pos_groups.values() if len(g) > 1])} "
              f"position groups")

    # Print top matches per landmark + save images for inspection
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("   📸 Saving top CLIP match images to output/clip_match_lm_*...")
    # Track: for each landmark j, which node index is the BEST raw angle
    best_raw_for_landmark = {}  # j → node_index (pre-fusion best)

    for j, lm in enumerate(landmarks):
        scores = similarity_matrix[:, j]
        top_idx = np.argsort(scores)[::-1][:3]

        # ── Fix 3: For each top position, use the BEST ANGLE node ──
        # top_idx might point to any angle at that position (fused scores
        # are identical). Replace each with the actual best-raw-angle node.
        corrected_top = []
        seen_positions = set()
        for idx in top_idx:
            px = round(float(poses[idx, 0]), 2)
            py = round(float(poses[idx, 1]), 2)
            pos_key = (px, py)
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)
            # Use best raw angle at this position
            best_angle_idx = best_raw_node_per_position.get((px, py, j), idx)
            corrected_top.append(best_angle_idx)

        # Pad with original top if needed
        while len(corrected_top) < 3 and len(corrected_top) < len(top_idx):
            for idx in top_idx:
                if idx not in corrected_top:
                    corrected_top.append(idx)
                    break

        # The first corrected node is the true best for this landmark
        best_raw_for_landmark[j] = corrected_top[0] if corrected_top else top_idx[0]

        top_str = ", ".join(
            f"node_{node_ids[i]}({scores[i]:.3f})" for i in corrected_top
        )
        print(f"   🏷️  \"{lm}\" → top matches: {top_str}")

        # Save top-3 images (best angle at each position)
        for rank, idx in enumerate(corrected_top):
            src_img = images[idx]
            dst = output_dir / f"clip_match_lm{j}_rank{rank+1}_node{node_ids[idx]}.png"
            src_img.save(str(dst))
        print(f"      Saved: clip_match_lm{j}_rank1-3 (node images for visual check)")
    print()

    # ─── Step 4: Graph Search ───
    print("━" * 40)
    print("Step 4: Finding optimal walk (Dijkstra DP)...")
    print("━" * 40)

    alpha = config["search"]["alpha"]
    result = find_optimal_route(
        graph=graph,
        similarity_matrix=similarity_matrix,
        node_ids=node_ids,
        start=start_node,
        alpha=alpha,
    )

    walk = result["walk"]
    walk_nodes = get_walk_nodes(walk)
    landmark_nodes = get_landmark_assignments(walk)
    walk_distance = compute_walk_distance(walk, graph)

    print(f"   ✅ Walk: {walk_nodes}")
    print(f"   📏 Total distance: {walk_distance:.2f} meters")
    print(f"   🏷️  Landmark nodes: {landmark_nodes}")
    print()

    # Warn on trivial (zero-distance) walk
    if len(set(walk_nodes)) == 1 and len(walk_nodes) > 1:
        print(
            f"   ⚠️  TRIVIAL WALK: Start node ({start_node}) is the same as the "
            f"destination. This usually means:\n"
            f"      • The robot is already at the target landmark (correct!), OR"
            f"\n"
            f"      • The start node is wrong — try running with Gazebo active"
            f" (AMCL) or use --start-node N\n"
        )

    # Print walk with landmark annotations
    print("   Walk detail:")
    lm_idx = 0
    for node_id, change in walk:
        if node_id in node_ids:
            idx = node_ids.index(node_id)
            pose = poses[idx]
            if change == -1 and lm_idx < len(landmarks):
                print(
                    f"   → Node {node_id:3d} ({pose[0]:7.2f}, {pose[1]:7.2f}) "
                    f"🏷️  LANDMARK: \"{landmarks[lm_idx]}\""
                )
                lm_idx += 1
            else:
                print(f"   → Node {node_id:3d} ({pose[0]:7.2f}, {pose[1]:7.2f})")
    print()

    # ─── Step 5: Visualization ───
    print("━" * 40)
    print("Step 5: Generating visualization...")
    print("━" * 40)

    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to find the occupancy grid map
    map_path = (
        project_root.parent
        / "aws-robomaker-small-house-world"
        / "maps"
        / "turtlebot3_waffle_pi"
        / "map.pgm"
    )
    map_origin = (-12.5, -12.5) if map_path.exists() else None
    map_resolution = 0.05 if map_path.exists() else None

    visualize_walk(
        graph=graph,
        walk=walk,
        landmarks=landmarks,
        node_ids=node_ids,
        poses=poses,
        similarity_matrix=similarity_matrix,
        map_image_path=str(map_path) if map_path.exists() else None,
        map_origin=map_origin,
        map_resolution=map_resolution,
        output_path=str(output_dir / "walk_visualization.png"),
        title="LM-Nav Planned Walk — AWS Small House",
        instruction=instruction,
    )

    # ─── Step 6: Save Walk for Execution ───
    walk_data = {
        "instruction": instruction,
        "landmarks": landmarks,
        "walk_nodes": walk_nodes,
        "landmark_nodes": landmark_nodes,
        "walk_distance_m": walk_distance,
        "waypoints": [],
    }

    # ── Build waypoints with smart theta ──────────────────────────────
    # Intermediate nodes: theta = direction TOWARD next node (no spinning!)
    # Landmark / final node: theta = original capture angle
    # Also deduplicate same-position consecutive nodes.
    raw_wps = []
    landmark_node_set = set(landmark_nodes)
    for node_id in walk_nodes:
        if node_id in node_ids:
            idx = node_ids.index(node_id)
            raw_wps.append({
                "node_id": node_id,
                "x": float(poses[idx, 0]),
                "y": float(poses[idx, 1]),
                "theta": float(poses[idx, 2]),
                "is_landmark": node_id in landmark_node_set,
            })

    # Deduplicate consecutive waypoints at the same (x, y)
    deduped_wps = []
    for wp in raw_wps:
        if deduped_wps:
            prev = deduped_wps[-1]
            dx = abs(wp["x"] - prev["x"])
            dy = abs(wp["y"] - prev["y"])
            if dx < 0.1 and dy < 0.1:
                # Same position — keep the later one (which might be landmark)
                # Merge is_landmark flag
                wp["is_landmark"] = wp["is_landmark"] or prev.get("is_landmark", False)
                deduped_wps[-1] = wp
                continue
        deduped_wps.append(wp)

    # For intermediate waypoints, set theta to face the NEXT waypoint
    for i in range(len(deduped_wps) - 1):
        if not deduped_wps[i].get("is_landmark", False):
            cur = deduped_wps[i]
            nxt = deduped_wps[i + 1]
            dx = nxt["x"] - cur["x"]
            dy = nxt["y"] - cur["y"]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                cur["theta"] = float(math.atan2(dy, dx))

    # ── Compute approach goals for landmark waypoints ─────────────────
    print()
    print("━" * 40)
    print("Step 6: Computing approach goals...")
    print("━" * 40)

    lm_j = 0  # landmark index into landmarks list
    for wp in deduped_wps:
        if wp.get("is_landmark") and lm_j < len(landmarks):
            # Find the best raw angle node for this landmark
            best_idx = best_raw_for_landmark.get(lm_j, None)
            if best_idx is not None:
                # Store the best capture theta so executor can use it as fallback
                wp["best_clip_theta"] = float(poses[best_idx, 2])
                wp["best_clip_node"] = int(node_ids[best_idx])

                # Try depth-based approach goal
                approach_info = _compute_approach_goal(
                    clip_scorer=clip_scorer,
                    landmark_text=landmarks[lm_j],
                    best_node_idx=best_idx,
                    node_ids=node_ids,
                    poses=poses,
                    graph_data_dir=graph_data_dir,
                    output_dir=output_dir,
                )
                wp["approach_goal"] = approach_info["approach_goal"]
                wp["approach_source"] = approach_info["approach_source"]
            else:
                wp["approach_goal"] = None
                wp["approach_source"] = "no_best_node"

            lm_j += 1

    walk_data["waypoints"] = deduped_wps
    if len(deduped_wps) != len(raw_wps):
        print(f"   📍 Waypoints (deduped): {len(deduped_wps)} "
              f"(from {len(raw_wps)} raw)")

    walk_json_path = output_dir / "planned_walk.json"
    with open(walk_json_path, "w") as f:
        json.dump(walk_data, f, indent=2)
    print(f"💾 Walk saved to: {walk_json_path}")

    print()
    print("=" * 60)
    print("🎉 Pipeline complete!")
    print("=" * 60)
    print(f"   Instruction: \"{instruction}\"")
    print(f"   Landmarks:   {landmarks}")
    print(f"   Walk:        {walk_nodes}")
    print(f"   Distance:    {walk_distance:.2f}m")
    print(f"   Viz:         output/walk_visualization.png")
    print(f"   Walk JSON:   output/planned_walk.json")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="LM-Nav Pipeline — Zero-shot Vision-Language Navigation"
    )
    parser.add_argument(
        "--instruction", "-i",
        type=str,
        required=True,
        help="Natural language navigation instruction",
    )
    parser.add_argument(
        "--start-node", "-s",
        type=int,
        default=None,
        help="Override start node ID (default: auto-detect from AMCL/saved state)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()

    run_pipeline(
        instruction=args.instruction,
        start_node=args.start_node,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
