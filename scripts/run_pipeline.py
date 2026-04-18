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
import sys
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


def load_config(config_path: str) -> dict:
    """Load pipeline configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(
    instruction: str,
    start_node: int = 0,
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
    # Use softmax normalization for sharper node discrimination.
    # Raw cosine similarity gives very close scores (~0.20-0.24) for all nodes,
    # making differences meaningless. Softmax with low temperature amplifies
    # small differences into a probability distribution P(node | landmark).
    similarity_matrix = clip_scorer.score_softmax(
        images, landmarks, temperature=0.01
    )

    # Print top matches per landmark
    for j, lm in enumerate(landmarks):
        scores = similarity_matrix[:, j]
        top_idx = np.argsort(scores)[::-1][:3]
        top_str = ", ".join(
            f"node_{node_ids[i]}({scores[i]:.3f})" for i in top_idx
        )
        print(f"   🏷️  \"{lm}\" → top matches: {top_str}")
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
    for node_id in walk_nodes:
        if node_id in node_ids:
            idx = node_ids.index(node_id)
            walk_data["waypoints"].append({
                "node_id": node_id,
                "x": float(poses[idx, 0]),
                "y": float(poses[idx, 1]),
                "theta": float(poses[idx, 2]),
            })

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
        default=0,
        help="Starting node ID (default: 0)",
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
