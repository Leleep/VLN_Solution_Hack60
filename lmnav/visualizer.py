"""
Walk Visualizer
===============
Visualizes the planned walk on a 2D overhead map of the environment.

Shows:
  - All graph nodes as dots
  - Edges as light lines
  - Planned walk as a colored path (green → red)
  - Landmark-matched nodes with text annotations
  - Optional: occupancy grid map as background
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from lmnav.graph_search import get_walk_nodes, get_landmark_assignments


def visualize_walk(
    graph,
    walk: List[Tuple[int, int]],
    landmarks: List[str],
    node_ids: List[int],
    poses: np.ndarray,
    similarity_matrix: Optional[np.ndarray] = None,
    map_image_path: Optional[str] = None,
    map_origin: Optional[Tuple[float, float]] = None,
    map_resolution: Optional[float] = None,
    output_path: str = "output/walk_visualization.png",
    title: str = "LM-Nav Planned Walk",
    instruction: str = "",
):
    """
    Create a 2D visualization of the planned walk.
    
    Args:
        graph: networkx.Graph
        walk: Walk output from find_optimal_route
        landmarks: List of landmark strings
        node_ids: List of node IDs
        poses: (N, 3) array of poses (x, y, theta)
        similarity_matrix: Optional (N, M) CLIP scores for annotation
        map_image_path: Path to occupancy grid PGM (optional background)
        map_origin: (x, y) origin of the map in world coordinates
        map_resolution: Meters per pixel of the map
        output_path: Where to save the visualization
        title: Plot title
        instruction: Original instruction text (shown as subtitle)
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # ─── Background: Occupancy Grid Map ───
    if map_image_path and Path(map_image_path).exists():
        map_img = np.array(Image.open(map_image_path))
        if map_origin and map_resolution:
            ox, oy = map_origin
            h, w = map_img.shape[:2]
            extent = [
                ox,
                ox + w * map_resolution,
                oy,
                oy + h * map_resolution,
            ]
            ax.imshow(map_img, cmap="gray", extent=extent, alpha=0.3, origin="lower")

    # ─── Draw All Edges ───
    for u, v in graph.edges():
        if u in node_ids and v in node_ids:
            idx_u = node_ids.index(u)
            idx_v = node_ids.index(v)
            ax.plot(
                [poses[idx_u, 0], poses[idx_v, 0]],
                [poses[idx_u, 1], poses[idx_v, 1]],
                color="#cccccc",
                linewidth=0.8,
                alpha=0.5,
                zorder=1,
            )

    # ─── Draw All Nodes ───
    ax.scatter(
        poses[:, 0],
        poses[:, 1],
        c="#aaaaaa",
        s=40,
        zorder=2,
        edgecolors="white",
        linewidths=0.5,
        label="Graph nodes",
    )

    # Add node ID labels
    for i, nid in enumerate(node_ids):
        ax.annotate(
            str(nid),
            (poses[i, 0], poses[i, 1]),
            fontsize=6,
            ha="center",
            va="bottom",
            color="#666666",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # ─── Draw Planned Walk ───
    walk_nodes = get_walk_nodes(walk)
    walk_indices = [node_ids.index(n) for n in walk_nodes if n in node_ids]

    if len(walk_indices) >= 2:
        walk_x = [poses[i, 0] for i in walk_indices]
        walk_y = [poses[i, 1] for i in walk_indices]

        # Draw walk path with color gradient
        n_segments = len(walk_x) - 1
        cmap = plt.cm.RdYlGn_r  # green → yellow → red
        for seg in range(n_segments):
            color = cmap(seg / max(n_segments - 1, 1))
            ax.plot(
                [walk_x[seg], walk_x[seg + 1]],
                [walk_y[seg], walk_y[seg + 1]],
                color=color,
                linewidth=3,
                zorder=4,
                solid_capstyle="round",
            )
            # Arrow head for direction
            if seg < n_segments - 1:
                dx = walk_x[seg + 1] - walk_x[seg]
                dy = walk_y[seg + 1] - walk_y[seg]
                mid_x = (walk_x[seg] + walk_x[seg + 1]) / 2
                mid_y = (walk_y[seg] + walk_y[seg + 1]) / 2
                ax.annotate(
                    "",
                    xy=(mid_x + dx * 0.1, mid_y + dy * 0.1),
                    xytext=(mid_x - dx * 0.1, mid_y - dy * 0.1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    zorder=5,
                )

        # Highlight walk nodes
        ax.scatter(
            walk_x,
            walk_y,
            c=range(len(walk_x)),
            cmap="RdYlGn_r",
            s=100,
            zorder=5,
            edgecolors="black",
            linewidths=1.5,
            label="Walk path",
        )

        # Start marker
        ax.scatter(
            [walk_x[0]], [walk_y[0]],
            c="green", s=200, marker="*", zorder=6,
            edgecolors="black", linewidths=1, label="Start",
        )
        # End marker
        ax.scatter(
            [walk_x[-1]], [walk_y[-1]],
            c="red", s=200, marker="X", zorder=6,
            edgecolors="black", linewidths=1, label="End",
        )

    # ─── Annotate Landmark Nodes ───
    landmark_nodes = get_landmark_assignments(walk)
    landmark_idx = 0
    for node_id, change in walk:
        if change == -1 and landmark_idx < len(landmarks):
            if node_id in node_ids:
                idx = node_ids.index(node_id)
                ax.annotate(
                    f"🏷️ {landmarks[landmark_idx]}",
                    (poses[idx, 0], poses[idx, 1]),
                    fontsize=9,
                    fontweight="bold",
                    color="#1a1a2e",
                    ha="left",
                    va="top",
                    xytext=(10, -10),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="#e8f4fd",
                        edgecolor="#3498db",
                        alpha=0.9,
                    ),
                    arrowprops=dict(arrowstyle="->", color="#3498db"),
                    zorder=7,
                )
            landmark_idx += 1

    # ─── Labels and Formatting ───
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if instruction:
        ax.set_title(
            f"{title}\n\"{instruction}\"",
            fontsize=14,
            fontweight="bold",
        )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # Landmarks summary box
    if landmarks:
        landmarks_text = "Landmarks:\n" + "\n".join(
            f"  {i+1}. {lm}" for i, lm in enumerate(landmarks)
        )
        props = dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8)
        ax.text(
            0.98, 0.02, landmarks_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=props,
        )

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Visualization saved to: {output_path}")
