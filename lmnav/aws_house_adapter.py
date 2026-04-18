"""
AWS Small House Adapter
=======================
Loads the topological graph from exploration data collected by explore_house.py.

Expected data layout in graph_data_dir:
  data/aws_house_graph/
  ├── node_000.png
  ├── node_001.png
  ├── ...
  └── poses.json   ← [{id: 0, x: ..., y: ..., theta: ...}, ...]
"""

import json
import os
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from PIL import Image

from lmnav.adapter import EnvironmentAdapter


class AWSHouseAdapter(EnvironmentAdapter):
    """
    Environment adapter for the AWS RoboMaker Small House World.
    
    Reads pre-collected exploration data (images + poses) and builds
    a topological graph with Euclidean-distance edges.
    """

    def __init__(self, graph_data_dir: str, pose_edge_threshold_m: float = 3.0):
        """
        Args:
            graph_data_dir: Path to directory containing node_*.png and poses.json
            pose_edge_threshold_m: Maximum distance (meters) to create an edge
                                   between two nodes
        """
        self.graph_data_dir = Path(graph_data_dir)
        self.pose_edge_threshold_m = pose_edge_threshold_m

        # Validate data exists
        poses_file = self.graph_data_dir / "poses.json"
        if not poses_file.exists():
            raise FileNotFoundError(
                f"Exploration data not found at {self.graph_data_dir}.\n"
                f"Run 'python scripts/explore_house.py' first to collect "
                f"images and poses from the AWS House World in Gazebo."
            )

        # Load poses
        with open(poses_file, "r") as f:
            self._poses_data = json.load(f)

        # Build internal data structures
        self._node_ids = [p["id"] for p in self._poses_data]
        self._poses = {
            p["id"]: np.array([p["x"], p["y"], p["theta"]])
            for p in self._poses_data
        }
        self._images = {}  # lazy-loaded
        self._graph = None  # built on first access

    def get_graph(self) -> nx.Graph:
        """Build and return the topological graph with Euclidean-distance edges."""
        if self._graph is not None:
            return self._graph

        G = nx.Graph()

        # Add nodes with image and pose attributes
        for node_id in self._node_ids:
            G.add_node(
                node_id,
                image=self.get_image(node_id),
                pose=self.get_pose(node_id),
            )

        # Add edges between nodes within distance threshold
        positions = self.get_all_poses()
        for i, nid_i in enumerate(self._node_ids):
            for j, nid_j in enumerate(self._node_ids):
                if i >= j:
                    continue
                pos_i = positions[i, :2]  # (x, y) only
                pos_j = positions[j, :2]
                dist = np.linalg.norm(pos_i - pos_j)
                if dist <= self.pose_edge_threshold_m:
                    G.add_edge(nid_i, nid_j, weight=dist)

        self._graph = G
        return G

    def get_image(self, node_id: int) -> Image.Image:
        """Load and return the RGB image for the given node."""
        if node_id not in self._images:
            img_path = self.graph_data_dir / f"node_{node_id:03d}.png"
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {img_path}. "
                    f"Re-run exploration to capture this node."
                )
            self._images[node_id] = Image.open(img_path).convert("RGB")
        return self._images[node_id]

    def get_pose(self, node_id: int) -> np.ndarray:
        """Return (x, y, theta) pose for the given node."""
        if node_id not in self._poses:
            raise KeyError(f"Node {node_id} not found in poses data.")
        return self._poses[node_id]

    def get_node_ids(self) -> List[int]:
        """Return all node IDs in order."""
        return list(self._node_ids)

    def __repr__(self) -> str:
        n = len(self._node_ids)
        G = self.get_graph()
        e = G.number_of_edges()
        return (
            f"AWSHouseAdapter(nodes={n}, edges={e}, "
            f"threshold={self.pose_edge_threshold_m}m, "
            f"data_dir='{self.graph_data_dir}')"
        )
