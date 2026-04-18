"""
Matterport3D Adapter — Phase 2 Placeholder
===========================================
Loads the topological graph from Matterport3D scan data.

Uses the pre-computed connectivity graph (from Matterport3DSimulator repo)
and skybox images from the MP3D download.

To switch to MP3D:
  1. Extract skybox images:
     cd matterport_data/v1/scans/17DRP5sb8fy/
     unzip matterport_skybox_images.zip
  
  2. Update config/pipeline_config.yaml:
     environment:
       backend: "mp3d"
       mp3d_scan_id: "17DRP5sb8fy"
       mp3d_data_root: "../matterport_data/v1/scans"

  3. Run pipeline as normal — same command, different data.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import networkx as nx
import numpy as np
from PIL import Image

from lmnav.adapter import EnvironmentAdapter

# URL for connectivity graphs from Matterport3DSimulator repo
CONNECTIVITY_URL = (
    "https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/"
    "master/connectivity/{scan_id}_connectivity.json"
)


class Matterport3DAdapter(EnvironmentAdapter):
    """
    Environment adapter for Matterport3D scans.
    
    Uses pre-computed connectivity graphs and skybox images.
    No Gazebo rendering needed for graph construction — only for execution.
    
    Connectivity JSON format (per viewpoint):
      {
        "image_id": "10c252c90fa24ef3b698c6f54d984c5c",
        "pose": [16 floats → 4×4 SE3 matrix, row-major],
        "included": true/false,
        "unobstructed": [N booleans → navigable edges],
        "height": 1.53
      }
    """

    def __init__(
        self,
        scan_id: str = "17DRP5sb8fy",
        data_root: str = "../matterport_data/v1/scans",
        connectivity_dir: Optional[str] = None,
    ):
        self.scan_id = scan_id
        self.data_root = Path(data_root)
        self.scan_dir = self.data_root / scan_id

        # Load or download connectivity graph
        if connectivity_dir:
            conn_path = Path(connectivity_dir) / f"{scan_id}_connectivity.json"
        else:
            conn_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "connectivity"
                / f"{scan_id}_connectivity.json"
            )

        if not conn_path.exists():
            self._download_connectivity(conn_path)

        with open(conn_path, "r") as f:
            self._connectivity = json.load(f)

        # Filter to included viewpoints and build index
        self._viewpoints = [
            vp for vp in self._connectivity if vp.get("included", False)
        ]
        self._image_id_to_idx = {
            vp["image_id"]: i for i, vp in enumerate(self._viewpoints)
        }
        # Also need full index for unobstructed lookups
        self._all_image_ids = [vp["image_id"] for vp in self._connectivity]

        self._node_ids = list(range(len(self._viewpoints)))
        self._graph = None
        self._images = {}

    def _download_connectivity(self, dest_path: Path):
        """Download connectivity JSON from Matterport3DSimulator GitHub."""
        import requests

        url = CONNECTIVITY_URL.format(scan_id=self.scan_id)
        print(f"📥 Downloading connectivity graph from: {url}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(dest_path, "w") as f:
            f.write(response.text)
        print(f"   ✅ Saved to: {dest_path}")

    def _parse_pose(self, pose_list: list) -> np.ndarray:
        """
        Extract (x, y, theta) from 4×4 SE3 matrix (row-major, 16 floats).
        
        Position = translation column: pose[3], pose[7], pose[11]
        Heading θ = atan2(R[1,0], R[0,0]) = atan2(pose[4], pose[0])
        """
        x = pose_list[3]
        y = pose_list[7]
        theta = np.arctan2(pose_list[4], pose_list[0])
        return np.array([x, y, theta])

    def get_graph(self) -> nx.Graph:
        if self._graph is not None:
            return self._graph

        G = nx.Graph()

        # Add included viewpoints as nodes
        for idx, vp in enumerate(self._viewpoints):
            pose = self._parse_pose(vp["pose"])
            G.add_node(idx, pose=pose, image=self.get_image(idx))

        # Add edges from unobstructed adjacency
        for idx, vp in enumerate(self._viewpoints):
            unobstructed = vp["unobstructed"]
            pose_i = self._parse_pose(vp["pose"])

            for j, is_connected in enumerate(unobstructed):
                if not is_connected:
                    continue
                # j indexes into the FULL connectivity list (all viewpoints)
                other_image_id = self._all_image_ids[j]
                if other_image_id not in self._image_id_to_idx:
                    continue  # skip excluded viewpoints
                other_idx = self._image_id_to_idx[other_image_id]

                if idx < other_idx:  # avoid duplicates
                    pose_j = self._parse_pose(
                        self._connectivity[j]["pose"]
                    )
                    dist = np.linalg.norm(pose_i[:2] - pose_j[:2])
                    G.add_edge(idx, other_idx, weight=dist)

        self._graph = G
        return G

    def get_image(self, node_id: int) -> Image.Image:
        """
        Load skybox image for the given node.
        
        Skybox path: {scan_dir}/matterport_skybox_images/{image_id}/{image_id}_skybox{face}_sami.jpg
        Face 4 = front-facing view (most useful for CLIP).
        
        Falls back to a colored placeholder if images not extracted.
        """
        if node_id in self._images:
            return self._images[node_id]

        vp = self._viewpoints[node_id]
        image_id = vp["image_id"]

        # Try loading skybox front face (face 4)
        skybox_dir = self.scan_dir / "matterport_skybox_images" / image_id
        for face_idx in [4, 0, 1, 2, 3, 5]:  # prefer front face
            img_path = skybox_dir / f"{image_id}_skybox{face_idx}_sami.jpg"
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                self._images[node_id] = img
                return img

        # Fallback: colored placeholder
        pose = self._parse_pose(vp["pose"])
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        self._images[node_id] = img
        return img

    def get_pose(self, node_id: int) -> np.ndarray:
        vp = self._viewpoints[node_id]
        return self._parse_pose(vp["pose"])

    def get_node_ids(self) -> List[int]:
        return list(self._node_ids)

    def __repr__(self) -> str:
        n = len(self._viewpoints)
        return f"Matterport3DAdapter(scan='{self.scan_id}', viewpoints={n})"
