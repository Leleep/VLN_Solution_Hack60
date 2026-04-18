"""
Environment Adapter — Abstract Base Class
==========================================
All downstream modules (CLIP scorer, LLM extractor, graph search, navigator)
program against this common interface. Swapping environments (AWS House ↔ MP3D)
requires zero changes to the pipeline code.
"""

from abc import ABC, abstractmethod
from typing import List

import networkx as nx
import numpy as np
from PIL import Image


class EnvironmentAdapter(ABC):
    """
    Abstract base class for environment backends.
    
    Each adapter provides a topological graph of the environment where:
      - Nodes represent viewpoints (robot poses with captured images)
      - Edges represent navigable connections between viewpoints
      - Edge weights are Euclidean distances between poses
    """

    @abstractmethod
    def get_graph(self) -> nx.Graph:
        """
        Returns a networkx graph where:
          - Each node has attributes:
              'image': PIL.Image.Image (RGB image at this viewpoint)
              'pose':  np.ndarray shape (3,) = (x, y, theta)
          - Each edge has attribute:
              'weight': float = Euclidean distance between node poses
        
        Node IDs are integers (0, 1, 2, ...) to match the lm_nav convention.
        """
        ...

    @abstractmethod
    def get_image(self, node_id: int) -> Image.Image:
        """Returns the RGB image captured at the given node/viewpoint."""
        ...

    @abstractmethod
    def get_pose(self, node_id: int) -> np.ndarray:
        """Returns (x, y, theta) pose for the given node."""
        ...

    @abstractmethod
    def get_node_ids(self) -> List[int]:
        """Returns list of all node IDs in the graph."""
        ...

    def get_all_images(self) -> List[Image.Image]:
        """Returns images for all nodes, ordered by node ID."""
        return [self.get_image(nid) for nid in self.get_node_ids()]

    def get_all_poses(self) -> np.ndarray:
        """Returns (N, 3) array of all poses, ordered by node ID."""
        return np.array([self.get_pose(nid) for nid in self.get_node_ids()])
