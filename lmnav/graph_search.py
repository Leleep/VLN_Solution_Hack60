"""
Graph Search — Dijkstra-based Dynamic Programming
==================================================
Finds the optimal walk through a topological graph that visits
landmarks in order while minimizing travel distance.

This is Algorithm 1 from the LM-Nav paper (Shah et al., 2022).

Directly adapted from lm_nav/optimal_route.py with these changes:
  - Uses networkx.Graph instead of custom NavigationGraph class
  - CLIP scoring is decoupled (takes pre-computed similarity matrix)
  - Node IDs are integers matching graph node indices

Original code reference:
  lm_nav/lm_nav/optimal_route.py, lines 16-92
"""

import heapq
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def dijkstra_transform(
    initial: np.ndarray,
    graph: nx.Graph,
    node_ids: List[int],
    alpha: float,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Dijkstra-based score propagation through the graph.
    
    Propagates scores from high-scoring nodes to their neighbors,
    applying a distance-based decay. This finds the best way to
    "reach" each node from any starting point.
    
    Adapted from lm_nav/optimal_route.py:dijskra_transform (lines 16-36):
        next[neighbor] = max(next[neighbor], value - alpha * weight)
    
    Args:
        initial: Score vector of shape (N,) — one score per node
        graph: networkx.Graph with 'weight' edge attributes
        node_ids: List of node IDs (maps index → node ID)
        alpha: Distance penalty factor (higher = more penalty for distance)
        
    Returns:
        next_scores: Updated score vector after propagation
        prev: Predecessor table for backtracking — 
              prev[i] = (predecessor_node_index, table_change)
              table_change is 0 (same landmark) or -1 (previous landmark)
    """
    n = len(node_ids)
    # Map node_id → index for fast lookup
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    next_scores = np.copy(initial)
    prev = [(i, -1) for i in range(n)]  # (predecessor_idx, table_change)

    # Max-heap: we negate values because heapq is a min-heap
    priority_queue = [(-initial[i], i) for i in range(n)]
    heapq.heapify(priority_queue)

    while priority_queue:
        neg_value, node_idx = heapq.heappop(priority_queue)
        value = -neg_value

        # Skip if this entry is stale (node already has better score)
        if next_scores[node_idx] != value:
            continue

        node_id = node_ids[node_idx]

        # Propagate to neighbors
        for neighbor_id in graph.neighbors(node_id):
            if neighbor_id not in id_to_idx:
                continue
            neighbor_idx = id_to_idx[neighbor_id]
            weight = graph[node_id][neighbor_id]["weight"]
            new_score = value - alpha * weight

            if next_scores[neighbor_idx] < new_score:
                next_scores[neighbor_idx] = new_score
                heapq.heappush(priority_queue, (-new_score, neighbor_idx))
                prev[neighbor_idx] = (node_idx, 0)

    return next_scores, prev


def find_optimal_route(
    graph: nx.Graph,
    similarity_matrix: np.ndarray,
    node_ids: List[int],
    start: int,
    alpha: float = 0.0002,
) -> Dict:
    """
    Find the optimal walk that visits landmarks in order.
    
    This is the core LM-Nav algorithm (Algorithm 1 from the paper).
    
    Adapted from lm_nav/optimal_route.py:find_optimal_route (lines 68-92):
        1. Initialize score vector, set start=0, rest=-inf
        2. Dijkstra transform to propagate reachability
        3. For each landmark:
           a. Add CLIP similarity scores
           b. Dijkstra transform again
        4. Backtrack from argmax to find the walk
    
    Args:
        graph: networkx.Graph with 'weight' edge attributes
        similarity_matrix: (N_nodes, N_landmarks) CLIP similarity scores
        node_ids: List of node IDs (same order as similarity_matrix rows)
        start: Starting node ID (where the robot currently is)
        alpha: Distance penalty factor
        
    Returns:
        Dict with:
          "walk": List of (node_id, table_change) tuples
                  table_change = 0 means same segment
                  table_change = -1 means landmark transition
          "score": Final score vector
          "similarity_matrix": Input similarity matrix (for visualization)
          "prev_tables": All predecessor tables (for debugging)
    """
    n = len(node_ids)
    n_landmarks = similarity_matrix.shape[1]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    if start not in id_to_idx:
        raise ValueError(
            f"Start node {start} not found in node_ids. "
            f"Available: {node_ids[:5]}..."
        )
    start_idx = id_to_idx[start]

    # Step 1: Initialize scores — start node = 0, all others = -inf
    score = np.full(n, -1e9, dtype=np.float64)  # float64 for precision
    score[start_idx] = 0.0

    # Step 2: Initial Dijkstra — propagate reachability from start
    score, prev1 = dijkstra_transform(score, graph, node_ids, alpha)
    prev_tables = [prev1]

    # DEBUG: Show reachability scores
    print(f"   [DEBUG] After initial Dijkstra (reachability from start={start}):")
    for idx in np.argsort(score)[::-1][:5]:
        print(f"     Node {node_ids[idx]:2d}: score={score[idx]:.6f}")

    # Step 3: For each landmark, add CLIP scores and re-propagate
    for i in range(n_landmarks):
        score = score + similarity_matrix[:, i]
        print(f"   [DEBUG] After adding landmark {i} CLIP scores:")
        for idx in np.argsort(score)[::-1][:5]:
            print(f"     Node {node_ids[idx]:2d}: score={score[idx]:.6f} (clip={similarity_matrix[idx,i]:.4f})")
        score, prev = dijkstra_transform(score, graph, node_ids, alpha)
        print(f"   [DEBUG] After Dijkstra propagation:")
        for idx in np.argsort(score)[::-1][:5]:
            print(f"     Node {node_ids[idx]:2d}: score={score[idx]:.6f}")
        prev_tables.append(prev)

    # Step 4: Backtrack from the best final node
    best_idx = int(np.argmax(score))
    best_node = node_ids[best_idx]

    table_index = len(prev_tables) - 1
    traversal = [(best_node, 0)]

    current_idx = best_idx
    while current_idx != start_idx or table_index > 0:
        pred_idx, table_change = prev_tables[table_index][current_idx]
        table_index += table_change
        current_idx = pred_idx
        traversal.append((node_ids[current_idx], table_change))

    walk = list(reversed(traversal))

    return {
        "walk": walk,
        "score": score,
        "similarity_matrix": similarity_matrix,
        "prev_tables": prev_tables,
    }


def get_walk_nodes(walk: List[Tuple[int, int]]) -> List[int]:
    """Extract just the node IDs from a walk (drop table_change info)."""
    return [node_id for node_id, _ in walk]


def get_landmark_assignments(walk: List[Tuple[int, int]]) -> List[int]:
    """
    Find which nodes in the walk correspond to landmark transitions.
    
    Returns list of node IDs where table_change == -1 (landmark was matched).
    These are the nodes the Dijkstra DP selected as best matching each landmark.
    """
    return [node_id for node_id, change in walk if change == -1]


def compute_walk_distance(walk: List[Tuple[int, int]], graph: nx.Graph) -> float:
    """Compute total Euclidean distance of the planned walk."""
    nodes = get_walk_nodes(walk)
    total = 0.0
    for i in range(len(nodes) - 1):
        if graph.has_edge(nodes[i], nodes[i + 1]):
            total += graph[nodes[i]][nodes[i + 1]]["weight"]
        else:
            # Nodes not directly connected — compute shortest path distance
            try:
                path_len = nx.shortest_path_length(
                    graph, nodes[i], nodes[i + 1], weight="weight"
                )
                total += path_len
            except nx.NetworkXNoPath:
                total += float("inf")
    return total
