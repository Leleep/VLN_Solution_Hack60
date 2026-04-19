#!/usr/bin/env python3
"""
CLIP + DINOv2 Evaluation Script
===============================
Evaluate how accurately CLIP and/or DINOv2+CLIP predict the correct image
given a text prompt for the AWS Small House topological graph.

Modes:
    clip       — Score full images with CLIP only (default)
    dino+clip  — DINOv2 discovers salient regions, CLIP scores each crop
    compare    — Run both modes side-by-side and compare accuracy

Usage:
    python scripts/evaluate_clip.py --prompt "kitchen"
    python scripts/evaluate_clip.py --prompt "kitchen" --mode dino+clip
    python scripts/evaluate_clip.py --prompt "kitchen" --mode compare
    python scripts/evaluate_clip.py --evaluate-all --mode dino+clip
    python scripts/evaluate_clip.py --evaluate-all --mode compare --show
    python scripts/evaluate_clip.py --prompt "kitchen" --prompt-engineering

Features:
    - CLIP-only: rank images by cosine similarity to text
    - DINOv2+CLIP: region discovery then per-crop CLIP scoring
    - Compare mode: side-by-side accuracy of both approaches
    - Prompt engineering with multi-prompt ensemble
    - Ground truth evaluation with top-k accuracy
    - Matplotlib visualization of top matches
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so `lmnav` can be imported
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lmnav.clip_scorer import CLIPScorer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "aws_house_graph"
OUTPUT_DIR = PROJECT_ROOT / "output" / "clip_eval"

# ---------------------------------------------------------------------------
# Ground Truth Mapping
# ---------------------------------------------------------------------------
# Based on visual inspection of the AWS RoboMaker Small House World.
# Each node is a (position, heading) pair — groups of 3 nodes share the same
# physical position but face different directions (~120° apart).
#
# Mapping: landmark → list of node IDs that depict that landmark.
GROUND_TRUTH: Dict[str, List[int]] = {
    "kitchen": [0, 1, 2, 3, 4, 5],          # pos ~(4.65,-1.95) & (6.1,-2.4) — fridge, stove, hood, cabinets
    "dining table": [0, 1, 2],               # same area has the dining table with blue chairs
    "bedroom": [21, 22, 23, 24, 25, 26],     # pos ~(-7.8,-0.1) & (-7.9,-2.95) — bed, closet, brick wall
    "living room": [12, 13, 14, 15, 16, 17, # pos ~(4.05,0.15) & (-4.05,0.1) — couches, TV
                    30, 31, 32, 33, 34, 35], # pos ~(-5.7,-0.2) & (-1.25,-4.1) — open living area
    "gym": [6, 7, 8, 27, 28, 29],            # pos ~(0.0,1.5) & (1.45,4.2) — exercise ball, weights, bench
    "hallway": [9, 10, 11, 18, 19, 20],      # pos ~(-4.75,-3.55) & (-2.45,-1.0) — doors, corridor
    "bathroom": [36, 37, 38, 39, 40, 41,     # pos ~(8.25,2.0) & (8.3,0.75) & (0.7,-4.0) — corner/window
                 42, 43, 44],
    # Object-level landmarks
    "television": [33, 34, 35, 15, 16, 17],  # TV visible from living room nodes
    "refrigerator": [0, 1, 2, 3, 4, 5],      # fridge in kitchen
    "couch": [12, 13, 14, 15, 16, 17],       # sofa set in living room
    "exercise ball": [6, 7, 8, 27, 28, 29],  # blue ball in gym
    "bed": [21, 22, 23],                      # bed visible at (-7.8,-0.1)
}

# ---------------------------------------------------------------------------
# Prompt Engineering Variations
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "kitchen": [
        "A photo of kitchen",
        "a kitchen",
        "a modern kitchen",
        "a photo of a kitchen",
        "indoor kitchen scene",
        "a kitchen with appliances",
        "kitchen with refrigerator and stove",
    ],
    "bedroom": [
        "A photo of bedroom",
        "a bedroom",
        "a cozy bedroom",
        "a photo of a bedroom",
        "indoor bedroom scene",
        "a bedroom with a bed",
        "bedroom with bed and closet",
    ],
    "living room": [
        "A photo of living room",
        "a living room",
        "a modern living room",
        "a photo of a living room",
        "indoor living room scene",
        "a living room with a couch and television",
        "living room with sofa and TV",
    ],
    "gym": [
        "A photo of gym",
        "a gym",
        "a home gym",
        "a photo of a gym",
        "indoor gym scene",
        "a gym with exercise equipment",
        "gym with weights and exercise ball",
    ],
    "hallway": [
        "A photo of hallway",
        "a hallway",
        "a hallway with doors",
        "a photo of a hallway",
        "indoor hallway scene",
        "a corridor",
        "hallway with glass doors",
    ],
    "bathroom": [
        "A photo of bathroom",
        "a bathroom",
        "a modern bathroom",
        "a photo of a bathroom",
        "indoor bathroom scene",
    ],
    "dining table": [
        "A photo of a dining table",
        "a large table for eating meals",
        "a dining room table with chairs",
        "dining table in a kitchen area",
        "a dining area setup for meals",
        "dinner table with place settings",
    ],
    "study table": [
        "a study desk",
        "a computer desk in a bedroom",
        "a work table against a wall",
        "an office desk without food on it",
        "a small desk for studying or working",
        "a simple table in a bedroom",
    ],
    "study desk": [
        "a study desk",
        "a computer desk in a bedroom",
        "a work table against a wall",
        "an office desk without food on it",
        "a small desk for studying or working",
        "a simple table in a bedroom",
    ],
    "curtains": [
        "window curtains",
        "fabric curtains hanging over a window",
        "window drapes",
        "closed curtains in a room",
        "long curtains covering a window",
    ],
    "television": [
        "A photo of television",
        "a television",
        "a TV on a wall",
        "a photo of a TV screen",
        "a large flat screen TV",
    ],
    "refrigerator": [
        "A photo of refrigerator",
        "a refrigerator",
        "a stainless steel refrigerator",
        "a photo of a fridge",
        "kitchen refrigerator",
    ],
    "couch": [
        "A photo of couch",
        "a couch",
        "a sofa",
        "a photo of a sofa",
        "a living room couch",
        "a large cushioned sofa for sitting",
    ],
    "sofa": [
        "a sofa",
        "a couch in a living room",
        "a large cushioned sofa for sitting",
        "a comfortable sofa",
        "a living room sofa set",
    ],
    "exercise ball": [
        "A photo of exercise ball",
        "an exercise ball",
        "a blue exercise ball",
        "a photo of a fitness ball",
        "gym ball on the floor",
    ],
    "bed": [
        "A photo of bed",
        "a bed",
        "a modern bed",
        "a photo of a bed",
        "a bed with pillows",
    ],
    "chimney": [
        "a kitchen exhaust hood above a stove",
        "a range hood mounted over a cooktop",
        "an electric chimney in a kitchen",
        "a kitchen hood for ventilation",
        "a stainless steel range hood above the stove",
        "a kitchen chimney extractor fan",
    ],
    "wall art": [
        "a framed picture hanging on a wall",
        "canvas art on an indoor wall",
        "a decorative painting on a wall",
        "a photo of wall art",
        "framed artwork inside a house",
    ],
    "air conditioner": [
        "an air conditioner unit mounted on a wall",
        "a wall-mounted AC unit",
        "a split air conditioner indoors",
        "an indoor air conditioning unit",
        "a white air conditioner on a bedroom wall",
    ],
    "window": [
        "an indoor view of a window",
        "a glass window looking outside",
        "a window with blinds or curtains",
        "a window in a room",
        "sunlight coming through a window",
    ],
    "chair": [
        "a chair",
        "a single chair for sitting",
        "an orange chair",
        "a photo of a chair in a room",
        "a modern chair",
    ],
    "staircase": [
        "a staircase inside a house",
        "indoor stairs",
        "a flight of stairs going up",
        "a wooden staircase",
        "stairs with a railing",
    ],
}


# ============================================================================
# Helper functions
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize a text prompt: lowercase, strip articles."""
    text = text.lower().strip()
    # Remove leading articles
    text = re.sub(r"^(a |an |the )", "", text)
    return text


def load_node_images(data_dir: Path) -> Tuple[List[int], List[Image.Image], List[str]]:
    """
    Load all node_*.png images from the data directory.

    Returns:
        node_ids: Sorted list of node integer IDs
        images: Corresponding PIL images
        filenames: Corresponding filenames (e.g. "node_000.png")
    """
    node_files = sorted(data_dir.glob("node_*.png"))
    if not node_files:
        print(f"❌ No node images found in {data_dir}")
        sys.exit(1)

    node_ids = []
    images = []
    filenames = []
    for f in node_files:
        # Extract ID from node_XXX.png
        stem = f.stem  # "node_000"
        nid = int(stem.split("_")[1])
        node_ids.append(nid)
        images.append(Image.open(f).convert("RGB"))
        filenames.append(f.name)

    print(f"📂 Loaded {len(images)} node images from {data_dir}")
    return node_ids, images, filenames


# ============================================================================
# DINOv2 Region Extractor
# ============================================================================

class DINOv2RegionExtractor:
    """
    Uses DINOv2 self-attention maps to discover salient regions in images.

    Pipeline:
        1. Forward pass → extract [CLS] attention from last layer
        2. Threshold attention → binary mask
        3. Find connected components → bounding boxes
        4. Crop original image at each bounding box
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        device: Optional[str] = None,
    ):
        import torch as _torch
        self._torch = _torch

        if device is None:
            self.device = "cuda" if _torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🦕 Loading DINOv2 {model_name} on {self.device}...")
        self.model = _torch.hub.load(
            "facebookresearch/dinov2", model_name, verbose=False
        )
        self.model = self.model.to(self.device).eval()
        self.patch_size = 14
        self.model_name = model_name

        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.crop_size = 518
        self.n_patches = self.crop_size // self.patch_size  # 37
        print(f"✅ DINOv2 loaded. Patch grid: {self.n_patches}×{self.n_patches}")

    def get_attention_map(self, image: Image.Image) -> np.ndarray:
        """
        Extract the [CLS] token attention map from the last self-attention layer.

        Returns:
            np.ndarray of shape (n_patches, n_patches) with values in [0, 1].
        """
        _torch = self._torch
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with _torch.no_grad():
            # Register a hook on the last attention layer to capture attention weights
            attn_weights = []

            def hook_fn(module, input, output):
                # output is a tuple; for DINOv2, attention weights are computed internally
                # We need to recompute them from Q, K
                pass

            # Use the model's get_intermediate_layers or forward_features
            out = self.model.forward_features(img_tensor)

            # DINOv2 stores attention in the last block via get_last_selfattention
            # Fallback: use forward with attention output if available
            try:
                attn = self.model.get_last_selfattention(img_tensor)
                # attn shape: (batch, n_heads, n_tokens, n_tokens)
                # Take mean over heads, extract [CLS] row (index 0)
                attn_map = attn[:, :, 0, 1:].mean(dim=1)  # (1, n_patches^2)
            except AttributeError:
                # Fallback for models without get_last_selfattention
                # Use patch token norms as a proxy for saliency
                patch_tokens = out["x_norm_patchtokens"]  # (1, N, D)
                attn_map = patch_tokens.norm(dim=-1)  # (1, N)

        attn_map = attn_map.squeeze(0).cpu().numpy()  # (n_patches^2,)
        n = self.n_patches
        attn_map = attn_map.reshape(n, n)

        # Normalize to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map

    def extract_regions(
        self,
        image: Image.Image,
        num_regions: int = 4,
        min_region_frac: float = 0.03,
        attn_threshold_pct: float = 60.0,
    ) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Extract salient region crops from an image using DINOv2 attention.

        Args:
            image: Input PIL image
            num_regions: Maximum number of regions to extract
            min_region_frac: Minimum region size as fraction of image area
            attn_threshold_pct: Percentile threshold for attention binarization

        Returns:
            crops: List of cropped PIL images
            boxes: List of (x1, y1, x2, y2) bounding boxes in original image coords
            attn_map: The raw attention map (n_patches × n_patches)
        """
        from scipy import ndimage

        attn_map = self.get_attention_map(image)
        n = self.n_patches

        # Binarize attention map
        threshold = np.percentile(attn_map, attn_threshold_pct)
        binary = (attn_map > threshold).astype(np.uint8)

        # Find connected components
        labeled, num_features = ndimage.label(binary)

        # Get bounding boxes for each component, sorted by attention mass
        regions = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            mass = attn_map[mask].sum()
            ys, xs = np.where(mask)
            # Patch-space bounding box
            py1, py2 = ys.min(), ys.max() + 1
            px1, px2 = xs.min(), xs.max() + 1
            area_frac = mask.sum() / (n * n)
            if area_frac >= min_region_frac:
                regions.append((mass, px1, py1, px2, py2))

        # Sort by attention mass (descending) and take top K
        regions.sort(key=lambda r: r[0], reverse=True)
        regions = regions[:num_regions]

        # If no regions found, fall back to full image
        if not regions:
            regions = [(1.0, 0, 0, n, n)]

        # Convert patch-space boxes to pixel-space crops
        # The image was center-cropped to crop_size during preprocessing
        orig_w, orig_h = image.size
        scale_x = orig_w / n
        scale_y = orig_h / n

        crops = []
        boxes = []
        for _, px1, py1, px2, py2 in regions:
            # Add padding (10% of box size)
            pad_x = max(1, int((px2 - px1) * 0.1))
            pad_y = max(1, int((py2 - py1) * 0.1))
            px1_pad = max(0, px1 - pad_x)
            py1_pad = max(0, py1 - pad_y)
            px2_pad = min(n, px2 + pad_x)
            py2_pad = min(n, py2 + pad_y)

            # To pixel coords
            x1 = int(px1_pad * scale_x)
            y1 = int(py1_pad * scale_y)
            x2 = int(px2_pad * scale_x)
            y2 = int(py2_pad * scale_y)

            # Ensure minimum crop size
            min_dim = 32
            if x2 - x1 < min_dim:
                x2 = min(orig_w, x1 + min_dim)
            if y2 - y1 < min_dim:
                y2 = min(orig_h, y1 + min_dim)

            crop = image.crop((x1, y1, x2, y2))
            crops.append(image.crop((x1, y1, x2, y2)))
            boxes.append((x1, y1, x2, y2))

        return crops, boxes, attn_map



def rank_images_hybrid(
    scorer: CLIPScorer,
    dino: DINOv2RegionExtractor,
    images: List[Image.Image],
    node_ids: List[int],
    filenames: List[str],
    prompt: str,
    num_regions: int = 4,
) -> List[Tuple[int, str, float]]:
    """
    Hybrid DINOv2 + CLIP ranking — Auto Best-Of.

    For each image:
      1. Compute CLIP score on the full image (global)
      2. DINOv2 extracts salient region crops
      3. CLIP scores each crop (local)
      4. Node score = max(global_score, best_local_score)

    This ensures CLIP-only wins when it's better (most cases),
    and DINOv2 crop wins when it finds something CLIP missed
    (e.g., small objects like a television).

    Returns list of (node_id, filename, best_score) sorted descending.
    """
    from tqdm import tqdm

    print(f"\n🔤 Text query: \"A photo of {prompt}\"")
    print(f"🦕 Auto mode: best-of CLIP-global vs DINOv2-crops...")

    # Multi-prompt ensemble text features
    text_features = _get_ensemble_text_features(scorer, prompt)  # (1, D)

    results = []
    for idx, (nid, fname, img) in enumerate(
        tqdm(zip(node_ids, filenames, images), total=len(images), desc="DINO+CLIP")
    ):
        # Extract regions
        crops, boxes, _ = dino.extract_regions(img, num_regions=num_regions)

        # Full image first, then crops
        all_candidates = [img] + crops

        # Encode all candidates with CLIP
        img_features = scorer._encode_images_batch(all_candidates)

        # Compute similarities
        sims = (img_features @ text_features.cpu().T).numpy().flatten()

        # Best-of: max(global_score, best_crop_score)
        best_score = float(sims.max())
        results.append((nid, fname, best_score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def _get_ensemble_text_features(scorer: CLIPScorer, prompt: str):
    """
    Get CLIP text features using multi-prompt ensemble averaging.

    If the prompt has entries in PROMPT_TEMPLATES, encode all variations
    and average them (L2-normalized). This is the #1 technique from
    CLIP's original paper for improving zero-shot accuracy.

    For prompts WITHOUT templates, auto-generates multiple variations
    so every prompt benefits from ensemble scoring.

    IMPORTANT: All variations are full prompt text, so we encode them
    directly via the tokenizer (NOT scorer._encode_text which would
    double-prefix with "A photo of ").
    """
    import torch

    normalized = normalize_text(prompt)
    variations = PROMPT_TEMPLATES.get(normalized)

    if not variations or len(variations) < 2:
        # Auto-generate variations for any unknown prompt
        # Uses patterns proven effective in CLIP's original paper
        p = prompt.strip()
        variations = [
            f"A photo of {p}",
            f"a {p}",
            f"a photo of a {p}",
            f"a photo of the {p}",
            f"a {p} in a house",
            f"indoor photo showing a {p}",
        ]
        print(f"  🔧 Auto-ensemble: generated {len(variations)} variations for \"{p}\"")

    # Encode variations directly — they already contain the full prompt text
    tokens = scorer.tokenizer(variations).to(scorer.device)
    with torch.no_grad():
        all_features = scorer.model.encode_text(tokens).float()
    all_features /= all_features.norm(dim=-1, keepdim=True)

    # Average and re-normalize
    ensemble = all_features.mean(dim=0, keepdim=True)  # (1, D)
    ensemble = ensemble / ensemble.norm(dim=-1, keepdim=True)
    print(f"  🎯 Ensemble: averaged {len(variations)} prompt variations")
    return ensemble

# ============================================================================
# Graph Score Smoothing
# ============================================================================

def _build_spatial_neighbors(
    data_dir: Path,
    distance_threshold: float = 3.0,
) -> Dict[int, List[int]]:
    """
    Build a neighbor map from chamber_nodes.json.

    Two nodes are neighbors if:
      1. They share the same parent_position (co-located views, different angles)
      2. Their physical positions are within distance_threshold meters

    Returns dict mapping node_id -> [neighbor_ids].
    """
    import json
    import math

    nodes_file = data_dir / "chamber_nodes.json"
    if not nodes_file.exists():
        return {}

    with open(nodes_file) as f:
        nodes = json.load(f)

    # Build node info: id -> (x, y, parent_position)
    node_info = {}
    for i, node in enumerate(nodes):
        node_info[i] = {
            "x": node["x"],
            "y": node["y"],
            "parent": node.get("parent_position", ""),
        }

    # Build neighbor map
    neighbors: Dict[int, List[int]] = {i: [] for i in node_info}
    node_ids = list(node_info.keys())

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            ni, nj = node_ids[i], node_ids[j]
            info_i, info_j = node_info[ni], node_info[nj]

            # Same physical position (different camera angles)
            same_parent = (info_i["parent"] == info_j["parent"] and info_i["parent"] != "")

            # Close in physical space
            dx = info_i["x"] - info_j["x"]
            dy = info_i["y"] - info_j["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            close = dist < distance_threshold

            if same_parent or close:
                neighbors[ni].append(nj)
                neighbors[nj].append(ni)

    return neighbors


def smooth_scores(
    ranked: List[Tuple[int, str, float]],
    neighbors: Dict[int, List[int]],
    alpha: float = 0.7,
) -> List[Tuple[int, str, float]]:
    """
    Apply graph-based score smoothing.

    smoothed_score = alpha * own_score + (1 - alpha) * mean(neighbor_scores)

    This penalizes isolated false positives: if a node has a high score
    but none of its spatial neighbors do, its score gets pulled down.

    Args:
        ranked: Original ranked list of (node_id, filename, score)
        neighbors: Dict mapping node_id -> [neighbor_ids]
        alpha: Weight for own score vs neighbor average (0.7 = 70% own)

    Returns:
        Re-ranked list with smoothed scores.
    """
    if not neighbors:
        return ranked

    # Build score lookup
    score_map = {nid: score for nid, _, score in ranked}

    smoothed = []
    for nid, fname, score in ranked:
        nbrs = neighbors.get(nid, [])
        if nbrs:
            nbr_scores = [score_map.get(n, 0.0) for n in nbrs]
            avg_nbr = sum(nbr_scores) / len(nbr_scores)
            new_score = alpha * score + (1 - alpha) * avg_nbr
        else:
            new_score = score
        smoothed.append((nid, fname, new_score))

    smoothed.sort(key=lambda x: x[2], reverse=True)
    return smoothed



def compute_accuracy(
    ranked: List[Tuple[int, str, float]],
    ground_truth_ids: List[int],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Compute Top-K accuracy: is any ground-truth node in the top-K predictions?

    Returns dict like {"top_1": 1.0, "top_3": 1.0, "top_5": 0.0, ...}
    """
    results = {}
    gt_set = set(ground_truth_ids)
    for k in k_values:
        top_k_ids = {r[0] for r in ranked[:k]}
        hit = len(top_k_ids & gt_set) > 0
        results[f"top_{k}"] = 1.0 if hit else 0.0
    return results


def print_rankings(
    ranked: List[Tuple[int, str, float]],
    prompt: str,
    ground_truth_ids: Optional[List[int]] = None,
    top_n: int = 5,
):
    """Print the top-N ranked images with scores and ground-truth markers."""
    gt_set = set(ground_truth_ids) if ground_truth_ids else set()

    print(f"\n{'='*60}")
    print(f"  CLIP Rankings for: \"{prompt}\"")
    print(f"{'='*60}")

    for i, (nid, fname, score) in enumerate(ranked[:top_n]):
        marker = "  ✅ GT" if nid in gt_set else ""
        print(f"  Rank {i+1}: {fname}  (score: {score:.4f}){marker}")

    if gt_set:
        print(f"\n  Ground truth nodes: {sorted(gt_set)}")
        # Find where GT nodes appear in the ranking
        for nid in sorted(gt_set):
            rank_pos = next(
                (i + 1 for i, (rid, _, _) in enumerate(ranked) if rid == nid), None
            )
            if rank_pos is not None:
                score = next(s for rid, _, s in ranked if rid == nid)
                marker = "✅" if rank_pos <= top_n else "⚠️"
                print(f"    {marker} node_{nid:03d} → rank {rank_pos} (score: {score:.4f})")

    print(f"{'='*60}\n")


def print_all_scores(
    ranked: List[Tuple[int, str, float]],
    prompt: str,
    ground_truth_ids: Optional[List[int]] = None,
):
    """Print similarity scores for ALL nodes (debugging)."""
    gt_set = set(ground_truth_ids) if ground_truth_ids else set()

    print(f"\n{'─'*60}")
    print(f"  Full Similarity Distribution for: \"{prompt}\"")
    print(f"{'─'*60}")

    scores = [s for _, _, s in ranked]
    print(f"  Min: {min(scores):.4f}  Max: {max(scores):.4f}  "
          f"Mean: {np.mean(scores):.4f}  Std: {np.std(scores):.4f}")
    print()

    for i, (nid, fname, score) in enumerate(ranked):
        bar_len = int((score - min(scores)) / (max(scores) - min(scores) + 1e-8) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ← GT" if nid in gt_set else ""
        print(f"  {fname} │{bar}│ {score:.4f}{marker}")

    print(f"{'─'*60}\n")


def display_top_matches(
    ranked: List[Tuple[int, str, float]],
    data_dir: Path,
    prompt: str,
    ground_truth_ids: Optional[List[int]] = None,
    top_n: int = 5,
    save_path: Optional[Path] = None,
):
    """Display top-N matching images using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("⚠️  matplotlib not available. Skipping visualization.")
        return

    gt_set = set(ground_truth_ids) if ground_truth_ids else set()

    fig, axes = plt.subplots(1, min(top_n, len(ranked)), figsize=(4 * top_n, 5))
    if top_n == 1:
        axes = [axes]

    fig.suptitle(f'CLIP Top-{top_n} Matches for "{prompt}"', fontsize=14, fontweight="bold")

    for i, (nid, fname, score) in enumerate(ranked[:top_n]):
        img = Image.open(data_dir / fname)
        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")

        is_gt = nid in gt_set
        color = "#2ecc71" if is_gt else "#e74c3c"
        label = f"Rank {i+1}: {fname}\nScore: {score:.4f}"
        if is_gt:
            label += "\n✅ Ground Truth"

        ax.set_title(label, fontsize=9, color=color, fontweight="bold")

        # Add colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Visualization saved: {save_path}")

    plt.show()



# ============================================================================
# Full evaluation across all ground-truth landmarks
# ============================================================================

def evaluate_all(
    scorer: CLIPScorer,
    images: List[Image.Image],
    node_ids: List[int],
    filenames: List[str],
    show: bool = False,
    data_dir: Path = DEFAULT_DATA_DIR,
    dino: Optional[DINOv2RegionExtractor] = None,
    neighbors: Dict[int, List[int]] = None,
):
    """Run evaluation across all ground-truth landmarks using optimal pipeline."""
    print("\n" + "=" * 70)
    print(f"  FULL EVALUATION — Auto (DINOv2 + CLIP) + Graph Smoothing")
    print("=" * 70)

    room_landmarks = ["kitchen", "bedroom", "living room", "gym", "hallway",
                      "wall art", "dining table", "television", "refrigerator",
                      "couch", "exercise ball", "bed", "chair", "chimney", "air conditioner"]

    all_accuracies = {f"top_{k}": [] for k in [1, 3, 5, 10]}

    for landmark in room_landmarks:
        gt_ids = GROUND_TRUTH.get(landmark, [])
        if not gt_ids:
            continue

        ranked = rank_images_hybrid(
            scorer, dino, images, node_ids, filenames, landmark
        )

        # Apply graph smoothing
        if neighbors:
            ranked = smooth_scores(ranked, neighbors)

        print_rankings(ranked, landmark, gt_ids, top_n=5)
        acc = compute_accuracy(ranked, gt_ids)

        for key in all_accuracies:
            all_accuracies[key].append(acc[key])

        if show:
            save_name = landmark.replace(" ", "_")
            display_top_matches(
                ranked, data_dir, landmark, gt_ids, top_n=5,
                save_path=OUTPUT_DIR / f"top5_{save_name}_auto.png",
            )

    # --- Aggregate Results ---
    print("\n" + "=" * 70)
    print("  AGGREGATE ACCURACY (Auto + Graph Smoothing)")
    print("=" * 70)
    for key, values in all_accuracies.items():
        if values:
            mean_acc = np.mean(values) * 100
            print(f"  {key.replace('_', '-'):>8s} accuracy: {mean_acc:6.1f}%  "
                  f"({sum(v == 1.0 for v in values)}/{len(values)} landmarks hit)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Auto DINOv2+CLIP image-text matching with Graph Smoothing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/evaluate_clip.py --prompt "kitchen"
  python scripts/evaluate_clip.py --evaluate-all
        """,
    )
    parser.add_argument(
        "--prompt", "-p", type=str, default=None,
        help="Text prompt to evaluate (e.g. 'kitchen', 'bedroom')",
    )
    parser.add_argument(
        "--evaluate-all", "-a", action="store_true",
        help="Evaluate all ground-truth landmarks",
    )
    parser.add_argument(
        "--show", "-s", action="store_true",
        help="Display/save top matches with matplotlib",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print similarity scores for ALL nodes",
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of top matches to display (default: 5)",
    )
    parser.add_argument(
        "--num-regions", type=int, default=4,
        help="Number of DINOv2 regions per image (default: 4)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
        help=f"Path to graph data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' or 'cpu' (auto-detected)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not args.prompt and not args.evaluate_all:
        parser.error("Specify --prompt <text> or --evaluate-all")

    # --- Load data ---
    node_ids, images, filenames = load_node_images(data_dir)

    # --- Initialize models ---
    scorer = CLIPScorer(device=args.device)
    dino = DINOv2RegionExtractor(device=args.device)

    # --- Build spatial neighbor graph for smoothing ---
    neighbors = _build_spatial_neighbors(data_dir)
    print(f"🗺️  Graph smoothing enabled: {len(neighbors)} nodes with spatial neighbors")

    # --- Single prompt evaluation ---
    if args.prompt:
        prompt = args.prompt.strip()
        normalized = normalize_text(prompt)
        gt_ids = GROUND_TRUTH.get(normalized)

        ranked = rank_images_hybrid(
            scorer, dino, images, node_ids, filenames, prompt,
            num_regions=args.num_regions,
        )

        # Apply graph smoothing
        if neighbors:
            ranked = smooth_scores(ranked, neighbors)

        print_rankings(ranked, prompt, gt_ids, top_n=args.top_n)

        if gt_ids:
            acc = compute_accuracy(ranked, gt_ids)
            print(f"  📊 Accuracy (Auto + Graph Smooth):")
            for key, val in acc.items():
                status = "✅" if val > 0 else "❌"
                print(f"    {key.replace('_', '-'):>8s}: {val:.0%}  {status}")
            print()

        if args.verbose:
            print_all_scores(ranked, prompt, gt_ids)

        if args.show:
            save_name = normalized.replace(" ", "_")
            display_top_matches(
                ranked, data_dir, prompt, gt_ids, top_n=args.top_n,
                save_path=OUTPUT_DIR / f"top{args.top_n}_{save_name}_auto.png",
            )

    # --- Full evaluation ---
    if args.evaluate_all:
        evaluate_all(
            scorer, images, node_ids, filenames,
            show=args.show,
            data_dir=data_dir,
            dino=dino,
            neighbors=neighbors,
        )


if __name__ == "__main__":
    main()