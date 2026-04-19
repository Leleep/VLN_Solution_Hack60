"""
CLIP Scorer
===========
Scores graph node images against landmark text using CLIP ViT-L/14.

Adapted from lm_nav/optimal_route.py:nodes_landmarks_similarity()
Key changes:
  - Uses open_clip instead of deprecated openai/CLIP
  - Batch inference instead of per-node loop
  - No fisheye rectification (not needed for TurtleBot3)

Reference (lm_nav original, lines 39-65):
    text_labels = ["A photo of " + desc for desc in landmarks]
    similarity = text_features @ image_features.T
    result[i, :] = np.max(similarity, axis=1)
"""

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False


class CLIPScorer:
    """
    Scores images against textual landmarks using CLIP.
    
    Returns a (N_nodes × N_landmarks) similarity matrix where higher values
    indicate stronger visual-semantic alignment.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: Optional[str] = None,
        prompt_template: str = "A photo of ",
    ):
        """
        Args:
            model_name: CLIP model architecture (default: ViT-L-14)
            pretrained: Pretrained weights source (default: openai)
            device: "cuda" or "cpu" (auto-detected if None)
            prompt_template: Text prefix for landmarks 
                             (paper uses "A photo of ", lm_nav uses same)
        """
        if not HAS_OPEN_CLIP:
            raise ImportError(
                "open_clip_torch is required. Install: pip install open_clip_torch"
            )

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🔍 Loading CLIP {model_name} ({pretrained}) on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.prompt_template = prompt_template
        print(f"✅ CLIP loaded. Prompt template: '{self.prompt_template}{{landmark}}'")

    def _encode_text(self, landmarks: List[str]) -> torch.Tensor:
        """Encode landmark texts to CLIP feature vectors."""
        # Apply prompt template — matching lm_nav/optimal_route.py L46:
        #   text_labels = ["A photo of " + desc for desc in landmarks]
        text_labels = [self.prompt_template + lm for lm in landmarks]
        tokens = self.tokenizer(text_labels).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _encode_images_batch(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> torch.Tensor:
        """Encode a list of PIL images to CLIP feature vectors (batched)."""
        all_features = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            processed = torch.stack([self.preprocess(img) for img in batch])
            processed = processed.to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(processed).float()
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0)

    def score(
        self,
        images: List[Image.Image],
        landmarks: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute CLIP similarity between all images and all landmarks.
        
        Matching the lm_nav approach (optimal_route.py L63):
            similarity = text_features @ image_features.T
            result[i, :] = np.max(similarity, axis=1)
        
        For our case each node has one image, so max is identity.
        
        Args:
            images: List of N PIL images (one per graph node)
            landmarks: List of M landmark strings
            batch_size: Batch size for image encoding
            
        Returns:
            np.ndarray of shape (N, M) — similarity scores.
            Higher = better match. Raw cosine similarity (not softmax).
        """
        if not images or not landmarks:
            return np.zeros((len(images), len(landmarks)))

        print(f"🖼️  Encoding {len(images)} images × {len(landmarks)} landmarks...")

        # Encode text
        text_features = self._encode_text(landmarks)  # (M, D)

        # Encode images in batches
        image_features = self._encode_images_batch(images, batch_size)  # (N, D)

        # Compute similarity matrix: (N, M)
        # Each entry [i, j] = cosine_similarity(image_i, landmark_j)
        # Both must be on CPU for numpy conversion
        text_features_cpu = text_features.cpu()
        similarity = (image_features @ text_features_cpu.T).numpy()

        print(f"✅ Similarity matrix computed: shape {similarity.shape}")
        return similarity

    def score_softmax(
        self,
        images: List[Image.Image],
        landmarks: List[str],
        batch_size: int = 32,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Same as score() but applies softmax normalization per landmark.
        
        This gives P(node | landmark) — a probability distribution over nodes
        for each landmark, as described in the LM-Nav paper.
        
        Args:
            images: List of N PIL images
            landmarks: List of M landmark strings
            batch_size: Batch size for encoding
            temperature: Softmax temperature (lower = sharper)
            
        Returns:
            np.ndarray of shape (N, M) — softmax-normalized scores per column.
        """
        raw_scores = self.score(images, landmarks, batch_size)
        # Softmax per column (per landmark)
        from scipy.special import softmax
        return softmax(raw_scores / temperature, axis=0)

    def compute_patch_heatmap(
        self, image_path: str, landmark_text: str
    ):
        """
        Compute a 16×16 per-patch CLIP similarity heatmap for a single image.

        Uses a forward hook on the last ViT transformer block to extract the
        256 patch token embeddings (before final projection/pooling). Each
        patch's embedding is compared against the landmark text embedding to
        produce a spatial similarity map.

        Args:
            image_path: Path to the RGB image file.
            landmark_text: The landmark text (e.g., "a fridge").

        Returns:
            (px_orig, py_orig, heatmap):
                px_orig, py_orig — pixel (u, v) of the landmark centroid
                                   in original image coordinates.
                                   None if confidence is low.
                heatmap — 16×16 numpy array of per-patch similarities.
        """
        from PIL import Image as PILImage

        orig_img = PILImage.open(image_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        img_tensor = self.preprocess(orig_img).unsqueeze(0).to(self.device)

        # Register a forward hook on the last transformer block
        patch_tokens_out = []
        def _hook(module, inp, out):
            # open_clip ViT: out shape is (seq_len, batch, dim)
            patch_tokens_out.append(out.permute(1, 0, 2).detach())

        # Navigate to the last transformer block
        # open_clip ViT-L/14 structure: model.visual.transformer.resblocks[-1]
        hook_target = self.model.visual.transformer.resblocks[-1]
        hook_handle = hook_target.register_forward_hook(_hook)

        try:
            with torch.no_grad():
                _ = self.model.encode_image(img_tensor)
                text_prompt = self.prompt_template + landmark_text
                text_tokens = self.tokenizer([text_prompt]).to(self.device)
                text_feat = self.model.encode_text(text_tokens)  # [1, dim]
        finally:
            hook_handle.remove()

        if not patch_tokens_out:
            return None, None, np.zeros((16, 16))

        # patch_tokens_out[0] shape: [1, 257, dim] (1 CLS + 256 patches)
        all_tokens = patch_tokens_out[0][0]  # [257, dim]
        patches = all_tokens[1:, :]  # [256, dim] — drop CLS token

        # Normalize and compute cosine similarity
        patches = patches / patches.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sim = (patches @ text_feat.T).squeeze().cpu().numpy()  # [256]
        heatmap = sim.reshape(16, 16)

        # Weighted centroid of patches above the 60th percentile
        threshold = np.percentile(heatmap, 60)
        mask = heatmap > threshold
        if mask.sum() == 0:
            return None, None, heatmap  # low confidence

        rows, cols = np.where(mask)
        weights = heatmap[rows, cols]
        patch_row = np.average(rows, weights=weights)
        patch_col = np.average(cols, weights=weights)

        # Center of patch in 224×224 pixel space
        patch_size = 14
        px_224 = patch_col * patch_size + patch_size / 2.0
        py_224 = patch_row * patch_size + patch_size / 2.0

        # Scale to original image resolution
        px_orig = int(px_224 * orig_w / 224.0)
        py_orig = int(py_224 * orig_h / 224.0)

        return px_orig, py_orig, heatmap

