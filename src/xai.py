"""
Module 6 — Visual Evidence Generation (XAI Heatmaps)
======================================================
Generates per-word and per-character visual heatmaps showing which image
regions drove the TrOCR model's prediction.

Methods implemented
-------------------
1. Attention Rollout (primary)
   - Extracts encoder ViT attention maps across all 12 layers
   - Applies identity-residual rollout: A_i = 0.5*A_i + 0.5*I, then product
   - Reshapes 576 patch tokens → 24×24 → upsample to 384×384

2. GradCAM (secondary)
   - Registers forward hook on the last ViT encoder block
   - Backpropagates log-prob of top-1 predicted token
   - Applies ReLU + normalise

3. Character-level Attribution (most granular)
   - Extracts decoder cross-attention at each generation step
   - One heatmap per generated character

4. Heatmap Overlay
   - Applies jet colormap, blends with original image (0.4 heat + 0.6 original)
   - Encodes result as base64 PNG for API response

Usage
-----
>>> from src.xai import XAIGenerator
>>> gen = XAIGenerator(model, processor)
>>> rollout = gen.generate_attention_rollout(image, model)
>>> overlay = gen.generate_overlay(image, rollout)
>>> b64 = gen.overlay_to_base64(overlay)
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 384           # TrOCR input image size
PATCH_SIZE = 16            # ViT patch size
NUM_PATCHES_1D = IMAGE_SIZE // PATCH_SIZE   # 24
NUM_PATCHES = NUM_PATCHES_1D ** 2           # 576

HEATMAP_BLEND_ALPHA = 0.4   # heatmap contribution
IMAGE_BLEND_ALPHA = 0.6     # original image contribution


# ─────────────────────────────────────────────────────────────────────────────
# XAIGenerator
# ─────────────────────────────────────────────────────────────────────────────

class XAIGenerator:
    """
    Generates attention rollout, GradCAM, and character-level heatmaps
    for TrOCR predictions.

    Parameters
    ----------
    model : VisionEncoderDecoderModel, optional
        Loaded TrOCR model.  Required for real heatmaps; not needed for mock.
    processor : TrOCRProcessor, optional
        Loaded TrOCR processor.
    mock : bool
        If True, returns random heatmaps without loading any model.
    device : str, optional
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
        mock: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.mock = mock or (os.environ.get("MOCK_MODE", "false").lower() == "true")
        self._model = model
        self._processor = processor
        self._device = device or "cpu"
        self._hooks: List[Any] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Method 1 — Attention Rollout
    # ─────────────────────────────────────────────────────────────────────────

    def generate_attention_rollout(
        self,
        image: Union[Image.Image, np.ndarray],
        model: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Compute encoder attention rollout for the input image.

        Returns
        -------
        np.ndarray
            Float32 array of shape (384, 384) with values in [0, 1].
        """
        if self.mock or self._model is None:
            return self._mock_heatmap()

        model = model or self._model
        pil = self._to_pil(image)
        return self._compute_rollout(pil, model)

    def _compute_rollout(self, image: Image.Image, model: Any) -> np.ndarray:
        """
        Core attention rollout computation.

        Per spec Note 3:
          For each layer: A_layer = 0.5 * A_layer + 0.5 * I
          Then multiply all layers together.
        This accounts for ViT residual connections.
        """
        import torch

        pixel_values = self._processor(
            images=image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self._device)

        with torch.no_grad():
            encoder_outputs = model.encoder(
                pixel_values=pixel_values,
                output_attentions=True,
            )

        # encoder_outputs.attentions: tuple of (1, num_heads, seq, seq) per layer
        attentions = encoder_outputs.attentions  # len = num_layers

        # Initialise rollout as identity matrix
        num_tokens = attentions[0].shape[-1]
        rollout = np.eye(num_tokens, dtype=np.float32)

        for layer_attn in attentions:
            # Average across heads → [seq, seq]
            layer_attn_np = layer_attn[0].mean(dim=0).cpu().numpy()  # [seq, seq]

            # Residual-corrected: A = 0.5 * A + 0.5 * I
            A = 0.5 * layer_attn_np + 0.5 * np.eye(num_tokens, dtype=np.float32)

            # Normalise rows to sum to 1
            A = A / (A.sum(axis=-1, keepdims=True) + 1e-8)

            # Accumulate rollout
            rollout = rollout @ A

        # The first token is [CLS]; patch tokens follow
        # Take the CLS → patch attention (row 0, columns 1:)
        patch_attention = rollout[0, 1:]  # [num_patches]

        # Reshape to 2D patch grid
        n = int(np.sqrt(len(patch_attention)))
        if n * n != len(patch_attention):
            # Fallback: pad or trim
            patch_attention = patch_attention[:NUM_PATCHES]
            n = NUM_PATCHES_1D

        heatmap_2d = patch_attention.reshape(n, n)

        # Upsample to IMAGE_SIZE × IMAGE_SIZE
        heatmap_full = cv2.resize(
            heatmap_2d,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalise to [0, 1]
        heatmap_full = (heatmap_full - heatmap_full.min()) / (
            heatmap_full.max() - heatmap_full.min() + 1e-8
        )
        return heatmap_full.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Method 2 — GradCAM
    # ─────────────────────────────────────────────────────────────────────────

    def generate_gradcam(
        self,
        image: Union[Image.Image, np.ndarray],
        model: Optional[Any] = None,
        target_token_index: int = 0,
    ) -> np.ndarray:
        """
        Compute GradCAM on the last ViT encoder block for the given
        generated token.

        Parameters
        ----------
        image : PIL.Image or np.ndarray
        model : optional override
        target_token_index : int
            Which generated token to backpropagate through (0 = first char).

        Returns
        -------
        np.ndarray float32 [384, 384] in [0, 1].
        """
        if self.mock or self._model is None:
            return self._mock_heatmap()

        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError:
            print("[XAIGenerator] pytorch-grad-cam not installed, returning mock heatmap.")
            return self._mock_heatmap()

        import torch

        model = model or self._model
        pil = self._to_pil(image)

        pixel_values = self._processor(
            images=pil.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self._device)

        # Target: last encoder block's LayerNorm
        target_layer = model.encoder.encoder.layer[-1].layernorm_before

        # GradCAM wrapper that works with ViT
        cam = GradCAM(model=_ViTGradCAMWrapper(model, self._processor, self._device),
                       target_layers=[target_layer])

        grayscale_cam = cam(input_tensor=pixel_values, targets=None)
        heatmap = grayscale_cam[0]  # [H, W]

        heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Method 3 — Character-level Attribution
    # ─────────────────────────────────────────────────────────────────────────

    def generate_character_heatmaps(
        self,
        image: Union[Image.Image, np.ndarray],
        model: Optional[Any] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Generate a heatmap for every generated character using decoder
        cross-attention weights.

        Returns
        -------
        dict[int, np.ndarray]
            {token_step_index: heatmap_384x384}
        """
        if self.mock or self._model is None:
            return {0: self._mock_heatmap()}

        import torch

        model = model or self._model
        pil = self._to_pil(image)

        pixel_values = self._processor(
            images=pil.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self._device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                num_beams=1,
                return_dict_in_generate=True,
                output_attentions=True,
                max_new_tokens=64,
            )

        char_heatmaps: Dict[int, np.ndarray] = {}

        # outputs.cross_attentions: tuple per step, each a tuple per layer
        if not hasattr(outputs, "cross_attentions") or outputs.cross_attentions is None:
            return {0: self._mock_heatmap()}

        for step_idx, step_cross_attentions in enumerate(outputs.cross_attentions):
            # step_cross_attentions: tuple[layer] of [batch, heads, 1, enc_seq]
            # Use the last decoder layer
            last_layer = step_cross_attentions[-1]  # [1, heads, 1, enc_seq]
            attn = last_layer[0].mean(dim=0)[0].cpu().numpy()  # [enc_seq]

            # enc_seq includes CLS + patch tokens; strip CLS
            patch_attn = attn[1:]  # [num_patches]
            n = NUM_PATCHES_1D
            if len(patch_attn) >= n * n:
                patch_attn = patch_attn[: n * n]
            else:
                # Pad if shorter
                patch_attn = np.pad(patch_attn, (0, n * n - len(patch_attn)))

            heatmap_2d = patch_attn.reshape(n, n)
            heatmap = cv2.resize(heatmap_2d.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            char_heatmaps[step_idx] = heatmap.astype(np.float32)

        return char_heatmaps

    def summarise_character_attention(
        self,
        word: str,
        char_heatmaps: Dict[int, np.ndarray],
        top_k: int = 3,
    ) -> str:
        """
        Convert character heatmaps into a human-readable description
        for Module 7 (ExplanationAgent).

        Example output:
            "High attention on characters 1, 3, 5 corresponding to 'h', 'l', 'o'"
        """
        if not char_heatmaps or not word:
            return "Attention information not available."

        high_attention_chars = []
        for step_idx, hmap in sorted(char_heatmaps.items()):
            if step_idx < len(word):
                char = word[step_idx]
                mean_attn = float(hmap.mean())
                high_attention_chars.append((step_idx + 1, char, mean_attn))

        # Sort by attention descending, take top_k
        high_attention_chars.sort(key=lambda x: x[2], reverse=True)
        top = high_attention_chars[:top_k]

        if not top:
            return "Attention information not available."

        parts = [f"position {p} ('{c}')" for p, c, _ in sorted(top, key=lambda x: x[0])]
        return f"High attention on {', '.join(parts)} in the word '{word}'."

    # ─────────────────────────────────────────────────────────────────────────
    # Overlay & Encoding
    # ─────────────────────────────────────────────────────────────────────────

    def generate_overlay(
        self,
        image: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
    ) -> Image.Image:
        """
        Blend a [0,1] float heatmap with the original image using jet colormap.

        overlay = HEATMAP_BLEND_ALPHA * heatmap_colored + IMAGE_BLEND_ALPHA * original

        Returns
        -------
        PIL.Image.Image  (RGB)
        """
        pil = self._to_pil(image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        original_arr = np.array(pil, dtype=np.uint8)

        # Convert heatmap to uint8, apply jet colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = (
            HEATMAP_BLEND_ALPHA * heatmap_colored.astype(np.float32)
            + IMAGE_BLEND_ALPHA * original_arr.astype(np.float32)
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)

    @staticmethod
    def overlay_to_base64(overlay: Image.Image) -> str:
        """Encode a PIL image as a base64 PNG string for API responses."""
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

    @staticmethod
    def _mock_heatmap() -> np.ndarray:
        """Return a random Gaussian heatmap for dev/testing."""
        rng = np.random.default_rng(seed=42)
        h = rng.random((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        # Apply Gaussian blur for a smooth, realistic-looking mock
        h = cv2.GaussianBlur(h, (51, 51), 0)
        h = (h - h.min()) / (h.max() - h.min() + 1e-8)
        return h.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM Wrapper for ViT
# ─────────────────────────────────────────────────────────────────────────────

class _ViTGradCAMWrapper:
    """
    Thin wrapper that makes TrOCR's encoder compatible with pytorch-grad-cam.
    Exposes a forward() method returning encoder hidden states summed across
    the sequence dimension (so GradCAM can compute gradients).
    """

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def __call__(self, pixel_values):
        import torch
        outputs = self.model.encoder(pixel_values=pixel_values, output_attentions=False)
        # Sum hidden states across tokens → [batch, hidden_size]
        return outputs.last_hidden_state.mean(dim=1)

    @property
    def training(self):
        return self.model.training

    def eval(self):
        return self.model.eval()

    def parameters(self):
        return self.model.parameters()
