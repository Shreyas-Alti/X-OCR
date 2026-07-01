"""
Module 2 — Document Layout Understanding
==========================================
Identifies semantic regions (paragraphs, headers, tables) using LayoutLMv3,
then crops each region for independent OCR processing.

Key design decisions
--------------------
* Uses microsoft/layoutlmv3-base via HuggingFace transformers.
* Fine-tuned on FUNSD (labels: header, question, answer, other).
* Reading order: sort regions by (y1, x1); multi-column handled via
  x1-clustering heuristic.
* Falls back to treating the full image as a single "other" region if
  LayoutLMv3 is unavailable or MOCK_MODE=true.

Usage
-----
>>> from src.layout import LayoutAnalyser
>>> analyser = LayoutAnalyser()
>>> regions = analyser.analyse(pil_image)
>>> for r in regions:
...     print(r["region_type"], r["bbox"])
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

# Lazy imports — only loaded when real model is needed
_transformers_available = False
try:
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    import torch
    _transformers_available = True
except ImportError:
    pass

# Configure pytesseract path (needed when tesseract is installed via conda)
try:
    import pytesseract
    _tess_cmd = os.environ.get(
        "TESSERACT_CMD",
        r"C:\Users\shreyas.bairyks\miniconda3\envs\xocr\Library\bin\tesseract.exe",
    )
    if os.path.isfile(_tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Default 4-label flat scheme (used with base model / when no labels.json found).
# When a fine-tuned checkpoint is loaded, the label list is read from
# labels.json saved alongside the checkpoint by notebook 02b.
# This avoids the flat-vs-BIO mismatch if the notebook used the full BIO tagset.
FUNSD_LABELS = ["other", "header", "question", "answer"]
LABEL2ID = {l: i for i, l in enumerate(FUNSD_LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(FUNSD_LABELS)}

# Model names
DEFAULT_MODEL_NAME = "microsoft/layoutlmv3-base"
FINETUNED_PATH = os.environ.get("LAYOUTLMV3_FINETUNED_PATH", "models/layoutlmv3_finetuned")

# Column clustering: if two boxes have x1 values differing by < this fraction
# of page width, they are considered same column.
COLUMN_X1_THRESHOLD_FRACTION = 0.3


def _has_model_files(path: str) -> bool:
    """Return True only if path contains real HuggingFace model weight files.

    An empty local directory (e.g. only .gitkeep) returns False so the caller
    falls back to the HuggingFace Hub instead of failing silently.
    """
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    model_files = {"pytorch_model.bin", "model.safetensors", "config.json"}
    return bool(files & model_files)


def _load_label_scheme(checkpoint_dir: str) -> tuple:
    """
    Load label list from labels.json in the checkpoint directory.
    Notebook 02b saves this file alongside the model weights so the
    label scheme is always consistent with the fine-tuned checkpoint.

    Falls back to FUNSD_LABELS (4-label flat scheme) if the file is absent.
    """
    label_file = os.path.join(checkpoint_dir, "labels.json")
    if os.path.isfile(label_file):
        import json
        with open(label_file) as f:
            labels = json.load(f)
        label2id = {l: i for i, l in enumerate(labels)}
        id2label  = {i: l for i, l in enumerate(labels)}
        print(f"[LayoutAnalyser] Loaded {len(labels)}-label scheme from {label_file}")
        return labels, label2id, id2label
    return FUNSD_LABELS, LABEL2ID, ID2LABEL


# ─────────────────────────────────────────────────────────────────────────────
# LayoutAnalyser
# ─────────────────────────────────────────────────────────────────────────────

class LayoutAnalyser:
    """
    Detects and crops semantic regions in a document image.

    Parameters
    ----------
    model_path : str, optional
        Path to a fine-tuned LayoutLMv3 checkpoint.  If the path does not
        exist, falls back to the HuggingFace Hub base model.
    mock : bool
        If True, skip model loading and return the full image as one region.
        Useful for development without a GPU.
    device : str, optional
        "cuda" or "cpu".  Defaults to auto-detect.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        mock: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.mock = mock or (os.environ.get("MOCK_MODE", "false").lower() == "true")
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None

        if not self.mock and _transformers_available:
            self._device = device or ("cuda" if _cuda_available() else "cpu")
            self._load_model(model_path or FINETUNED_PATH)
        else:
            self._device = "cpu"

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect semantic regions in a document image.

        Parameters
        ----------
        image : PIL.Image.Image
            Full document page (RGB or grayscale — converted internally).

        Returns
        -------
        list[dict]
            Each dict has keys:
              - region_type : str  ("header", "question", "answer", "other")
              - bbox        : list[int]  [x1, y1, x2, y2]
              - cropped_image : PIL.Image.Image
        """
        image = image.convert("RGB")

        if self.mock or self._model is None:
            return self._fallback_single_region(image)

        return self._run_layoutlmv3(image)

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: str) -> None:
        """Load processor and model from local path or HuggingFace Hub.

        Falls back to the HuggingFace Hub base model when the local
        checkpoint directory is empty or missing model weight files.

        If the checkpoint directory contains a labels.json file (written
        by notebook 02b after fine-tuning), the label scheme is loaded from
        it. This handles the BIO vs flat-label mismatch between the base model
        (4-label) and the fine-tuned model (7-label BIO tagset from FUNSD).
        """
        import torch
        src = path if _has_model_files(path) else DEFAULT_MODEL_NAME

        # Load label scheme — from labels.json if present, else default 4-label
        checkpoint_for_labels = path if _has_model_files(path) else ""
        labels, label2id, id2label = _load_label_scheme(checkpoint_for_labels)

        try:
            self._processor = LayoutLMv3Processor.from_pretrained(src, apply_ocr=True)
            self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                src,
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id,
            ).to(self._device)
            self._model.eval()
            self._id2label = id2label   # store for inference
        except Exception as exc:
            # Graceful degradation: fall back to single-region mode
            print(f"[LayoutAnalyser] Model load failed ({exc}). Using fallback.")
            self._model = None
            self._id2label = ID2LABEL

    # ── LayoutLMv3 Inference ──────────────────────────────────────────────────

    def _run_layoutlmv3(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run LayoutLMv3 and return sorted, cropped regions."""
        import torch

        encoding = self._processor(image, return_tensors="pt", truncation=True)
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)

        logits = outputs.logits  # [1, seq_len, num_labels]
        predictions = logits.argmax(dim=-1)[0].tolist()  # [seq_len]

        # LayoutLMv3Processor provides word-level boxes when apply_ocr=True
        # boxes are in (x1, y1, x2, y2) format, normalised to 0-1000
        boxes = encoding.get("bbox", None)
        if boxes is None:
            return self._fallback_single_region(image)

        boxes_np = boxes[0].cpu().numpy()  # [seq_len, 4]
        img_w, img_h = image.size

        # Aggregate consecutive tokens with same label into regions
        regions_raw = self._aggregate_regions(predictions, boxes_np, img_w, img_h)

        # Sort by reading order and crop
        regions = self._sort_and_crop(regions_raw, image)
        return regions

    # ── Region Aggregation ────────────────────────────────────────────────────

    @staticmethod
    def _aggregate_regions(
        predictions: list,
        boxes_np: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge consecutive tokens that share the same predicted label.
        Returns list of {region_type, bbox} dicts with pixel coordinates.
        """
        regions: List[Dict[str, Any]] = []
        if len(predictions) == 0:
            return regions

        current_label = predictions[0]
        current_boxes = [boxes_np[0]]

        def flush(label_id: int, token_boxes: list) -> None:
            stacked = np.stack(token_boxes, axis=0)
            x1 = int(stacked[:, 0].min() * img_w / 1000)
            y1 = int(stacked[:, 1].min() * img_h / 1000)
            x2 = int(stacked[:, 2].max() * img_w / 1000)
            y2 = int(stacked[:, 3].max() * img_h / 1000)
            if x2 > x1 and y2 > y1:
                regions.append({
                    "region_type": ID2LABEL.get(label_id, "other"),
                    "bbox": [x1, y1, x2, y2],
                })

        for pred, box in zip(predictions[1:], boxes_np[1:]):
            if pred == current_label:
                current_boxes.append(box)
            else:
                flush(current_label, current_boxes)
                current_label = pred
                current_boxes = [box]
        flush(current_label, current_boxes)

        return regions

    # ── Reading Order Sort & Crop ─────────────────────────────────────────────

    def _sort_and_crop(
        self,
        regions_raw: List[Dict[str, Any]],
        image: Image.Image,
    ) -> List[Dict[str, Any]]:
        """
        Sort regions in reading order (y1 first, then x1) and add crops.

        For multi-column documents, x1-clustering groups columns so that
        reading order respects column boundaries.
        """
        img_w, _ = image.size
        col_threshold = img_w * COLUMN_X1_THRESHOLD_FRACTION

        def sort_key(r: Dict[str, Any]) -> tuple:
            x1, y1, _, _ = r["bbox"]
            # Assign column index by x1 clustering
            col = round(x1 / col_threshold) if col_threshold > 0 else 0
            return (col, y1, x1)

        sorted_regions = sorted(regions_raw, key=sort_key)

        result = []
        for r in sorted_regions:
            x1, y1, x2, y2 = r["bbox"]
            crop = image.crop((x1, y1, x2, y2))
            result.append({
                "region_type": r["region_type"],
                "bbox": r["bbox"],
                "cropped_image": crop,
            })
        return result

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_single_region(image: Image.Image) -> List[Dict[str, Any]]:
        """Return the whole image as a single 'other' region."""
        w, h = image.size
        return [{
            "region_type": "other",
            "bbox": [0, 0, w, h],
            "cropped_image": image.copy(),
        }]

    # ── Training Helper ───────────────────────────────────────────────────────

    @classmethod
    def train_on_funsd(
        cls,
        output_dir: str = "models/layoutlmv3_finetuned",
        num_epochs: int = 5,
        learning_rate: float = 2e-5,
        batch_size: int = 8,
    ) -> None:
        """
        Fine-tune LayoutLMv3 on the FUNSD dataset.

        Requires: transformers, datasets, torch with CUDA recommended.
        See notebooks/02_trocr_training.ipynb for a step-by-step guide.
        """
        if not _transformers_available:
            raise ImportError("Install transformers and datasets to fine-tune.")

        from datasets import load_dataset
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            TrainingArguments,
            Trainer,
        )
        import torch

        print("[LayoutAnalyser] Loading FUNSD dataset …")
        dataset = load_dataset("nielsr/funsd")

        processor = LayoutLMv3Processor.from_pretrained(DEFAULT_MODEL_NAME, apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            DEFAULT_MODEL_NAME,
            num_labels=len(FUNSD_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        # NOTE: Full tokenise-and-align implementation with seqeval F1 metric
        # and EarlyStoppingCallback is in notebooks/02b_layoutlmv3_training.ipynb
        # This stub shows the Trainer structure; run the notebook for complete training.
        print("[LayoutAnalyser] Training stub — see notebooks/02b_layoutlmv3_training.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
