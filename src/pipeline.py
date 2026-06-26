"""
Full Pipeline Orchestrator
===========================
Connects all 7 modules into a single callable pipeline.

  OCRPipeline.run(image_bytes) → OCRResult

Error handling strategy
-----------------------
* Module 1 (preprocess) failure → use raw image
* Module 2 (layout) failure     → treat full image as one region
* Module 5 (context) failure    → fall back to visual-only top-1
* Module 6 (XAI) failure        → omit heatmap (return empty string)
* Module 7 (explanation) failure → return fallback placeholder

Word segmentation (CRITICAL)
-----------------------------
TrOCR is a *word-level* model fine-tuned on single-word IAM crops.
Feeding it a paragraph produces garbage. _segment_and_recognise
uses OpenCV connected-components on a binarised + horizontally-dilated
image to split regions into individual word crops before passing each
to TrOCR independently.

Tunable via env var:
  WORD_SEG_KERNEL_WIDTH  (int, default 15)
    Larger → good for widely spaced handwriting.
    Smaller (10–12) → dense printed text.

Context window (global)
-----------------------
Context is built from a *global* flat word list across ALL regions in
reading order — not per-region. This prevents the cold-start problem
where the first/last words of short regions have zero context.

Timing
------
Wall-clock seconds for each module are stored in OCRResult.timings.

Usage
-----
>>> from src.pipeline import OCRPipeline
>>> pipeline = OCRPipeline()
>>> result = pipeline.run(open("document.jpg", "rb").read())
>>> print(result.to_dict())
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.candidates import CandidateSet, OCRCandidate, build_candidate_sets
from src.context import ContextReasoner
from src.explanation import ExplanationAgent, ExplanationOutput
from src.layout import LayoutAnalyser
from src.ocr import TrOCREngine
from src.preprocessing import Preprocessor
from src.xai import XAIGenerator

# Tunable word-segmentation dilation kernel width (pixels).
# Increase for widely spaced handwriting; decrease for dense print.
_WORD_SEG_KERNEL_WIDTH = int(os.environ.get("WORD_SEG_KERNEL_WIDTH", "15"))


# ─────────────────────────────────────────────────────────────────────────────
# Result Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WordResult:
    text: str
    confidence: float
    alternatives: List[Dict[str, Any]]
    heatmap_base64: str
    explanation: Optional[Dict[str, Any]]


@dataclass
class RegionResult:
    region_type: str
    bbox: List[int]
    words: List[WordResult]


@dataclass
class OCRResult:
    regions: List[RegionResult]
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full result to a JSON-compatible dict."""
        return {
            "regions": [
                {
                    "region_type": region.region_type,
                    "bbox": region.bbox,
                    "words": [
                        {
                            "text": w.text,
                            "confidence": w.confidence,
                            "alternatives": w.alternatives,
                            "heatmap_base64": w.heatmap_base64,
                            "explanation": w.explanation,
                        }
                        for w in region.words
                    ],
                }
                for region in self.regions
            ],
            "timings": self.timings,
        }

    @property
    def full_text(self) -> str:
        """Convenience: concatenate all recognised words into plain text."""
        lines = []
        for region in self.regions:
            words = " ".join(w.text for w in region.words)
            if words.strip():
                lines.append(words)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OCRPipeline
# ─────────────────────────────────────────────────────────────────────────────

class OCRPipeline:
    """
    Full X-OCR pipeline. All models are passed in (loaded once at startup).

    Parameters
    ----------
    preprocessor : Preprocessor, optional
    layout_analyser : LayoutAnalyser, optional
    ocr_engine : TrOCREngine, optional
    context_reasoner : ContextReasoner, optional
    xai_generator : XAIGenerator, optional
    explanation_agent : ExplanationAgent, optional
    """

    def __init__(
        self,
        preprocessor: Optional[Preprocessor] = None,
        layout_analyser: Optional[LayoutAnalyser] = None,
        ocr_engine: Optional[TrOCREngine] = None,
        context_reasoner: Optional[ContextReasoner] = None,
        xai_generator: Optional[XAIGenerator] = None,
        explanation_agent: Optional[ExplanationAgent] = None,
    ) -> None:
        self.preprocessor = preprocessor or Preprocessor()
        self.layout_analyser = layout_analyser or LayoutAnalyser(mock=True)
        self.ocr_engine = ocr_engine or TrOCREngine(mock=True)
        self.context_reasoner = context_reasoner or ContextReasoner(mode="mock")
        self.xai_generator = xai_generator or XAIGenerator(mock=True)
        self.explanation_agent = explanation_agent or ExplanationAgent(mode="mock")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, image_bytes: bytes) -> OCRResult:
        """
        Run the full 7-module pipeline on raw image bytes.

        Parameters
        ----------
        image_bytes : bytes
            Raw image content (JPEG, PNG, TIFF, etc.).

        Returns
        -------
        OCRResult
        """
        timings: Dict[str, float] = {}
        regions_out: List[RegionResult] = []

        # ── Step 1: Decode ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        timings["decode"] = round(time.perf_counter() - t0, 4)

        # ── Step 2: Preprocess (Module 1) ─────────────────────────────────────
        t0 = time.perf_counter()
        try:
            clean_image = self.preprocessor.transform(pil_image)
            clean_pil = clean_image if isinstance(clean_image, Image.Image) else pil_image
        except Exception as exc:
            print(f"[Pipeline] Module 1 (Preprocessor) failed: {exc}")
            clean_pil = pil_image
        timings["preprocessing"] = round(time.perf_counter() - t0, 4)

        # ── Step 3: Layout Analysis (Module 2) ───────────────────────────────
        t0 = time.perf_counter()
        try:
            regions = self.layout_analyser.analyse(clean_pil)
        except Exception as exc:
            print(f"[Pipeline] Module 2 (LayoutAnalyser) failed: {exc}. Using fallback.")
            w, h = clean_pil.size
            regions = [{
                "region_type": "other",
                "bbox": [0, 0, w, h],
                "cropped_image": clean_pil,
            }]
        timings["layout"] = round(time.perf_counter() - t0, 4)

        # ── Step 4: OCR — segment all regions first (Module 3) ───────────────
        # Run segmentation + TrOCR across ALL regions before building context
        # windows, so we can use a GLOBAL word list for context (not per-region).
        t0 = time.perf_counter()
        region_candidate_lists: List[List[List[OCRCandidate]]] = []
        for region in regions:
            crop = region["cropped_image"]
            try:
                raw = self._segment_and_recognise(crop)
            except Exception as exc:
                print(f"[Pipeline] Module 3 (TrOCREngine) failed on region: {exc}")
                raw = [[]]
            region_candidate_lists.append(raw)
        timings["ocr"] = round(time.perf_counter() - t0, 4)

        # Build global flat word list (top-1 per position) for context windows
        flat_all_candidates: List[List[OCRCandidate]] = [
            cands
            for region_cands in region_candidate_lists
            for cands in region_cands
        ]
        flat_top_words: List[str] = [
            min(cands, key=lambda c: c.rank).word if cands else ""
            for cands in flat_all_candidates
        ]

        # ── Steps 5–7: Context + XAI + Explanation (per region) ──────────────
        context_time = 0.0
        xai_time = 0.0
        explanation_time = 0.0

        global_word_offset = 0
        context_window = 3

        for region_idx, region in enumerate(regions):
            region_type = region["region_type"]
            bbox = region["bbox"]
            crop = region["cropped_image"]
            raw_candidates_per_word = region_candidate_lists[region_idx]

            # ── Module 4: Build CandidateSets with GLOBAL context window ──────
            candidate_sets: List[CandidateSet] = []
            for local_i, cands in enumerate(raw_candidates_per_word):
                global_i = global_word_offset + local_i
                left_start = max(0, global_i - context_window)
                right_end = min(len(flat_top_words), global_i + context_window + 1)
                cs = CandidateSet(
                    candidates=cands,
                    context_left=flat_top_words[left_start:global_i],
                    context_right=flat_top_words[global_i + 1:right_end],
                    region_type=region_type,
                    position=global_i,
                )
                candidate_sets.append(cs)
            global_word_offset += len(raw_candidates_per_word)

            # ── Module 5: Context Reasoning ───────────────────────────────────
            t0 = time.perf_counter()
            try:
                for cs in candidate_sets:
                    cs.candidates = self.context_reasoner.score(cs)
            except Exception as exc:
                print(f"[Pipeline] Module 5 (ContextReasoner) failed: {exc}. Visual-only.")
                for cs in candidate_sets:
                    for c in cs.candidates:
                        c.final_score = c.visual_score
            context_time += time.perf_counter() - t0

            # ── Modules 6 & 7: XAI + Explanation per word ────────────────────
            word_results: List[WordResult] = []

            for cs in candidate_sets:
                if not cs.candidates:
                    continue

                best = max(cs.candidates, key=lambda c: c.final_score)

                # Module 6: Heatmap
                t0 = time.perf_counter()
                heatmap_b64 = ""
                char_attn_desc = "Attention information not available."
                try:
                    rollout = self.xai_generator.generate_attention_rollout(crop)
                    overlay = self.xai_generator.generate_overlay(crop, rollout)
                    heatmap_b64 = self.xai_generator.overlay_to_base64(overlay)

                    # Character-level attribution.
                    # Note: uses greedy decoding (num_beams=1) for speed —
                    # the decoding path may differ slightly from the 5-beam
                    # result used for the final word selection in Module 3.
                    char_heatmaps = self.xai_generator.generate_character_heatmaps(crop)
                    char_attn_desc = self.xai_generator.summarise_character_attention(
                        best.word, char_heatmaps
                    )
                except Exception as exc:
                    print(f"[Pipeline] Module 6 (XAIGenerator) failed: {exc}")
                xai_time += time.perf_counter() - t0

                # Module 7: Explanation
                t0 = time.perf_counter()
                explanation_dict: Optional[Dict[str, Any]] = None
                try:
                    exp: ExplanationOutput = self.explanation_agent.explain(
                        word=best.word,
                        visual_score=best.visual_score,
                        candidates=cs.candidates,
                        context_left=cs.context_left,
                        context_right=cs.context_right,
                        character_attention_description=char_attn_desc,
                    )
                    explanation_dict = exp.model_dump()
                except Exception as exc:
                    print(f"[Pipeline] Module 7 (ExplanationAgent) failed: {exc}")
                    explanation_dict = {"error": "Explanation generation failed."}
                explanation_time += time.perf_counter() - t0

                word_results.append(WordResult(
                    text=best.word,
                    confidence=round(best.final_score, 4),
                    alternatives=[
                        {
                            "word": c.word,
                            "visual_score": c.visual_score,
                            "context_score": c.context_score,
                            "final_score": c.final_score,
                            "rank": c.rank,
                        }
                        for c in sorted(cs.candidates, key=lambda c: c.rank)
                    ],
                    heatmap_base64=heatmap_b64,
                    explanation=explanation_dict,
                ))

            regions_out.append(RegionResult(
                region_type=region_type,
                bbox=bbox,
                words=word_results,
            ))

        timings["context_reasoning"] = round(context_time, 4)
        timings["xai"] = round(xai_time, 4)
        timings["explanation"] = round(explanation_time, 4)

        return OCRResult(regions=regions_out, timings=timings)

    # ── Word Segmentation (Module 3 helper) ──────────────────────────────────

    def _segment_and_recognise(
        self, region_crop: Image.Image
    ) -> List[List[OCRCandidate]]:
        """
        Segment a region crop into individual word images using OpenCV
        connected-components, then run TrOCR on each word independently.

        TrOCR is a word-level model (fine-tuned on single-word IAM crops).
        Feeding it a full paragraph produces garbage — this method is the
        critical bridge between Layout (Module 2) and OCR (Module 3).

        Algorithm
        ---------
        1. Grayscale + Otsu binarise (text = white on black)
        2. Horizontal dilation (kernel width = _WORD_SEG_KERNEL_WIDTH)
           — merges characters within a word but not across word gaps
        3. connectedComponentsWithStats → one bounding box per word blob
        4. Filter noise (area < 50 px or w/h < 5 px)
        5. Sort reading order: row (y // 10) then x
        6. Crop each word with small padding → TrOCR → candidates

        Falls back to whole-region inference if no components are found.
        """
        import cv2
        import numpy as np

        # Step 1: Grayscale + Otsu binarise
        arr = np.array(region_crop.convert("L"))
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Step 2: Horizontal dilation to merge characters into word blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_WORD_SEG_KERNEL_WIDTH, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Step 3: Connected components
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated)

        # Step 4: Filter and collect word boxes
        word_boxes: List[Tuple[int, int, int, int]] = []
        for i in range(1, num_labels):  # label 0 = background
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 50 or w < 5 or h < 5:
                continue
            word_boxes.append((x, y, x + w, y + h))

        if not word_boxes:
            candidates = self.ocr_engine.recognise(region_crop)
            return [candidates] if candidates else [[]]

        # Step 5: Reading-order sort — row bucket (y // 10), then x
        word_boxes.sort(key=lambda b: (b[1] // 10, b[0]))

        # Step 6: Crop and recognise each word independently
        all_candidates: List[List[OCRCandidate]] = []
        img_w, img_h = region_crop.size
        pad = 4
        for (x1, y1, x2, y2) in word_boxes:
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(img_w, x2 + pad)
            y2p = min(img_h, y2 + pad)
            word_crop = region_crop.crop((x1p, y1p, x2p, y2p))
            candidates = self.ocr_engine.recognise(word_crop)
            if candidates:
                all_candidates.append(candidates)

        return all_candidates if all_candidates else [[]]
