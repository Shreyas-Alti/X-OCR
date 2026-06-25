"""
Full Pipeline Orchestrator
===========================
Connects all 7 modules into a single callable pipeline.

  OCRPipeline.run(image_bytes) → OCRResult

Error handling strategy
-----------------------
* Module 2 (layout) failure → treat full image as one region
* Module 5 (context) failure → use visual-only top-1 candidate
* Module 6 (XAI) failure    → omit heatmap (return empty string)
* Module 7 (explanation) failure → return fallback placeholder

Timing
------
Wall-clock time for each module is recorded via time.perf_counter()
and included in OCRResult.timings for profiling.

Usage
-----
>>> from src.pipeline import OCRPipeline
>>> pipeline = OCRPipeline()
>>> result = pipeline.run(open("document.jpg", "rb").read())
>>> print(result.to_dict())
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from src.candidates import OCRCandidate, build_candidate_sets
from src.context import ContextReasoner
from src.explanation import ExplanationAgent, ExplanationOutput
from src.layout import LayoutAnalyser
from src.ocr import TrOCREngine
from src.preprocessing import Preprocessor
from src.xai import XAIGenerator


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
    Full X-OCR pipeline.  All models are passed in (loaded once at startup).

    Parameters
    ----------
    preprocessor : Preprocessor, optional
        If None, a default Preprocessor is created.
    layout_analyser : LayoutAnalyser, optional
        If None, a mock LayoutAnalyser is created.
    ocr_engine : TrOCREngine, optional
        If None, a mock TrOCREngine is created.
    context_reasoner : ContextReasoner, optional
        If None, a mock ContextReasoner is created.
    xai_generator : XAIGenerator, optional
        If None, a mock XAIGenerator is created.
    explanation_agent : ExplanationAgent, optional
        If None, a mock ExplanationAgent is created.
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
            Structured result with regions, words, heatmaps, and explanations.
        """
        timings: Dict[str, float] = {}
        regions_out: List[RegionResult] = []

        # ── Step 1: Decode ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        timings["decode"] = round(time.perf_counter() - t0, 4)

        # ── Step 2: Preprocess ────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            clean_image = self.preprocessor.transform(pil_image)
            clean_pil = clean_image if isinstance(clean_image, Image.Image) else pil_image
        except Exception as exc:
            print(f"[Pipeline] Module 1 (Preprocessor) failed: {exc}")
            clean_pil = pil_image
        timings["preprocessing"] = round(time.perf_counter() - t0, 4)

        # ── Step 3: Layout Analysis ───────────────────────────────────────────
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

        # ── Steps 4–7: Per-region OCR + XAI + Explanation ────────────────────
        ocr_time = 0.0
        context_time = 0.0
        xai_time = 0.0
        explanation_time = 0.0

        for region in regions:
            region_type = region["region_type"]
            bbox = region["bbox"]
            crop = region["cropped_image"]

            # --- Module 3: OCR (word-level) ---
            # For simplicity, treat the whole region crop as one "word" image.
            # In production, add a word-segmentation step between layout and OCR.
            t0 = time.perf_counter()
            try:
                raw_candidates_per_word = self._segment_and_recognise(crop)
            except Exception as exc:
                print(f"[Pipeline] Module 3 (TrOCREngine) failed: {exc}")
                raw_candidates_per_word = [[]]
            ocr_time += time.perf_counter() - t0

            # --- Module 4: Build CandidateSets ---
            candidate_sets = build_candidate_sets(region_type, raw_candidates_per_word)

            # --- Module 5: Context Reasoning ---
            t0 = time.perf_counter()
            try:
                for cs in candidate_sets:
                    cs.candidates = self.context_reasoner.score(cs)
            except Exception as exc:
                print(f"[Pipeline] Module 5 (ContextReasoner) failed: {exc}. Using visual-only.")
                for cs in candidate_sets:
                    for c in cs.candidates:
                        c.final_score = c.visual_score
            context_time += time.perf_counter() - t0

            # --- Modules 6 & 7: XAI + Explanation per word ---
            word_results: List[WordResult] = []
            for cs in candidate_sets:
                if not cs.candidates:
                    continue

                # Pick best candidate
                best = max(cs.candidates, key=lambda c: c.final_score)

                # Module 6: Heatmap
                t0 = time.perf_counter()
                heatmap_b64 = ""
                try:
                    rollout = self.xai_generator.generate_attention_rollout(crop)
                    overlay = self.xai_generator.generate_overlay(crop, rollout)
                    heatmap_b64 = self.xai_generator.overlay_to_base64(overlay)

                    # Character-level attribution
                    char_heatmaps = self.xai_generator.generate_character_heatmaps(crop)
                    char_attn_desc = self.xai_generator.summarise_character_attention(
                        best.word, char_heatmaps
                    )
                except Exception as exc:
                    print(f"[Pipeline] Module 6 (XAIGenerator) failed: {exc}")
                    char_attn_desc = "Attention information not available."
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

        timings["ocr"] = round(ocr_time, 4)
        timings["context_reasoning"] = round(context_time, 4)
        timings["xai"] = round(xai_time, 4)
        timings["explanation"] = round(explanation_time, 4)

        return OCRResult(regions=regions_out, timings=timings)

    # ── Word Segmentation Helper ──────────────────────────────────────────────

    def _segment_and_recognise(self, region_crop: Image.Image) -> List[List[OCRCandidate]]:
        """
        Simple word segmentation: run OCR on the whole region crop.

        In the full pipeline, replace this with a proper word-segmentation
        step (e.g. connected-component analysis on the binarized image) so
        each word is recognised independently.
        """
        candidates = self.ocr_engine.recognise(region_crop)
        if not candidates:
            return [[]]
        # Treat the whole region as one word (single-word mode)
        return [candidates]
