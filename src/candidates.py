"""
Module 4 — Candidate Generation Layer
=======================================
Structures the top-5 candidates from Module 3 into CandidateSet objects
consumed by Module 5 (ContextReasoner).

This is a pure data-formatting layer — no models are loaded here.

Usage
-----
>>> from src.candidates import OCRCandidate, CandidateSet, build_candidate_sets
>>> candidates = [OCRCandidate(word="hello", visual_score=0.92, rank=1), ...]
>>> candidate_set = CandidateSet(
...     candidates=candidates,
...     context_left=["The", "quick"],
...     context_right=["fox", "jumps"],
...     region_type="paragraph",
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OCRCandidate:
    """A single candidate word produced by beam search."""
    word: str
    visual_score: float   # normalised log-probability, range [0, 1]
    rank: int             # 1 = highest visual confidence

    # Added by ContextReasoner (Module 5)
    context_score: float = 0.0
    final_score: float = 0.0


@dataclass
class CandidateSet:
    """
    All candidates for one word position, together with its context window.

    Parameters
    ----------
    candidates : list[OCRCandidate]
        Top-5 candidates from TrOCR beam search (rank 1 first).
    context_left : list[str]
        Up to 3 words immediately to the left of this position.
    context_right : list[str]
        Up to 3 words immediately to the right of this position.
    region_type : str
        Semantic region label from LayoutLMv3 (e.g. "paragraph", "header").
    position : int
        Zero-based index of this word in the full region word list.
    """
    candidates: List[OCRCandidate]
    context_left: List[str]
    context_right: List[str]
    region_type: str
    position: int = 0

    @property
    def top_candidate(self) -> OCRCandidate:
        """Candidate with rank == 1 (highest visual confidence)."""
        return min(self.candidates, key=lambda c: c.rank)

    @property
    def masked_sentence(self) -> str:
        """Reconstruct the masked sentence for the LLM prompt."""
        left = " ".join(self.context_left)
        right = " ".join(self.context_right)
        parts = []
        if left:
            parts.append(left)
        parts.append("[MASK]")
        if right:
            parts.append(right)
        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_sets(
    region_type: str,
    all_candidates: List[List[OCRCandidate]],
    context_window: int = 3,
) -> List[CandidateSet]:
    """
    Build a CandidateSet for every word position in a region.

    Parameters
    ----------
    region_type : str
        Region label from LayoutAnalyser (e.g. "paragraph").
    all_candidates : list[list[OCRCandidate]]
        Outer list is one entry per word in the region;
        inner list is the top-N candidates for that word.
    context_window : int
        How many words to include on each side (default 3 as per spec).

    Returns
    -------
    list[CandidateSet]

    Note
    ----
    context_left and context_right are built from the **visual-only** top-1
    candidate (rank == 1) for each neighbouring position, because context
    scoring has not yet run at this stage.  This is a known architectural
    tradeoff — the LLM sees context constructed from potentially uncorrected
    visual predictions.  In practice the visual top-1 is correct for the
    majority of common words, so the context signal is still useful.
    A multi-pass approach (score, rebuild context, re-score) would eliminate
    this limitation but is out of scope for the current system.
    """
    # Extract the top-1 word from each position for context construction
    top_words: List[str] = [
        min(cands, key=lambda c: c.rank).word if cands else ""
        for cands in all_candidates
    ]

    candidate_sets: List[CandidateSet] = []
    for i, cands in enumerate(all_candidates):
        left_start = max(0, i - context_window)
        right_end = min(len(top_words), i + context_window + 1)

        context_left = top_words[left_start:i]
        context_right = top_words[i + 1:right_end]

        candidate_sets.append(
            CandidateSet(
                candidates=cands,
                context_left=context_left,
                context_right=context_right,
                region_type=region_type,
                position=i,
            )
        )

    return candidate_sets
