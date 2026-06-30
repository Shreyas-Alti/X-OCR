"""
Module 5 — Context-Aware Reasoning with LLM
=============================================
Uses an LLM to score the 5 candidates from Module 4 based on sentence-context
fit, then fuses with the visual score from Module 3 to produce a final score.

Supported backends
------------------
* "anthropic"  — Anthropic API (claude-haiku-4-5) — requires ANTHROPIC_API_KEY
* "ollama"     — Qwen 3 1.7B via local Ollama server — free, offline
* "gemini"     — Google Gemini API (gemini-2.5-flash) — requires GEMINI_API_KEY
* "mock"       — Returns uniform context scores (no LLM call, for dev/CI)

Fusion formula (spec):
    final_score = VISUAL_WEIGHT * visual_score + CONTEXT_WEIGHT * context_score
    default: 0.7 * visual + 0.3 * context

Usage
-----
>>> from src.context import ContextReasoner
>>> reasoner = ContextReasoner()
>>> scored = reasoner.score(candidate_set)
>>> best = max(scored, key=lambda c: c.final_score)
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from src.candidates import CandidateSet, OCRCandidate


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_VISUAL_WEIGHT = float(os.environ.get("FUSION_VISUAL_WEIGHT", "0.7"))
DEFAULT_CONTEXT_WEIGHT = float(os.environ.get("FUSION_CONTEXT_WEIGHT", "0.3"))

SYSTEM_PROMPT = (
    "You are an OCR post-correction assistant. "
    "You will be given a sentence with one word masked, and a list of candidate words. "
    "Score each candidate from 0 to 1 based on how well it fits the sentence context. "
    "Return only valid JSON. "
    "Return only the JSON object. "
    "Do not include markdown code blocks, explanations, or any text before or after the JSON object."
)

USER_PROMPT_TEMPLATE = (
    'Sentence with masked word: "{masked_sentence}"\n'
    "Candidates to score: {candidates_list}\n"
    "Return JSON in this exact format:\n"
    '{{"scored_candidates": [{{"word": "<string>", "context_score": <float 0-1>}}]}}'
)

# Fusion weight grid for validation (grid search)
FUSION_GRID = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
]


# ─────────────────────────────────────────────────────────────────────────────
# ContextReasoner
# ─────────────────────────────────────────────────────────────────────────────

class ContextReasoner:
    """
    Scores OCR candidates using LLM context reasoning and fuses with
    TrOCR's visual confidence scores.

    Parameters
    ----------
    mode : str
        One of "anthropic", "ollama", "gemini", "mock".  Defaults to LLM_MODE env var.
    visual_weight : float
        Weight for visual_score in fusion formula (default 0.7).
    context_weight : float
        Weight for context_score in fusion formula (default 0.3).
    anthropic_model : str
        Anthropic model name (default claude-haiku-4-5).
    ollama_model : str
        Ollama model name (default qwen3:1.7b).
    gemini_model : str
        Gemini model name (default gemini-1.5-flash).
    max_retries : int
        Max JSON parse retries per LLM call.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        visual_weight: float = DEFAULT_VISUAL_WEIGHT,
        context_weight: float = DEFAULT_CONTEXT_WEIGHT,
        anthropic_model: str = "claude-haiku-4-5",
        ollama_model: str = "qwen3:1.7b",
        gemini_model: str = "gemini-2.5-flash",
        max_retries: int = 1,
    ) -> None:
        self.mode = mode or os.environ.get("LLM_MODE", "mock")
        self.visual_weight = visual_weight
        self.context_weight = context_weight
        self.anthropic_model = anthropic_model
        self.ollama_model = ollama_model
        self.gemini_model = os.environ.get("GEMINI_MODEL", gemini_model)
        self.max_retries = max_retries

        # Lazily initialised clients
        self._anthropic_client = None
        self._init_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, candidate_set: CandidateSet) -> List[OCRCandidate]:
        """
        Score all candidates in a CandidateSet using LLM context reasoning,
        then apply the fusion formula.

        Parameters
        ----------
        candidate_set : CandidateSet
            Contains candidates + context window from Module 4.

        Returns
        -------
        list[OCRCandidate]
            Same candidates with context_score and final_score filled in.
            Sorted descending by final_score.
        """
        raw_context_scores = self._call_llm(candidate_set)
        fused = self._apply_fusion(candidate_set.candidates, raw_context_scores)
        fused.sort(key=lambda c: c.final_score, reverse=True)
        return fused

    def select_best(self, candidate_set: CandidateSet) -> OCRCandidate:
        """Convenience: score and return the single best candidate."""
        scored = self.score(candidate_set)
        return scored[0]

    # ── LLM Dispatch ─────────────────────────────────────────────────────────

    def _call_llm(self, candidate_set: CandidateSet) -> dict[str, float]:
        """
        Call the configured LLM and parse the context scores.
        Returns {word: context_score} mapping.
        Falls back to uniform scores on failure.
        """
        if self.mode == "mock":
            return self._mock_scores(candidate_set)

        masked = candidate_set.masked_sentence
        words = [c.word for c in candidate_set.candidates]
        user_msg = USER_PROMPT_TEMPLATE.format(
            masked_sentence=masked,
            candidates_list=json.dumps(words),
        )

        for attempt in range(self.max_retries + 1):
            extra = (
                ""
                if attempt == 0
                else (
                    " Your previous response was not valid JSON. "
                    "Return only the JSON object, no other text."
                )
            )
            try:
                raw = self._raw_llm_call(SYSTEM_PROMPT, user_msg + extra)
                return self._parse_response(raw, words)
            except Exception as exc:
                if attempt == self.max_retries:
                    print(f"[ContextReasoner] LLM failed after {attempt+1} attempts: {exc}")
                    return self._mock_scores(candidate_set)

        return self._mock_scores(candidate_set)

    def _raw_llm_call(self, system: str, user: str) -> str:
        """Dispatch to the appropriate LLM backend."""
        if self.mode == "anthropic":
            return self._call_anthropic(system, user)
        elif self.mode == "ollama":
            return self._call_ollama(system, user)
        elif self.mode == "gemini":
            return self._call_gemini(system, user)
        raise ValueError(f"Unknown LLM mode: {self.mode}")

    # ── Anthropic Backend ─────────────────────────────────────────────────────

    def _init_client(self) -> None:
        if self.mode == "anthropic":
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                )
            except ImportError:
                print("[ContextReasoner] anthropic package not installed. Falling back to mock.")
                self.mode = "mock"
        elif self.mode == "gemini":
            try:
                from google import genai  # noqa: F401 — validate install only
            except ImportError:
                print("[ContextReasoner] google-genai not installed. Falling back to mock.")
                self.mode = "mock"

    def _call_anthropic(self, system: str, user: str) -> str:
        # max_tokens=512 gives comfortable headroom for 5 candidates with long words.
        # The JSON response for 5 typical words is ~100-200 tokens; 256 could truncate
        # if the model adds any preamble despite the system prompt instruction.
        message = self._anthropic_client.messages.create(
            model=self.anthropic_model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text

    # ── Gemini Backend ────────────────────────────────────────────────────────

    def _call_gemini(self, system: str, user: str) -> str:
        """
        Call the Google Gemini API using the current google-genai SDK.
        The legacy google-generativeai package is end-of-life as of Nov 30, 2025.
        """
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        response = client.models.generate_content(
            model=self.gemini_model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
            ),
        )
        return response.text

    # ── Ollama Backend ────────────────────────────────────────────────────────

    def _call_ollama(self, system: str, user: str) -> str:
        import urllib.request
        import urllib.error

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = json.dumps({
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise Exception(
                f"Ollama server not reachable at {host}. "
                f"Is it running? Start with: ollama serve  (underlying error: {exc})"
            ) from exc
        return result["message"]["content"]

    # ── Response Parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str, words: List[str]) -> dict[str, float]:
        """
        Parse LLM JSON response into {word: score} dict.
        Raises ValueError on parse failure.
        """
        data = json.loads(raw.strip())
        scored = data.get("scored_candidates", [])
        result: dict[str, float] = {}
        for item in scored:
            w = item.get("word", "")
            s = float(item.get("context_score", 0.5))
            s = max(0.0, min(1.0, s))  # clamp
            result[w] = s
        # Fill missing words with 0.5
        for word in words:
            if word not in result:
                result[word] = 0.5
        return result

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _apply_fusion(
        self,
        candidates: List[OCRCandidate],
        context_scores: dict[str, float],
    ) -> List[OCRCandidate]:
        """Apply fusion formula and update candidate objects in place."""
        for cand in candidates:
            c_score = context_scores.get(cand.word, 0.5)
            cand.context_score = round(c_score, 4)
            cand.final_score = round(
                self.visual_weight * cand.visual_score
                + self.context_weight * cand.context_score,
                4,
            )
        return candidates

    # ── Mock ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _mock_scores(candidate_set: CandidateSet) -> dict[str, float]:
        """Return uniform 0.5 context scores — no LLM call."""
        return {c.word: 0.5 for c in candidate_set.candidates}

    # ── Grid Search Utility ───────────────────────────────────────────────────

    @staticmethod
    def grid_search_weights(
        candidate_sets: List[CandidateSet],
        ground_truths: List[str],
        compute_cer_fn,
    ) -> tuple[float, float, float]:
        """
        Grid search over fusion weight combinations to find the pair that
        minimises CER on the validation set.

        Parameters
        ----------
        candidate_sets : list[CandidateSet]
            One per word — candidates must already have context_score set.
        ground_truths : list[str]
            Ground-truth word string for each position.
        compute_cer_fn : callable
            Function(predictions: list[str], references: list[str]) -> float.

        Returns
        -------
        (best_visual_weight, best_context_weight, best_cer)
        """
        best_cer = float("inf")
        best_v, best_c = 0.7, 0.3

        for v_weight, c_weight in FUSION_GRID:
            predictions = []
            for cs in candidate_sets:
                scored = []
                for cand in cs.candidates:
                    fs = v_weight * cand.visual_score + c_weight * cand.context_score
                    scored.append((fs, cand.word))
                best_word = max(scored, key=lambda x: x[0])[1]
                predictions.append(best_word)

            cer = compute_cer_fn(predictions, ground_truths)
            print(f"  v={v_weight:.1f}, c={c_weight:.1f} → CER={cer:.4f}")

            if cer < best_cer:
                best_cer = cer
                best_v, best_c = v_weight, c_weight

        print(f"[GridSearch] Best: visual={best_v}, context={best_c}, CER={best_cer:.4f}")
        print(
            f"[GridSearch] Update your .env file:\n"
            f"  FUSION_VISUAL_WEIGHT={best_v}\n"
            f"  FUSION_CONTEXT_WEIGHT={best_c}"
        )
        return best_v, best_c, best_cer
