"""
Module 7 — Explanation Agent
=============================
Generates human-readable, 3-bullet natural language explanations for every
recognized word, describing:
  • Bullet 1 — visual features that supported the recognition
  • Bullet 2 — how sentence context favours the chosen word
  • Bullet 3 — why the top alternative was rejected

Output is Pydantic-validated JSON.  If the LLM returns malformed JSON,
one retry is attempted with an explicit correction instruction.
On double failure, a fallback explanation is returned.

Usage
-----
>>> from src.explanation import ExplanationAgent
>>> agent = ExplanationAgent()
>>> result = agent.explain(
...     word="world",
...     visual_score=0.91,
...     candidates=scored_candidates,
...     context_left=["Hello"],
...     context_right=["how", "are"],
...     character_attention_description="High attention on position 1 ('w').",
... )
>>> print(result.visual_reason)
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Output Schema
# ─────────────────────────────────────────────────────────────────────────────

class RejectedAlternative(BaseModel):
    word: str
    reason: str


class ExplanationOutput(BaseModel):
    """Validated output schema for one word's explanation."""
    word: str
    confidence_percent: int = Field(ge=0, le=100)
    visual_reason: str
    context_reason: str
    rejected: List[RejectedAlternative]

    @field_validator("confidence_percent", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0, min(100, int(v)))


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an OCR explainability assistant.
Given information about a recognized word, its confidence, alternatives, and sentence context,
generate a concise explanation in exactly 3 bullet points.
Bullet 1 (visual_reason): explain what visual features (strokes, ascenders, loops, ink density) \
supported the recognition.
Bullet 2 (context_reason): explain how the sentence context supports the chosen word \
over alternatives. Reference specific context words.
Bullet 3 (rejected): for each of the top 2 alternatives, explain why it was rejected \
(visual dissimilarity, contextual mismatch, spelling, etc.).
Return only valid JSON.
Return only the JSON object.
Do not include markdown code blocks, explanations, or any text before or after the JSON object.\
"""

USER_PROMPT_TEMPLATE = """\
Word recognized: "{word}"
Confidence: {confidence_percent}%
Top alternatives (with final scores):
{alternatives_text}
Sentence context: {context_left} [MASK] {context_right}
Character-level attention note: {char_attn_desc}

Return JSON in this exact structure:
{{
  "word": "{word}",
  "confidence_percent": {confidence_percent},
  "visual_reason": "<one sentence>",
  "context_reason": "<one sentence>",
  "rejected": [
    {{"word": "<alt1>", "reason": "<one sentence>"}},
    {{"word": "<alt2>", "reason": "<one sentence>"}}
  ]
}}\
"""

RETRY_SUFFIX = (
    " Your previous response was not valid JSON. "
    "Return only the JSON object, no other text."
)

FALLBACK_EXPLANATION_TEMPLATE = """\
{{
  "word": "{word}",
  "confidence_percent": {confidence_percent},
  "visual_reason": "The model assigned high visual confidence to this word based on the \
overall stroke pattern and letter shapes observed in the image.",
  "context_reason": "The surrounding context supports this word choice as it fits the \
semantic and syntactic structure of the sentence.",
  "rejected": [
    {{"word": "{alt1}", "reason": "Lower visual confidence and less compatible with context."}},
    {{"word": "{alt2}", "reason": "Alternative letter forms did not match the observed ink strokes."}}
  ]
}}\
"""


# ─────────────────────────────────────────────────────────────────────────────
# ExplanationAgent
# ─────────────────────────────────────────────────────────────────────────────

class ExplanationAgent:
    """
    Generates and validates natural language explanations for OCR decisions.

    Parameters
    ----------
    mode : str
        "anthropic", "ollama", or "mock".
    anthropic_model : str
        Anthropic model to use (default claude-haiku-4-5).
    ollama_model : str
        Ollama model to use (default qwen3:1.7b).
    max_retries : int
        Number of retry attempts on JSON parse failure (spec: 1).
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        anthropic_model: str = "claude-haiku-4-5",
        ollama_model: str = "qwen3:1.7b",
        max_retries: int = 1,
    ) -> None:
        self.mode = mode or os.environ.get("LLM_MODE", "mock")
        self.anthropic_model = anthropic_model
        self.ollama_model = ollama_model
        self.max_retries = max_retries
        self._client: Optional[Any] = None
        self._init_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        word: str,
        visual_score: float,
        candidates: list,   # list[OCRCandidate]
        context_left: List[str],
        context_right: List[str],
        character_attention_description: str = "",
    ) -> ExplanationOutput:
        """
        Generate a Pydantic-validated explanation for a single recognized word.

        Parameters
        ----------
        word : str
            The final chosen word.
        visual_score : float
            Normalised visual confidence (0–1).
        candidates : list[OCRCandidate]
            All scored candidates (used to pull top alternatives).
        context_left : list[str]
            Up to 3 words to the left.
        context_right : list[str]
            Up to 3 words to the right.
        character_attention_description : str
            Text description of character-level XAI from Module 6.

        Returns
        -------
        ExplanationOutput
            Pydantic-validated explanation object.
        """
        confidence_percent = round(visual_score * 100)

        # Build alternatives text (exclude the chosen word, take top 2)
        alternatives = [c for c in candidates if c.word != word][:2]
        alternatives_text = "\n".join(
            f"  - '{c.word}' (score: {c.final_score:.3f})" for c in alternatives
        )
        # Ensure at least 2 alternatives for the template
        while len(alternatives) < 2:
            alternatives.append(type("_", (), {"word": "—", "final_score": 0.0})())

        user_msg = USER_PROMPT_TEMPLATE.format(
            word=word,
            confidence_percent=confidence_percent,
            alternatives_text=alternatives_text or "  - (no alternatives available)",
            context_left=" ".join(context_left) if context_left else "(start of region)",
            context_right=" ".join(context_right) if context_right else "(end of region)",
            char_attn_desc=character_attention_description or "Not available.",
        )

        if self.mode == "mock":
            return self._mock_explanation(word, confidence_percent, alternatives)

        for attempt in range(self.max_retries + 1):
            suffix = RETRY_SUFFIX if attempt > 0 else ""
            try:
                raw = self._raw_llm_call(SYSTEM_PROMPT, user_msg + suffix)
                return self._parse_and_validate(raw)
            except Exception as exc:
                if attempt == self.max_retries:
                    print(f"[ExplanationAgent] LLM failed after {attempt+1} attempts: {exc}")
                    return self._fallback_explanation(word, confidence_percent, alternatives)

        return self._fallback_explanation(word, confidence_percent, alternatives)

    # ── LLM Backends ─────────────────────────────────────────────────────────

    def _init_client(self) -> None:
        if self.mode != "anthropic":
            return
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
        except ImportError:
            print("[ExplanationAgent] anthropic not installed. Using mock mode.")
            self.mode = "mock"

    def _raw_llm_call(self, system: str, user: str) -> str:
        if self.mode == "anthropic":
            return self._call_anthropic(system, user)
        if self.mode == "ollama":
            return self._call_ollama(system, user)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _call_anthropic(self, system: str, user: str) -> str:
        message = self._client.messages.create(
            model=self.anthropic_model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text

    def _call_ollama(self, system: str, user: str) -> str:
        import json as _json
        import urllib.request

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = _json.dumps({
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())
        return result["message"]["content"]

    # ── Parsing & Validation ──────────────────────────────────────────────────

    @staticmethod
    def _parse_and_validate(raw: str) -> ExplanationOutput:
        """Parse LLM JSON and validate with Pydantic. Raises on failure."""
        data = json.loads(raw.strip())
        return ExplanationOutput.model_validate(data)

    # ── Fallbacks ─────────────────────────────────────────────────────────────

    @staticmethod
    def _mock_explanation(
        word: str, confidence_percent: int, alternatives: list
    ) -> ExplanationOutput:
        """Deterministic mock explanation — no LLM call."""
        alt1 = alternatives[0].word if alternatives else "—"
        alt2 = alternatives[1].word if len(alternatives) > 1 else "—"
        return ExplanationOutput(
            word=word,
            confidence_percent=confidence_percent,
            visual_reason=(
                f"The model identified '{word}' with {confidence_percent}% confidence "
                "based on distinctive stroke patterns and character shapes in the image crop."
            ),
            context_reason=(
                f"The word '{word}' fits naturally in the surrounding sentence context "
                "and is semantically coherent with the neighbouring words."
            ),
            rejected=[
                RejectedAlternative(
                    word=alt1,
                    reason=f"'{alt1}' received a lower visual confidence score and was "
                           "contextually less appropriate.",
                ),
                RejectedAlternative(
                    word=alt2,
                    reason=f"'{alt2}' was rejected due to minor character-level differences "
                           "in ink stroke profiles.",
                ),
            ],
        )

    @staticmethod
    def _fallback_explanation(
        word: str, confidence_percent: int, alternatives: list
    ) -> ExplanationOutput:
        """Safe fallback when LLM calls fail after all retries."""
        alt1 = alternatives[0].word if alternatives else "—"
        alt2 = alternatives[1].word if len(alternatives) > 1 else "—"
        return ExplanationOutput(
            word=word,
            confidence_percent=confidence_percent,
            visual_reason="Explanation unavailable — model assigned visual confidence based on image features.",
            context_reason="Explanation unavailable — sentence context was used in selection.",
            rejected=[
                RejectedAlternative(word=alt1, reason="Lower combined score."),
                RejectedAlternative(word=alt2, reason="Lower combined score."),
            ],
        )
