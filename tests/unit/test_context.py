"""
Unit tests for src/context.py — ContextReasoner.
All tests run in mock mode; no LLM API calls are made.
"""

import os
import pytest

# Force mock mode before any import of src.context
os.environ.setdefault("LLM_MODE", "mock")

from src.context import ContextReasoner
from src.candidates import OCRCandidate, CandidateSet


# ─────────────────────────────────────────────────────────────────────────────
# score() — basic behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestScore:
    def test_returns_all_candidates(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock")
        scored = reasoner.score(sample_candidate_set)
        assert len(scored) == len(sample_candidate_set.candidates)

    def test_returns_list_of_ocr_candidates(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock")
        scored = reasoner.score(sample_candidate_set)
        assert all(isinstance(c, OCRCandidate) for c in scored)

    def test_sorted_descending_by_final_score(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock")
        scored = reasoner.score(sample_candidate_set)
        final_scores = [c.final_score for c in scored]
        assert final_scores == sorted(final_scores, reverse=True)

    def test_visual_rank1_wins_on_uniform_context(self, sample_candidate_set):
        """
        Mock returns uniform 0.5 context for all words, so visual score is
        the tiebreaker.  rank-1 (workflow, 0.89 visual) must win.
        """
        reasoner = ContextReasoner(mode="mock")
        scored = reasoner.score(sample_candidate_set)
        assert scored[0].word == "workflow"

    def test_all_final_scores_in_unit_range(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock")
        scored = reasoner.score(sample_candidate_set)
        for c in scored:
            assert 0.0 <= c.final_score <= 1.0

    def test_context_score_set_on_candidates(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock")
        scored = reasoner.score(sample_candidate_set)
        for c in scored:
            assert c.context_score >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Fusion formula
# ─────────────────────────────────────────────────────────────────────────────

class TestFusion:
    def test_default_weights_70_30(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock", visual_weight=0.7, context_weight=0.3)
        scored = reasoner.score(sample_candidate_set)
        for c in scored:
            expected = round(0.7 * c.visual_score + 0.3 * c.context_score, 4)
            assert c.final_score == expected

    def test_custom_weights_applied(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock", visual_weight=0.5, context_weight=0.5)
        scored = reasoner.score(sample_candidate_set)
        for c in scored:
            expected = round(0.5 * c.visual_score + 0.5 * c.context_score, 4)
            assert c.final_score == expected

    def test_visual_only_weights(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock", visual_weight=1.0, context_weight=0.0)
        scored = reasoner.score(sample_candidate_set)
        for c in scored:
            assert c.final_score == round(c.visual_score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# _parse_response — static method
# ─────────────────────────────────────────────────────────────────────────────

class TestParseResponse:
    def test_happy_path(self):
        raw = '{"scored_candidates": [{"word": "hello", "context_score": 0.8}]}'
        result = ContextReasoner._parse_response(raw, ["hello"])
        assert result["hello"] == 0.8

    def test_clamps_score_above_1(self):
        raw = '{"scored_candidates": [{"word": "hello", "context_score": 1.5}]}'
        result = ContextReasoner._parse_response(raw, ["hello"])
        assert result["hello"] == 1.0

    def test_clamps_score_below_0(self):
        raw = '{"scored_candidates": [{"word": "hello", "context_score": -0.3}]}'
        result = ContextReasoner._parse_response(raw, ["hello"])
        assert result["hello"] == 0.0

    def test_fills_missing_words_with_neutral(self):
        raw = '{"scored_candidates": [{"word": "hello", "context_score": 0.8}]}'
        result = ContextReasoner._parse_response(raw, ["hello", "world"])
        assert result["world"] == 0.5

    def test_multiple_candidates(self):
        raw = '{"scored_candidates": [{"word": "a", "context_score": 0.9}, {"word": "b", "context_score": 0.1}]}'
        result = ContextReasoner._parse_response(raw, ["a", "b"])
        assert result["a"] == 0.9
        assert result["b"] == 0.1

    def test_raises_on_invalid_json(self):
        with pytest.raises(Exception):
            ContextReasoner._parse_response("not json", ["hello"])


# ─────────────────────────────────────────────────────────────────────────────
# select_best
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectBest:
    def test_returns_single_candidate(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock")
        best = reasoner.select_best(sample_candidate_set)
        assert isinstance(best, OCRCandidate)

    def test_returned_candidate_has_highest_final_score(self, sample_candidate_set):
        reasoner = ContextReasoner(mode="mock")
        best = reasoner.select_best(sample_candidate_set)
        all_scored = reasoner.score(sample_candidate_set)
        max_score = max(c.final_score for c in all_scored)
        assert best.final_score == max_score


# ─────────────────────────────────────────────────────────────────────────────
# Gemini mode init — import fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiModeInit:
    def test_gemini_falls_back_to_mock_when_package_missing(self, monkeypatch):
        """If google-genai is not installed, mode must silently fall back to mock."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "google" or name.startswith("google.genai"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        reasoner = ContextReasoner(mode="gemini")
        # Should have gracefully downgraded
        assert reasoner.mode == "mock"
