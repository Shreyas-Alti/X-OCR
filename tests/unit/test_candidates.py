"""
Unit tests for src/candidates.py — OCRCandidate, CandidateSet, build_candidate_sets.
No models, no LLM, no I/O.
"""

import pytest
from src.candidates import OCRCandidate, CandidateSet, build_candidate_sets


# ─────────────────────────────────────────────────────────────────────────────
# OCRCandidate
# ─────────────────────────────────────────────────────────────────────────────

class TestOCRCandidate:
    def test_required_fields_stored(self):
        c = OCRCandidate(word="hello", visual_score=0.9, rank=1)
        assert c.word == "hello"
        assert c.visual_score == 0.9
        assert c.rank == 1

    def test_context_score_defaults_to_zero(self):
        c = OCRCandidate(word="hello", visual_score=0.9, rank=1)
        assert c.context_score == 0.0

    def test_final_score_defaults_to_zero(self):
        c = OCRCandidate(word="hello", visual_score=0.9, rank=1)
        assert c.final_score == 0.0

    def test_scores_can_be_set_after_creation(self):
        c = OCRCandidate(word="hello", visual_score=0.9, rank=1)
        c.context_score = 0.7
        c.final_score = 0.85
        assert c.context_score == 0.7
        assert c.final_score == 0.85


# ─────────────────────────────────────────────────────────────────────────────
# CandidateSet.masked_sentence
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskedSentence:
    def test_includes_mask_token(self, sample_candidate_set):
        assert "[MASK]" in sample_candidate_set.masked_sentence

    def test_includes_left_context(self, sample_candidate_set):
        # Last left-context word must appear
        assert "my" in sample_candidate_set.masked_sentence

    def test_includes_right_context(self, sample_candidate_set):
        assert "every" in sample_candidate_set.masked_sentence

    def test_no_context_returns_just_mask(self):
        cs = CandidateSet(
            candidates=[OCRCandidate("hello", 0.9, 1)],
            context_left=[],
            context_right=[],
            region_type="other",
        )
        assert cs.masked_sentence == "[MASK]"

    def test_left_only_context(self):
        cs = CandidateSet(
            candidates=[OCRCandidate("hello", 0.9, 1)],
            context_left=["The", "quick"],
            context_right=[],
            region_type="paragraph",
        )
        masked = cs.masked_sentence
        assert masked == "The quick [MASK]"

    def test_right_only_context(self):
        cs = CandidateSet(
            candidates=[OCRCandidate("hello", 0.9, 1)],
            context_left=[],
            context_right=["fox", "jumps"],
            region_type="paragraph",
        )
        masked = cs.masked_sentence
        assert masked == "[MASK] fox jumps"


# ─────────────────────────────────────────────────────────────────────────────
# CandidateSet.top_candidate
# ─────────────────────────────────────────────────────────────────────────────

class TestTopCandidate:
    def test_returns_rank_1(self, sample_candidate_set):
        top = sample_candidate_set.top_candidate
        assert top.rank == 1

    def test_returns_highest_visual_confidence(self, sample_candidate_set):
        top = sample_candidate_set.top_candidate
        assert top.word == "workflow"
        assert top.visual_score == 0.89

    def test_rank1_even_when_not_first_in_list(self):
        # Rank-1 placed last intentionally
        candidates = [
            OCRCandidate(word="b", visual_score=0.10, rank=2),
            OCRCandidate(word="c", visual_score=0.05, rank=3),
            OCRCandidate(word="a", visual_score=0.85, rank=1),
        ]
        cs = CandidateSet(candidates=candidates, context_left=[], context_right=[], region_type="other")
        assert cs.top_candidate.word == "a"


# ─────────────────────────────────────────────────────────────────────────────
# build_candidate_sets
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCandidateSets:
    def _make_word_candidates(self, n: int):
        """Helper: n words, each with a single rank-1 candidate."""
        return [
            [OCRCandidate(word=f"word{i}", visual_score=0.9, rank=1)]
            for i in range(n)
        ]

    def test_empty_input_returns_empty_list(self):
        assert build_candidate_sets("paragraph", []) == []

    def test_length_matches_input(self):
        data = self._make_word_candidates(5)
        sets = build_candidate_sets("paragraph", data)
        assert len(sets) == 5

    def test_single_word_no_context(self):
        data = self._make_word_candidates(1)
        sets = build_candidate_sets("paragraph", data)
        assert sets[0].context_left == []
        assert sets[0].context_right == []

    def test_first_word_has_no_left_context(self):
        data = self._make_word_candidates(5)
        sets = build_candidate_sets("paragraph", data)
        assert sets[0].context_left == []

    def test_last_word_has_no_right_context(self):
        data = self._make_word_candidates(5)
        sets = build_candidate_sets("paragraph", data)
        assert sets[-1].context_right == []

    def test_middle_word_has_full_context(self):
        data = self._make_word_candidates(7)
        sets = build_candidate_sets("paragraph", data, context_window=3)
        # index 3 (middle) should have 3 left and 3 right
        assert len(sets[3].context_left) == 3
        assert len(sets[3].context_right) == 3

    def test_context_window_respected(self):
        data = self._make_word_candidates(10)
        sets = build_candidate_sets("paragraph", data, context_window=2)
        # index 5 has 2 left, 2 right
        assert len(sets[5].context_left) == 2
        assert len(sets[5].context_right) == 2

    def test_region_type_propagated(self):
        data = self._make_word_candidates(3)
        sets = build_candidate_sets("header", data)
        assert all(s.region_type == "header" for s in sets)

    def test_position_index_set_correctly(self):
        data = self._make_word_candidates(4)
        sets = build_candidate_sets("paragraph", data)
        for i, s in enumerate(sets):
            assert s.position == i

    def test_context_built_from_top1_words(self):
        # Candidates with distinct words to verify context extraction
        data = [
            [OCRCandidate(word="alpha", visual_score=0.9, rank=1)],
            [OCRCandidate(word="beta",  visual_score=0.9, rank=1)],
            [OCRCandidate(word="gamma", visual_score=0.9, rank=1)],
        ]
        sets = build_candidate_sets("paragraph", data)
        # Middle word's left context = ["alpha"], right context = ["gamma"]
        assert sets[1].context_left == ["alpha"]
        assert sets[1].context_right == ["gamma"]
