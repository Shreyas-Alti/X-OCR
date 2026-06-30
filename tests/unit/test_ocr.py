"""
Unit tests for src/ocr.py — TrOCREngine in mock mode.
No model weights are loaded; all tests use TrOCREngine(mock=True).
"""

import os
import pytest
from PIL import Image
import numpy as np

from src.ocr import TrOCREngine
from src.candidates import OCRCandidate

# Ensure we never accidentally try to download a model
os.environ.setdefault("MOCK_MODE", "true")


@pytest.fixture(scope="module")
def engine():
    """Single mock engine shared across all tests in this module."""
    return TrOCREngine(mock=True)


# ─────────────────────────────────────────────────────────────────────────────
# recognise() output structure
# ─────────────────────────────────────────────────────────────────────────────

class TestMockRecognise:
    def test_returns_list(self, engine, sample_image):
        result = engine.recognise(sample_image)
        assert isinstance(result, list)

    def test_returns_5_candidates(self, engine, sample_image):
        candidates = engine.recognise(sample_image)
        assert len(candidates) == 5

    def test_all_items_are_ocr_candidates(self, engine, sample_image):
        candidates = engine.recognise(sample_image)
        assert all(isinstance(c, OCRCandidate) for c in candidates)

    def test_candidates_sorted_by_rank(self, engine, sample_image):
        candidates = engine.recognise(sample_image)
        ranks = [c.rank for c in candidates]
        assert ranks == sorted(ranks)

    def test_ranks_are_1_through_5(self, engine, sample_image):
        candidates = engine.recognise(sample_image)
        ranks = [c.rank for c in candidates]
        assert ranks == [1, 2, 3, 4, 5]

    def test_visual_scores_sum_to_approx_1(self, engine, sample_image):
        candidates = engine.recognise(sample_image)
        total = sum(c.visual_score for c in candidates)
        assert abs(total - 1.0) < 0.01

    def test_rank1_has_highest_visual_score(self, engine, sample_image):
        candidates = engine.recognise(sample_image)
        rank1 = next(c for c in candidates if c.rank == 1)
        assert all(rank1.visual_score >= c.visual_score for c in candidates)

    def test_all_visual_scores_in_unit_range(self, engine, sample_image):
        candidates = engine.recognise(sample_image)
        assert all(0.0 <= c.visual_score <= 1.0 for c in candidates)

    def test_all_words_are_non_empty_strings(self, engine, sample_image):
        candidates = engine.recognise(sample_image)
        assert all(isinstance(c.word, str) and len(c.word) > 0 for c in candidates)


# ─────────────────────────────────────────────────────────────────────────────
# Input handling
# ─────────────────────────────────────────────────────────────────────────────

class TestInputHandling:
    def test_accepts_pil_image(self, engine):
        img = Image.fromarray(np.ones((50, 50, 3), dtype=np.uint8) * 128)
        result = engine.recognise(img)
        assert len(result) == 5

    def test_accepts_numpy_array(self, engine):
        arr = np.ones((50, 50, 3), dtype=np.uint8) * 128
        result = engine.recognise(arr)
        assert len(result) == 5

    def test_small_image_still_works(self, engine):
        tiny = Image.fromarray(np.ones((10, 10, 3), dtype=np.uint8) * 200)
        result = engine.recognise(tiny)
        assert len(result) == 5


# ─────────────────────────────────────────────────────────────────────────────
# recognise_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestBatch:
    def test_batch_length_matches_input(self, engine):
        images = [
            Image.fromarray(np.ones((50, 50, 3), dtype=np.uint8) * i)
            for i in [100, 150, 200]
        ]
        results = engine.recognise_batch(images)
        assert len(results) == 3

    def test_each_batch_item_has_5_candidates(self, engine):
        images = [Image.fromarray(np.ones((50, 50, 3), dtype=np.uint8) * 200)] * 4
        results = engine.recognise_batch(images)
        assert all(len(r) == 5 for r in results)
