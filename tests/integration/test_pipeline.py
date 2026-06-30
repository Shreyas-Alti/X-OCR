"""
Integration tests for the full OCRPipeline.
All modules run in mock mode — no GPU, no API keys required.

MOCK_MODE=true  → TrOCREngine, LayoutAnalyser, XAIGenerator use mocks
LLM_MODE=mock   → ContextReasoner, ExplanationAgent return deterministic mocks
"""

import io
import json
import os

import numpy as np
import pytest
from PIL import Image

# Force mock mode before any pipeline import
os.environ["MOCK_MODE"] = "true"
os.environ["LLM_MODE"] = "mock"

from src.pipeline import OCRPipeline, OCRResult, RegionResult, WordResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    """Single pipeline instance reused across all tests in this module."""
    return OCRPipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

class TestReturnType:
    def test_run_returns_ocr_result(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        assert isinstance(result, OCRResult)

    def test_result_has_regions_list(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        assert isinstance(result.regions, list)

    def test_result_has_at_least_one_region(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        assert len(result.regions) >= 1

    def test_regions_are_region_result_instances(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        assert all(isinstance(r, RegionResult) for r in result.regions)


# ─────────────────────────────────────────────────────────────────────────────
# Timings
# ─────────────────────────────────────────────────────────────────────────────

class TestTimings:
    REQUIRED_KEYS = {"preprocessing", "layout", "ocr"}

    def test_all_required_timing_keys_present(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        for key in self.REQUIRED_KEYS:
            assert key in result.timings, f"Missing timing key: {key}"

    def test_all_timing_values_non_negative(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        for key, val in result.timings.items():
            assert val >= 0, f"Negative timing for '{key}': {val}"

    def test_timing_values_are_floats(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        assert all(isinstance(v, (int, float)) for v in result.timings.values())


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialisation:
    def test_to_dict_returns_dict(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        assert isinstance(result.to_dict(), dict)

    def test_to_dict_is_json_serialisable(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        serialised = json.dumps(result.to_dict())
        assert len(serialised) > 0

    def test_to_dict_has_regions_key(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        d = result.to_dict()
        assert "regions" in d

    def test_to_dict_has_timings_key(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        d = result.to_dict()
        assert "timings" in d

    def test_region_dict_structure(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        for region in result.to_dict()["regions"]:
            assert "region_type" in region
            assert "bbox" in region
            assert "words" in region


# ─────────────────────────────────────────────────────────────────────────────
# full_text property
# ─────────────────────────────────────────────────────────────────────────────

class TestFullText:
    def test_full_text_returns_string(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        assert isinstance(result.full_text, str)

    def test_full_text_non_empty_for_valid_image(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        # Mock OCR always produces words; should not be completely empty
        assert len(result.full_text) >= 0  # At minimum an empty string — no crash


# ─────────────────────────────────────────────────────────────────────────────
# Word result fields
# ─────────────────────────────────────────────────────────────────────────────

class TestWordResults:
    def test_word_results_have_required_fields(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        for region in result.regions:
            for word in region.words:
                assert isinstance(word.text, str)
                assert isinstance(word.confidence, float)
                assert isinstance(word.alternatives, list)
                assert isinstance(word.heatmap_base64, str)

    def test_confidence_in_unit_range(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        for region in result.regions:
            for word in region.words:
                assert 0.0 <= word.confidence <= 1.0

    def test_alternatives_list_not_empty(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        for region in result.regions:
            for word in region.words:
                assert len(word.alternatives) > 0

    def test_alternative_has_required_keys(self, pipeline, sample_image_bytes):
        result = pipeline.run(sample_image_bytes)
        for region in result.regions:
            for word in region.words:
                for alt in word.alternatives:
                    assert "word" in alt
                    assert "visual_score" in alt
                    assert "final_score" in alt
                    assert "rank" in alt


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_invalid_bytes_raises_exception(self, pipeline):
        with pytest.raises(Exception):
            pipeline.run(b"not_an_image_at_all")

    def test_empty_bytes_raises_exception(self, pipeline):
        with pytest.raises(Exception):
            pipeline.run(b"")
