"""
Shared pytest fixtures for X-OCR test suite.
All fixtures are available to both unit and integration tests automatically.
"""

import pytest
from PIL import Image
import numpy as np

from src.candidates import OCRCandidate, CandidateSet


# ─────────────────────────────────────────────────────────────────────────────
# Image fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image():
    """100×100 white PIL RGB image — usable by any module that expects a PIL Image."""
    return Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255)


@pytest.fixture
def sample_image_bytes():
    """200×400 grey PNG as raw bytes — for pipeline / API endpoint tests."""
    import io
    img = Image.fromarray(np.ones((200, 400, 3), dtype=np.uint8) * 200)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Candidate fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_candidates():
    """Three OCRCandidates mimicking a real beam-search output."""
    return [
        OCRCandidate(word="workflow", visual_score=0.89, rank=1),
        OCRCandidate(word="workslow", visual_score=0.07, rank=2),
        OCRCandidate(word="workforce", visual_score=0.04, rank=3),
    ]


@pytest.fixture
def sample_candidate_set(sample_candidates):
    """A CandidateSet with realistic left/right context."""
    return CandidateSet(
        candidates=sample_candidates,
        context_left=["use", "in", "my"],
        context_right=["every", "day"],
        region_type="paragraph",
        position=3,
    )
