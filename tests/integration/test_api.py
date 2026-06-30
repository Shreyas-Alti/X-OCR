"""
Integration tests for the FastAPI endpoints in api/main.py.
Uses FastAPI's TestClient (synchronous wrapper around httpx).

Environment is forced to mock mode before any app import.
The lifespan context manager loads all modules in mock mode at TestClient startup.
"""

import io
import os

import numpy as np
import pytest
from PIL import Image

# Force mock mode before importing the app
os.environ["MOCK_MODE"] = "true"
os.environ["LLM_MODE"] = "mock"

from fastapi.testclient import TestClient
from api.main import app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_png_bytes(height: int = 200, width: int = 400, value: int = 200) -> io.BytesIO:
    """Create an in-memory PNG image as a BytesIO stream."""
    img = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * value)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Module-scoped TestClient so the lifespan (model loading) runs once.
    Using `with` ensures the lifespan context manager is entered/exited correctly.
    """
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_is_ok(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_model_loaded_is_true(self, client):
        response = client.get("/health")
        assert response.json()["model_loaded"] is True

    def test_content_type_is_json(self, client):
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]


# ─────────────────────────────────────────────────────────────────────────────
# POST /ocr — happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestOCREndpointHappyPath:
    def test_returns_200(self, client):
        buf = _make_png_bytes()
        response = client.post("/ocr", files={"file": ("test.png", buf, "image/png")})
        assert response.status_code == 200

    def test_response_has_regions(self, client):
        buf = _make_png_bytes()
        response = client.post("/ocr", files={"file": ("test.png", buf, "image/png")})
        assert "regions" in response.json()

    def test_response_has_timings(self, client):
        buf = _make_png_bytes()
        response = client.post("/ocr", files={"file": ("test.png", buf, "image/png")})
        assert "timings" in response.json()

    def test_regions_is_a_list(self, client):
        buf = _make_png_bytes()
        response = client.post("/ocr", files={"file": ("test.png", buf, "image/png")})
        assert isinstance(response.json()["regions"], list)

    def test_timings_has_preprocessing_key(self, client):
        buf = _make_png_bytes()
        response = client.post("/ocr", files={"file": ("test.png", buf, "image/png")})
        assert "preprocessing" in response.json()["timings"]

    def test_accepts_jpeg(self, client):
        img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 180)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post("/ocr", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# POST /ocr — word structure
# ─────────────────────────────────────────────────────────────────────────────

class TestOCRWordStructure:
    @pytest.fixture(scope="class")
    def ocr_response(self, client):
        buf = _make_png_bytes()
        return client.post("/ocr", files={"file": ("test.png", buf, "image/png")}).json()

    def test_region_has_region_type(self, ocr_response):
        for region in ocr_response["regions"]:
            assert "region_type" in region

    def test_region_has_bbox(self, ocr_response):
        for region in ocr_response["regions"]:
            assert "bbox" in region
            assert len(region["bbox"]) == 4

    def test_region_has_words(self, ocr_response):
        for region in ocr_response["regions"]:
            assert "words" in region
            assert isinstance(region["words"], list)

    def test_word_has_text(self, ocr_response):
        for region in ocr_response["regions"]:
            for word in region["words"]:
                assert "text" in word
                assert isinstance(word["text"], str)

    def test_word_has_confidence(self, ocr_response):
        for region in ocr_response["regions"]:
            for word in region["words"]:
                assert "confidence" in word
                assert isinstance(word["confidence"], float)

    def test_word_has_alternatives(self, ocr_response):
        for region in ocr_response["regions"]:
            for word in region["words"]:
                assert "alternatives" in word
                assert isinstance(word["alternatives"], list)

    def test_word_has_heatmap_base64(self, ocr_response):
        for region in ocr_response["regions"]:
            for word in region["words"]:
                assert "heatmap_base64" in word
                assert isinstance(word["heatmap_base64"], str)

    def test_word_has_explanation(self, ocr_response):
        for region in ocr_response["regions"]:
            for word in region["words"]:
                assert "explanation" in word


# ─────────────────────────────────────────────────────────────────────────────
# POST /ocr — error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestOCRErrorCases:
    def test_rejects_plain_text_file(self, client):
        response = client.post(
            "/ocr",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code in (400, 415, 422, 500)

    def test_rejects_empty_file(self, client):
        response = client.post(
            "/ocr",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert response.status_code in (400, 422, 500)

    def test_rejects_corrupted_image_bytes(self, client):
        response = client.post(
            "/ocr",
            files={"file": ("bad.png", io.BytesIO(b"\x89PNG\r\n\x1a\nCORRUPTED"), "image/png")},
        )
        assert response.status_code in (400, 500)
