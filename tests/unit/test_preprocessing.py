"""
Unit tests for src/preprocessing.py — Preprocessor.
Uses real OpenCV / scikit-image but no models.
"""

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import Preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _noisy(base_image: Image.Image, seed: int = 42) -> Image.Image:
    """Add random salt-and-pepper noise to a PIL image."""
    rng = np.random.default_rng(seed)
    arr = np.array(base_image)
    noise = rng.integers(0, 50, arr.shape, dtype=np.uint8)
    noisy = np.clip(arr.astype(np.int16) - noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# ─────────────────────────────────────────────────────────────────────────────
# Return types
# ─────────────────────────────────────────────────────────────────────────────

class TestReturnType:
    def test_pil_input_returns_pil(self, sample_image):
        p = Preprocessor()
        result = p.transform(sample_image)
        assert isinstance(result, Image.Image)

    def test_numpy_rgb_input_returns_pil(self):
        p = Preprocessor()
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = p.transform(arr)
        assert isinstance(result, Image.Image)

    def test_numpy_gray_input_returns_pil(self):
        p = Preprocessor()
        arr = np.ones((100, 100), dtype=np.uint8) * 128
        result = p.transform(arr)
        assert isinstance(result, Image.Image)

    def test_invalid_input_raises_type_error(self):
        p = Preprocessor()
        with pytest.raises(TypeError):
            p.transform("not an image")  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Config toggles
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigToggles:
    def test_all_steps_disabled_still_returns_pil(self, sample_image):
        p = Preprocessor(config={"deskew": False, "denoise": False,
                                  "contrast": False, "binarize": False})
        result = p.transform(sample_image)
        assert isinstance(result, Image.Image)

    def test_only_denoise_enabled(self, sample_image):
        p = Preprocessor(config={"deskew": False, "denoise": True,
                                  "contrast": False, "binarize": False})
        result = p.transform(sample_image)
        assert isinstance(result, Image.Image)

    def test_only_binarize_enabled(self, sample_image):
        p = Preprocessor(config={"deskew": False, "denoise": False,
                                  "contrast": False, "binarize": True})
        result = p.transform(sample_image)
        assert isinstance(result, Image.Image)


# ─────────────────────────────────────────────────────────────────────────────
# Output size
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputSize:
    def test_output_size_within_10_percent_of_input(self, sample_image):
        """
        Deskew may pad or crop by up to ~1px for tiny angles.
        Allow 10% tolerance.
        """
        p = Preprocessor()
        result = p.transform(sample_image)
        assert abs(result.width - sample_image.width) < sample_image.width * 0.1
        assert abs(result.height - sample_image.height) < sample_image.height * 0.1

    def test_larger_image_size_preserved(self):
        big = Image.fromarray(np.ones((400, 800, 3), dtype=np.uint8) * 200)
        p = Preprocessor()
        result = p.transform(big)
        assert abs(result.width - 800) < 80
        assert abs(result.height - 400) < 40


# ─────────────────────────────────────────────────────────────────────────────
# Noise handling
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseHandling:
    def test_noisy_image_processes_without_error(self, sample_image):
        noisy = _noisy(sample_image)
        p = Preprocessor()
        result = p.transform(noisy)
        assert isinstance(result, Image.Image)

    def test_all_black_image_processes_without_error(self):
        black = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        p = Preprocessor()
        result = p.transform(black)
        assert isinstance(result, Image.Image)

    def test_all_white_image_processes_without_error(self, sample_image):
        p = Preprocessor()
        result = p.transform(sample_image)
        assert isinstance(result, Image.Image)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationUtils:
    def test_psnr_identical_images_returns_inf(self):
        arr = np.ones((50, 50), dtype=np.uint8) * 128
        assert Preprocessor.compute_psnr(arr, arr) == float("inf")

    def test_psnr_different_images_returns_finite(self):
        a = np.ones((50, 50), dtype=np.uint8) * 100
        b = np.ones((50, 50), dtype=np.uint8) * 200
        psnr = Preprocessor.compute_psnr(a, b)
        assert 0 < psnr < float("inf")

    def test_detect_noise_level_returns_float(self):
        arr = np.ones((50, 50), dtype=np.uint8) * 128
        score = Preprocessor.detect_noise_level(arr)
        assert isinstance(score, float)
        assert score >= 0.0
