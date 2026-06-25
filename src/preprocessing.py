"""
Module 1 — Document Preprocessing
==================================
Cleans raw input images (phone photos, scans) before OCR.

Pipeline (all steps individually toggleable via config):
  1. Deskew          — detect and correct rotation via minAreaRect
  2. Denoise         — fastNlMeansDenoising + morphological opening
  3. Contrast (CLAHE) — Contrast Limited Adaptive Histogram Equalization
  4. Binarize        — Sauvola adaptive thresholding

Usage
-----
>>> from src.preprocessing import Preprocessor
>>> pre = Preprocessor()
>>> clean = pre.transform(pil_image)

Or with a custom config to ablate individual steps:
>>> pre = Preprocessor(config={"deskew": False, "denoise": True,
...                             "contrast": True, "binarize": True})
"""

from __future__ import annotations

import math
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola


# ─────────────────────────────────────────────────────────────────────────────
# Default configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, bool] = {
    "deskew": True,
    "denoise": True,
    "contrast": True,
    "binarize": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor
# ─────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Cleans document images before OCR.

    Parameters
    ----------
    config : dict, optional
        Boolean flags to enable/disable each preprocessing step.
        Keys: "deskew", "denoise", "contrast", "binarize".
        Defaults to all steps enabled.
    clahe_clip_limit : float
        clipLimit for CLAHE (default 2.0 as per spec).
    clahe_tile_grid : tuple[int, int]
        tileGridSize for CLAHE (default (8, 8)).
    sauvola_window : int
        Window size for Sauvola thresholding (default 25).
    """

    def __init__(
        self,
        config: Optional[dict[str, bool]] = None,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
        sauvola_window: int = 25,
    ) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )
        self.sauvola_window = sauvola_window

    # ── Public API ────────────────────────────────────────────────────────────

    def transform(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Image.Image:
        """
        Run the full preprocessing pipeline on a single image.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray
            Input document image (colour or grayscale).

        Returns
        -------
        PIL.Image.Image
            Cleaned, binarized image ready for OCR.
        """
        arr = self._to_gray_array(image)

        if self.config.get("deskew", True):
            arr = self._deskew(arr)

        if self.config.get("denoise", True):
            arr = self._denoise(arr)

        if self.config.get("contrast", True):
            arr = self._enhance_contrast(arr)

        if self.config.get("binarize", True):
            arr = self._binarize(arr)

        return Image.fromarray(arr)

    # ── Step 1: Deskew ────────────────────────────────────────────────────────

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew angle using minAreaRect on the binary image.
        Residual error should be ≤ 0.5°.
        """
        # Invert + threshold so text is white on black
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find all non-zero pixel coordinates
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 10:
            return gray  # not enough content to estimate angle

        # Fit a bounding box to the text mass
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]  # degrees in range [-90, 0)

        # Normalise to [-45°, 45°]
        if angle < -45:
            angle += 90

        # Skip tiny angles — avoids blurring nearly-straight images
        if abs(angle) < 0.5:
            return gray

        (h, w) = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    # ── Step 2: Denoise ───────────────────────────────────────────────────────

    @staticmethod
    def _denoise(gray: np.ndarray) -> np.ndarray:
        """
        Remove noise using fastNlMeansDenoising then morphological opening
        with a 2×2 kernel to eliminate salt-and-pepper artefacts.
        """
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        return opened

    # ── Step 3: Contrast Enhancement (CLAHE) ─────────────────────────────────

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE — Contrast Limited Adaptive Histogram Equalization.
        Handles documents with uneven lighting or faded ink.
        """
        return self._clahe.apply(gray)

    # ── Step 4: Binarization (Sauvola) ───────────────────────────────────────

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Sauvola adaptive thresholding — better than Otsu for handwriting
        with varying ink density.
        """
        thresh = threshold_sauvola(gray, window_size=self.sauvola_window)
        binary = (gray > thresh).astype(np.uint8) * 255
        return binary

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray_array(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Convert PIL Image or numpy array to uint8 grayscale numpy array."""
        if isinstance(image, Image.Image):
            image = image.convert("L")
            return np.array(image, dtype=np.uint8)
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image.astype(np.uint8)
        raise TypeError(f"Expected PIL Image or numpy array, got {type(image)}")

    # ── Evaluation Utility ────────────────────────────────────────────────────

    @staticmethod
    def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
        """
        Compute Peak Signal-to-Noise Ratio between two grayscale images.
        Higher PSNR = less noise introduced by preprocessing.
        """
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * math.log10(255.0 / math.sqrt(mse))

    @staticmethod
    def detect_noise_level(gray: np.ndarray) -> float:
        """
        Estimate noise level using Laplacian variance.
        Lower variance = blurrier / noisier image.
        Returns a score where < 100 suggests significant noise.
        """
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
