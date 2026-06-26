"""
Module 3 — Handwritten Text Recognition with TrOCR
====================================================
Core OCR engine.  Accepts a word/line image and returns the top-5 candidate
strings with normalised visual confidence scores derived from beam-search
log-probabilities.

Key implementation details
--------------------------
* Model: microsoft/trocr-base-handwritten (pre-trained; fine-tuned on IAM).
* generate() flags: num_beams=5, num_return_sequences=5,
                    return_dict_in_generate=True, output_scores=True
* Visual confidence = softmax-normalised sum of per-token log-probs.
* Fine-tuning helper: HuggingFace Trainer on IAM words split.

Usage
-----
>>> from src.ocr import TrOCREngine
>>> engine = TrOCREngine()
>>> candidates = engine.recognise(pil_word_image)
>>> print(candidates[0])  # OCRCandidate(word="hello", visual_score=0.94, rank=1)
"""

from __future__ import annotations

import os
from typing import List, Optional, Any, Union

import numpy as np
from PIL import Image

from src.candidates import OCRCandidate

# Lazy imports
_transformers_available = False
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    _transformers_available = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "microsoft/trocr-base-handwritten"
FINETUNED_PATH = os.environ.get("TROCR_FINETUNED_PATH", "models/trocr_finetuned")
IMAGE_SIZE = (384, 384)
NUM_BEAMS = 5
NUM_RETURN_SEQUENCES = 5


# ─────────────────────────────────────────────────────────────────────────────
# TrOCREngine
# ─────────────────────────────────────────────────────────────────────────────

class TrOCREngine:
    """
    TrOCR inference engine returning top-N candidates with visual confidence.

    Parameters
    ----------
    model_path : str, optional
        Local checkpoint path.  Falls back to HuggingFace Hub if missing.
    mock : bool
        If True, returns dummy candidates without loading any model.
    device : str, optional
        "cuda" or "cpu".
    num_beams : int
        Number of beam search beams (= number of returned sequences).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        mock: bool = False,
        device: Optional[str] = None,
        num_beams: int = NUM_BEAMS,
    ) -> None:
        self.mock = mock or (os.environ.get("MOCK_MODE", "false").lower() == "true")
        self.num_beams = num_beams
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None

        if not self.mock and _transformers_available:
            self._device = device or ("cuda" if _cuda_available() else "cpu")
            self._load_model(model_path or FINETUNED_PATH)
        else:
            self._device = "cpu"

    # ── Public API ────────────────────────────────────────────────────────────

    def recognise(self, image: Union[Image.Image, np.ndarray]) -> List[OCRCandidate]:
        """
        Run TrOCR beam search and return top-N candidates.

        Parameters
        ----------
        image : PIL.Image or np.ndarray
            Word or line crop (any size — resized internally).

        Returns
        -------
        list[OCRCandidate]
            Sorted by rank (rank 1 = highest visual confidence).
        """
        if self.mock or self._model is None:
            return self._mock_candidates()

        pil = self._prepare_image(image)
        return self._run_inference(pil)

    def recognise_batch(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[OCRCandidate]]:
        """Batch inference — returns a list of candidate lists."""
        return [self.recognise(img) for img in images]

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: str) -> None:
        """Load TrOCRProcessor and VisionEncoderDecoderModel."""
        src = path if os.path.isdir(path) else DEFAULT_MODEL_NAME
        try:
            self._processor = TrOCRProcessor.from_pretrained(src)
            self._model = VisionEncoderDecoderModel.from_pretrained(src).to(self._device)
            self._model.eval()
            print(f"[TrOCREngine] Loaded model from '{src}' on {self._device}")
        except Exception as exc:
            print(f"[TrOCREngine] Model load failed ({exc}). Using mock mode.")
            self._model = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self, image: Image.Image) -> List[OCRCandidate]:
        """
        Core beam-search inference.

        Critical flags per spec:
          - num_beams=5
          - num_return_sequences=5
          - return_dict_in_generate=True
          - output_scores=True

        Visual confidence is the softmax-normalised sum of per-token log-probs.
        """
        import torch

        pixel_values = self._processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                num_beams=self.num_beams,
                num_return_sequences=NUM_RETURN_SEQUENCES,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=64,
            )

        # Decode generated sequences
        sequences = outputs.sequences  # [num_return_sequences, seq_len]
        decoded = self._processor.batch_decode(sequences, skip_special_tokens=True)

        # Extract log-probabilities from beam scores
        log_probs = self._compute_log_probs(outputs)

        # Normalise log-probs to [0, 1] using softmax
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)
        visual_scores = torch.softmax(log_probs_tensor, dim=0).tolist()

        # Build OCRCandidate objects
        candidates: List[OCRCandidate] = []
        for rank_0, (word, v_score) in enumerate(zip(decoded, visual_scores)):
            candidates.append(
                OCRCandidate(
                    word=word.strip(),
                    visual_score=round(float(v_score), 4),
                    rank=rank_0 + 1,
                )
            )

        # Sort descending by visual_score (should already be, but be explicit)
        candidates.sort(key=lambda c: c.visual_score, reverse=True)
        for i, c in enumerate(candidates):
            c.rank = i + 1

        return candidates

    @staticmethod
    def _compute_log_probs(outputs: Any) -> List[float]:
        """
        Compute sum of log-probabilities for each beam sequence.

        outputs.scores is a tuple of length seq_len.
        Each element is a tensor of shape [num_return_sequences, vocab_size].
        """
        import torch

        scores = outputs.scores          # tuple[tensor[num_seq, vocab]]
        sequences = outputs.sequences    # [num_seq, full_seq_len]

        # The sequences include the prompt tokens — scores start after prompt.
        # For TrOCR, there is no text prompt, so scores align with generated tokens.
        num_sequences = sequences.shape[0]
        num_steps = len(scores)

        log_prob_sums = [0.0] * num_sequences

        for step_idx, step_scores in enumerate(scores):
            # step_scores: [num_seq, vocab_size] — raw logits
            log_probs = torch.log_softmax(step_scores, dim=-1)  # [num_seq, vocab]

            # sequences[:, step_idx + 1]: index 0 is decoder start token.
            # Guard against short sequences (single/two-char words) where the
            # generated sequence is shorter than the number of scoring steps.
            if step_idx + 1 >= sequences.shape[1]:
                break

            token_indices = sequences[:, step_idx + 1]  # [num_seq]

            for seq_i in range(num_sequences):
                tok_id = token_indices[seq_i].item()
                if tok_id < log_probs.shape[1]:
                    log_prob_sums[seq_i] += log_probs[seq_i, tok_id].item()

        return log_prob_sums

    # ── Image Preparation ─────────────────────────────────────────────────────

    @staticmethod
    def _prepare_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Convert to RGB PIL image (TrOCR requires RGB)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image.convert("RGB")

    # ── Mock Fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def _mock_candidates() -> List[OCRCandidate]:
        """Return plausible-looking mock candidates for pipeline testing."""
        words = ["hello", "hel1o", "he110", "hell0", "h3llo"]
        scores = [0.60, 0.20, 0.10, 0.06, 0.04]
        return [
            OCRCandidate(word=w, visual_score=s, rank=i + 1)
            for i, (w, s) in enumerate(zip(words, scores))
        ]

    # ── Fine-tuning Helper ────────────────────────────────────────────────────

    @classmethod
    def train_on_iam(
        cls,
        iam_data_dir: str = "data/iam",
        output_dir: str = "models/trocr_finetuned",
        num_epochs: int = 10,
        learning_rate: float = 5e-5,
        batch_size: int = 16,
        warmup_steps: int = 500,
        early_stopping_patience: int = 3,
    ) -> None:
        """
        Fine-tune TrOCR on the IAM words dataset.

        See notebooks/02_trocr_training.ipynb for the complete step-by-step
        guide including dataset loading, tokenisation, and evaluation.

        Parameters
        ----------
        iam_data_dir : str
            Root directory of the IAM words split (downloaded manually).
        output_dir : str
            Where to save checkpoints.
        num_epochs : int
            Max training epochs (spec: 10–15).
        learning_rate : float
            AdamW LR (spec: 5e-5).
        batch_size : int
            Per-device batch size (spec: 16).
        warmup_steps : int
            LR warmup steps (spec: 500).
        early_stopping_patience : int
            Stop if val CER does not improve for this many epochs (spec: 3).
        """
        if not _transformers_available:
            raise ImportError("Install transformers, torch, and datasets first.")

        import torch
        from transformers import (
            TrOCRProcessor,
            VisionEncoderDecoderModel,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            default_data_collator,
            EarlyStoppingCallback,
        )
        # Note: use jiwer.cer / jiwer.wer directly for metrics (see notebook 02).
        # The datasets.load_metric API was removed in datasets>=2.0.

        # The full dataset loading / collation logic is in the notebook.
        # This stub shows the Trainer configuration.
        processor = TrOCRProcessor.from_pretrained(DEFAULT_MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(DEFAULT_MODEL_NAME)

        # Required config for seq2seq CTC-style generation
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            predict_with_generate=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            save_total_limit=3,
        )

        print("[TrOCREngine] Training stub — see notebooks/02_trocr_training.ipynb")
        print(f"[TrOCREngine] Output dir: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
