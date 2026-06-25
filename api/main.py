"""
FastAPI Backend — X-OCR API
============================
Endpoints:
  POST /ocr     — accepts image file, returns OCRResult JSON
  GET  /health  — liveness probe

Models are loaded once at startup via FastAPI's lifespan context manager.
Never load models inside endpoint functions.

Configuration (environment variables):
  MOCK_MODE=true      — use lightweight mock models (no GPU required)
  LLM_MODE=anthropic  — LLM provider for context + explanation
  ANTHROPIC_API_KEY   — required when LLM_MODE=anthropic
  API_HOST            — bind host (default 0.0.0.0)
  API_PORT            — bind port (default 8000)
  CORS_ORIGINS        — comma-separated allowed origins (default *)

Run locally:
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env if present (no-op when running in Docker with env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.context import ContextReasoner
from src.explanation import ExplanationAgent
from src.layout import LayoutAnalyser
from src.ocr import TrOCREngine
from src.pipeline import OCRPipeline
from src.preprocessing import Preprocessor
from src.xai import XAIGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Global model state (set once at startup)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline: Optional[OCRPipeline] = None
_model_loaded: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — model loading at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup; release on shutdown."""
    global _pipeline, _model_loaded

    mock = os.environ.get("MOCK_MODE", "false").lower() == "true"
    llm_mode = os.environ.get("LLM_MODE", "mock")

    print(f"[Startup] Loading models (MOCK_MODE={mock}, LLM_MODE={llm_mode}) …")

    preprocessor = Preprocessor()
    layout_analyser = LayoutAnalyser(mock=mock)
    ocr_engine = TrOCREngine(mock=mock)
    xai_generator = XAIGenerator(
        model=ocr_engine._model if not mock else None,
        processor=ocr_engine._processor if not mock else None,
        mock=mock,
        device=ocr_engine._device,
    )
    context_reasoner = ContextReasoner(mode=llm_mode)
    explanation_agent = ExplanationAgent(mode=llm_mode)

    _pipeline = OCRPipeline(
        preprocessor=preprocessor,
        layout_analyser=layout_analyser,
        ocr_engine=ocr_engine,
        context_reasoner=context_reasoner,
        xai_generator=xai_generator,
        explanation_agent=explanation_agent,
    )
    _model_loaded = True
    print("[Startup] All models ready. API is live.")

    yield  # ← application runs here

    # Cleanup (optional GPU memory release)
    _pipeline = None
    _model_loaded = False
    print("[Shutdown] Models released.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="X-OCR: Explainable OCR API",
    description=(
        "Extract text from handwritten/printed documents with full explainability — "
        "visual heatmaps, confidence scores, and natural language explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_cors_origins_raw = os.environ.get("CORS_ORIGINS", "*")
_cors_origins = (
    ["*"] if _cors_origins_raw.strip() == "*"
    else [o.strip() for o in _cors_origins_raw.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Liveness probe")
async def health() -> Dict[str, Any]:
    """
    Returns service health status.

    Example response:
        {"status": "ok", "model_loaded": true}
    """
    return {"status": "ok", "model_loaded": _model_loaded}


@app.post("/ocr", summary="Run OCR pipeline on an uploaded image")
async def run_ocr(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, TIFF, BMP, WEBP)"),
) -> JSONResponse:
    """
    Run the full X-OCR pipeline on an uploaded document image.

    Returns a JSON object with:
    - **regions**: list of detected document regions
    - **words**: per-word text, confidence, candidates, heatmap, and explanation
    - **timings**: wall-clock time per module (seconds)

    Accepts multipart/form-data with field name **file**.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please retry.")

    # Validate content type
    allowed_types = {
        "image/jpeg", "image/jpg", "image/png", "image/tiff",
        "image/bmp", "image/webp", "image/gif",
    }
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {content_type}. "
                   f"Allowed: {', '.join(sorted(allowed_types))}",
        )

    # Read image bytes
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run pipeline
    try:
        result = _pipeline.run(image_bytes)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(exc)}",
        )

    return JSONResponse(content=result.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
