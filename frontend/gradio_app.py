"""
Gradio Demo — X-OCR Explainable OCR System
============================================
3-tab interface:
  Tab 1 — Extracted Text (plain text output)
  Tab 2 — Full JSON Result
  Tab 3 — Heatmap Gallery (one image per word)

Can be run in two modes:
  1. Direct mode (default, no API server needed):
     Set USE_API=false (or leave unset) — calls OCRPipeline directly in-process.
     All models are loaded ONCE at module import time (script startup), not
     per-request, exactly like api/main.py's lifespan pattern. Requires
     models to be available locally (or MOCK_MODE=true for fake output).

  2. API mode:
     Set USE_API=true and API_URL to your FastAPI server.
     Lighter-weight — this Gradio process does not load any ML models itself;
     it just forwards the image to a running FastAPI /ocr endpoint.

Run:
    python frontend/gradio_app.py
    # or
    USE_API=true API_URL=http://localhost:8000 python frontend/gradio_app.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys

import gradio as gr
from PIL import Image

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env BEFORE reading any environment variables below.
# Without this, MOCK_MODE / LLM_MODE / API keys from your .env file are
# silently ignored and every default kicks in instead (this was the bug —
# the pipeline previously defaulted to full mock mode regardless of .env).
from dotenv import load_dotenv
load_dotenv()

USE_API = os.environ.get("USE_API", "false").lower() == "true"
API_URL = os.environ.get("API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading — runs ONCE at script startup, not per-request
# ─────────────────────────────────────────────────────────────────────────────
# Only needed in direct mode. In API mode, the FastAPI server owns all models
# and this Gradio process stays lightweight.

_pipeline = None  # populated below if USE_API is False

if not USE_API:
    from src.pipeline import OCRPipeline
    from src.preprocessing import Preprocessor
    from src.layout import LayoutAnalyser
    from src.ocr import TrOCREngine
    from src.context import ContextReasoner
    from src.xai import XAIGenerator
    from src.explanation import ExplanationAgent

    _mock = os.environ.get("MOCK_MODE", "false").lower() == "true"
    _llm_mode = os.environ.get("LLM_MODE", "mock")

    print(f"[gradio_app] Loading models (MOCK_MODE={_mock}, LLM_MODE={_llm_mode}) …")

    _preprocessor = Preprocessor()
    _layout_analyser = LayoutAnalyser(mock=_mock)
    _ocr_engine = TrOCREngine(mock=_mock)

    # Guard: if non-mock mode but the TrOCR model failed to load, warn clearly
    # instead of silently falling back to mock heatmaps with no explanation.
    if not _mock and getattr(_ocr_engine, "_model", None) is None:
        print(
            "[gradio_app] WARNING: TrOCREngine model is None in non-mock mode. "
            "This usually means the model download failed or TROCR_FINETUNED_PATH "
            "is misconfigured. OCR output will be mock/random. "
            "Check your network connection or set MOCK_MODE=true."
        )

    _xai_generator = XAIGenerator(
        model=_ocr_engine._model if not _mock else None,
        processor=_ocr_engine._processor if not _mock else None,
        mock=_mock,
    )
    _context_reasoner = ContextReasoner(mode=_llm_mode)
    _explanation_agent = ExplanationAgent(mode=_llm_mode)

    _pipeline = OCRPipeline(
        preprocessor=_preprocessor,
        layout_analyser=_layout_analyser,
        ocr_engine=_ocr_engine,
        context_reasoner=_context_reasoner,
        xai_generator=_xai_generator,
        explanation_agent=_explanation_agent,
    )

    print("[gradio_app] Models loaded. Ready to accept requests.")
else:
    print(f"[gradio_app] Running in API mode — forwarding requests to {API_URL}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference Logic
# ─────────────────────────────────────────────────────────────────────────────

def _run_via_api(image_bytes: bytes) -> dict:
    """Send image to FastAPI /ocr endpoint and return JSON result."""
    import urllib.request

    boundary = "----GradioFormBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="upload.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{API_URL}/ocr",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _run_direct(image_bytes: bytes) -> dict:
    """
    Run pipeline in-process using the module-level _pipeline instance.
    Models were already loaded once at script startup — this function only
    runs inference, never re-initialises any model.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline not initialised. This should not happen when USE_API=false."
        )
    result = _pipeline.run(image_bytes)
    return result.to_dict()


def process_image(pil_image: Image.Image):
    """
    Main Gradio processing function.
    Accepts a PIL image, returns (text, json_str, gallery_images).
    """
    if pil_image is None:
        return "No image provided.", "{}", []

    # Convert PIL to PNG bytes
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    try:
        if USE_API:
            result = _run_via_api(image_bytes)
        else:
            result = _run_direct(image_bytes)
    except Exception as exc:
        err = f"Error: {exc}"
        return err, json.dumps({"error": str(exc)}, indent=2), []

    # ── Tab 1: Extracted Text ─────────────────────────────────────────────────
    text_lines = []
    for region in result.get("regions", []):
        region_label = f"[{region.get('region_type', 'region').upper()}]"
        words = " ".join(w.get("text", "") for w in region.get("words", []))
        if words.strip():
            text_lines.append(f"{region_label}\n{words}")
    full_text = "\n\n".join(text_lines) or "(No text detected)"

    # ── Tab 2: Full JSON ──────────────────────────────────────────────────────
    # Truncate base64 strings for readability in UI
    result_display = _truncate_base64(result)
    json_str = json.dumps(result_display, indent=2)

    # ── Tab 3: Heatmap Gallery ────────────────────────────────────────────────
    gallery_images = []
    for region in result.get("regions", []):
        for word_data in region.get("words", []):
            b64 = word_data.get("heatmap_base64", "")
            label = word_data.get("text", "?")
            conf = word_data.get("confidence", 0)
            if b64:
                try:
                    img_bytes = base64.b64decode(b64)
                    heatmap_pil = Image.open(io.BytesIO(img_bytes))
                    gallery_images.append(
                        (heatmap_pil, f'"{label}" ({conf:.1%})')
                    )
                except Exception:
                    pass

    return full_text, json_str, gallery_images


def _truncate_base64(obj, max_len: int = 100):
    """Recursively truncate base64 strings for display."""
    if isinstance(obj, dict):
        return {k: (_truncate_base64(v, max_len) if k != "heatmap_base64" else
                    (v[:max_len] + "…[truncated]" if isinstance(v, str) and len(v) > max_len else v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_base64(i, max_len) for i in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="X-OCR: Explainable OCR",
) as demo:
    gr.HTML("""
        <div class="header-text">
            <h1>🔍 X-OCR: Explainable OCR System</h1>
            <p>Upload a handwritten or printed document image to extract text with
            <strong>visual heatmaps</strong>, <strong>confidence scores</strong>,
            and <strong>natural language explanations</strong>.</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📄 Upload Document Image",
                sources=["upload", "clipboard"],
                height=400,
            )
            run_btn = gr.Button("🚀 Run X-OCR", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📝 Extracted Text"):
                    text_output = gr.Textbox(
                        label="Recognised Text",
                        lines=15,
                        max_lines=30,
                    )

                with gr.TabItem("📋 Full JSON Result"):
                    json_output = gr.Code(
                        label="OCRResult JSON",
                        language="json",
                        lines=20,
                    )

                with gr.TabItem("🌡️ Heatmap Gallery"):
                    gallery_output = gr.Gallery(
                        label="Word Heatmaps (attention rollout overlays)",
                        columns=3,
                        height=400,
                        object_fit="contain",
                        show_label=True,
                    )

    run_btn.click(
        fn=process_image,
        inputs=[image_input],
        outputs=[text_output, json_output, gallery_output],
        show_progress=True,
        api_name="ocr",
    )

    gr.Examples(
        examples=[],  # Add paths to sample images here after downloading IAM data
        inputs=[image_input],
        label="Example Images",
    )

    gr.HTML("""
        <div style="text-align:center; margin-top:1rem; color:#666; font-size:0.85rem;">
            X-OCR — TrOCR + LayoutLMv3 + GradCAM + Claude/Gemini/Qwen |
            <a href="http://localhost:8000/docs" target="_blank">API Docs</a>
        </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="violet"),
        css="""
            .header-text { text-align: center; margin-bottom: 1rem; }
            .badge { background: #7c3aed; color: white; padding: 2px 8px;
                     border-radius: 4px; font-size: 0.75rem; }
        """,
    )