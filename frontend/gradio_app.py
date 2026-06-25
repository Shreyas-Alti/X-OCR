"""
Gradio Demo — X-OCR Explainable OCR System
============================================
3-tab interface:
  Tab 1 — Extracted Text (plain text output)
  Tab 2 — Full JSON Result
  Tab 3 — Heatmap Gallery (one image per word)

Can be run in two modes:
  1. Direct mode (no API server needed):
     Set USE_API=false — calls OCRPipeline directly in-process.
     Requires models to be available locally.

  2. API mode (default):
     Set USE_API=true and API_URL to your FastAPI server.
     Lighter-weight — Gradio process does not load ML models.

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

USE_API = os.environ.get("USE_API", "false").lower() == "true"
API_URL = os.environ.get("API_URL", "http://localhost:8000")


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
    """Run pipeline in-process (loads models on first call)."""
    from src.pipeline import OCRPipeline
    pipeline = OCRPipeline()  # Uses mock mode unless env vars set otherwise
    result = pipeline.run(image_bytes)
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
    theme=gr.themes.Soft(primary_hue="violet"),
    css="""
        .header-text { text-align: center; margin-bottom: 1rem; }
        .badge { background: #7c3aed; color: white; padding: 2px 8px;
                 border-radius: 4px; font-size: 0.75rem; }
    """,
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
                        show_copy_button=True,
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
            X-OCR — TrOCR + LayoutLMv3 + GradCAM + Claude/Qwen |
            <a href="http://localhost:8000/docs" target="_blank">API Docs</a>
        </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=False,
        show_error=True,
    )
