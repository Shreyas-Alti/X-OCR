FROM python:3.11-slim

# ── System Dependencies (OpenCV needs these) ──────────────────────────────────
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python Dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code ──────────────────────────────────────────────────────────
COPY src/ ./src/
COPY api/ ./api/
COPY .env.example .env.example

# ── Model Weights: Mounted as volume at /app/models ──────────────────────────
# Do NOT copy model weights into the image.
# Run: docker run -v /your/local/models:/app/models ...
RUN mkdir -p /app/models/trocr_finetuned /app/models/layoutlmv3_finetuned

# ── Non-root User ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 xocr && chown -R xocr:xocr /app
USER xocr

# ── Port ─────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
