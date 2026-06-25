# X-OCR: Explainable OCR System

> Extract text from handwritten and printed documents with **full explainability** — confidence scores, visual heatmaps, LLM context reasoning, and natural language explanations for every recognized word.

---

## Architecture

```
Input Image
     │
     ▼
┌─────────────────────┐
│ Module 1            │  Preprocessor
│ Document Cleaning   │  deskew → denoise → CLAHE → Sauvola binarization
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Module 2            │  LayoutAnalyser
│ Layout Analysis     │  LayoutLMv3 → region crops + reading order
└────────┬────────────┘
         │  (per region)
         ▼
┌─────────────────────┐
│ Module 3            │  TrOCR (fine-tuned on IAM)
│ Handwriting OCR     │  beam search → top-5 candidates + visual scores
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Module 4            │  CandidateSet builder
│ Candidate Layer     │  sliding context window (±3 words)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Module 5            │  ContextReasoner (Claude / Qwen)
│ Context Fusion      │  LLM scores each candidate → fused score (0.7v+0.3c)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Module 6            │  XAIGenerator
│ Heatmap Generation  │  Attention Rollout + GradCAM + char-level attribution
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Module 7            │  ExplanationAgent (Claude / Qwen)
│ NL Explanation      │  3-bullet explanation per word (Pydantic validated)
└────────┬────────────┘
         │
         ▼
     OCRResult JSON
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <your-repo>
cd xocr

conda create -n xocr python=3.11
conda activate xocr

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY (or set LLM_MODE=mock for dev)
```

### 3. Download IAM dataset

Register at https://www.fki.inf.unibe.ch/databases/iam-handwriting-database  
Download the **words** split and extract to `data/iam/`.

FUNSD is auto-downloaded from HuggingFace when you first run layout training.

### 4. Run development server

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Run Gradio demo

```bash
python frontend/gradio_app.py
# Opens at http://localhost:7860
```

### 6. Run with Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# Frontend: http://localhost:3000
```

---

## Development Phases

| Phase | Module | Status |
|-------|--------|--------|
| 0 | Environment & dataset setup | 🔲 |
| 1 | Module 1: Preprocessing | 🔲 |
| 2 | Module 2: Layout Understanding | 🔲 |
| 3 | Module 3: TrOCR Fine-tuning | 🔲 |
| 4 | Modules 4 & 5: Candidates + Context | 🔲 |
| 5 | Module 6: XAI Heatmaps | 🔲 |
| 6 | Module 7: Explanation Agent | 🔲 |
| 7 | Pipeline + API + Frontend | 🔲 |
| 8 | Ablation Study | 🔲 |
| 9 | Report & Submission | 🔲 |

---

## Results

*(Filled in after evaluation — Phase 8)*

### Ablation Table (CER / WER on 200 test samples)

| Configuration | CER | WER |
|---------------|-----|-----|
| Baseline TrOCR (no fine-tuning) | — | — |
| + Preprocessing | — | — |
| + Layout Analysis | — | — |
| + Context Fusion | — | — |

### XAI Faithfulness (ROAR)
- Top-20% attention masking CER increase: —

### Explanation Quality (Human Eval, 30 samples)
| Criterion | Mean Score |
|-----------|-----------|
| Specificity | — |
| Accuracy | — |
| Usefulness | — |

---

## Repository Structure

```
xocr/
├── src/
│   ├── preprocessing.py    # Module 1 — Preprocessor
│   ├── layout.py           # Module 2 — LayoutAnalyser
│   ├── ocr.py              # Module 3 — TrOCR inference + training
│   ├── candidates.py       # Module 4 — OCRCandidate, CandidateSet
│   ├── context.py          # Module 5 — ContextReasoner
│   ├── xai.py              # Module 6 — XAIGenerator
│   ├── explanation.py      # Module 7 — ExplanationAgent
│   └── pipeline.py         # Orchestrator — OCRPipeline
├── api/
│   └── main.py             # FastAPI backend
├── frontend/
│   ├── gradio_app.py       # Gradio demo
│   └── react/              # React dashboard (Vite)
├── notebooks/              # Jupyter evaluation notebooks
├── data/                   # Datasets (gitignored)
├── models/                 # Model checkpoints (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| torch | 2.1.0 | Deep learning |
| transformers | 4.40.0 | TrOCR, LayoutLMv3 |
| opencv-python | 4.9.0.80 | Image preprocessing |
| scikit-image | 0.22.0 | Sauvola thresholding |
| pytorch-grad-cam | 1.5.0 | GradCAM XAI |
| jiwer | 3.0.3 | CER/WER metrics |
| fastapi | 0.111.0 | REST API |
| gradio | 4.31.0 | Demo UI |
| anthropic | 0.26.0 | LLM integration |
| pydantic | 2.7.0 | Output validation |

---

## Citation

- TrOCR: Li et al., 2022 — https://arxiv.org/abs/2109.10282  
- LayoutLMv3: Huang et al., 2022 — https://arxiv.org/abs/2204.08387  
- Attention Rollout: Abnar & Zuidema, 2020 — https://arxiv.org/abs/2005.00928  
- GradCAM: Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391  
- IAM Database: Marti & Bunke, 2002  
- FUNSD Dataset: Jaume et al., 2019
