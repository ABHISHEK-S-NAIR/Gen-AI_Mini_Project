# PaperMind

**Multi-document research analysis powered by semantic search, LLM reasoning, and a zero-friction local UI.**

PaperMind ingests research PDFs, chunks and embeds them using SciBERT, stores them in a persistent ChromaDB vector store, and exposes a rich set of analysis endpoints — Q&A with grounded citations, peer review, citation extraction, research gap detection, hypothesis generation, and multi-level explanations. Everything runs locally; cloud LLM providers are optional and can be swapped via environment variables.

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Getting Started](#getting-started)
4. [Configuration](#configuration)
5. [Using the Web UI](#using-the-web-ui)
6. [API Reference](#api-reference)
7. [Explanation Levels](#explanation-levels)
8. [Evaluation Harness](#evaluation-harness)
9. [Future Improvements](#future-improvements)

---

## Features

| Category | Capability |
|---|---|
| **Ingestion** | Multi-PDF upload, pdfplumber text + table extraction (PyPDF2 fallback), overlapping token chunking |
| **Embeddings** | SciBERT (`allenai/scibert_scivocab_uncased`, 768-dim) with MiniLM fallback; GPU auto-detection (CUDA / MPS / CPU) |
| **Retrieval** | ChromaDB vector store, cosine similarity, section-aware filtering, MMR diversity re-ranking |
| **Reranking** | Cross-encoder reranking (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) for better relevance |
| **Q&A** | RAG pipeline with grounded citation tags `[Excerpt N]`, multi-turn conversation history, disk-based caching |
| **Analysis** | Structured extraction, LLM-powered summarisation, novelty scoring, citation network insight |
| **Review** | Chain-of-Thought peer review per paper + cross-paper comparison |
| **Citations** | Regex + embedding deduplication, LLM enrichment (claim, relationship, canonical key) |
| **Explanation** | 7 levels: beginner / intermediate / expert / visual / training / pipeline / components |
| **Synthesis** | Research gap finder and hypothesis generator across selected papers |
| **Digest** | 1-page executive digest with key themes, notable results, open questions |
| **arXiv** | Keyword-driven related paper recommendations via the arXiv API |
| **UI** | Browser-based control deck with collapsible result sections, paper selection, real-time status |

---

## Architecture Overview

```
PDF Upload
    │
    ▼
Text Extraction (pdfplumber + PyPDF2 fallback)
    │
    ▼
Section Detection (regex headings + LLM fallback)
    │
    ├─► Overlapping Token Chunker
    │       │
    │       ▼
    │   SciBERT Embeddings ──► ChromaDB (persistent)
    │
    └─► Paper-level Embedding (abstract seed)

Task Request
    │
    ▼
Task Router ──► Q&A Service          (RAG + MMR + conversation history)
             ├─► Analysis Service     (structured extraction + LLM summarisation)
             ├─► Review Service       (CoT peer review)
             ├─► Citation Service     (regex + deduplication + LLM enrichment)
             ├─► Explanation Service  (7 levels + dimension inference)
             ├─► Synthesis Service    (gaps + hypotheses)
             ├─► Digest Service       (executive summary)
             └─► arXiv Service        (related paper recommendations)
```

**Key design choices:**
- State lives in `AppState` (thread-safe with `RLock`); ChromaDB persists vector data locally without a separate service.
- LLM calls are optional. Every service has a deterministic fallback so the system is fully usable without an API key.
- LLM provider priority: **Groq → Gemini → OpenRouter** (auto-selected from env vars).
- Config is locked after first ingestion to prevent dimension mismatches.

---

## Getting Started

### Prerequisites

- Python 3.11+
- ~2 GB disk space (SciBERT model download on first run)
- Optional: a Groq, Gemini, or OpenRouter API key for LLM-powered features

### Install

```bash
git clone <repo-url>
cd papermind
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (all optional — the system runs without them):

```env
# LLM provider — at least one recommended for best results
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AI...
OPENROUTER_API_KEY=sk-or-...

# Optional: override default models
GROQ_MODEL=llama-3.1-8b-instant
GEMINI_MODEL=gemini-2.5-flash
OPENROUTER_MODEL=openai/gpt-4o-mini

# Optional: tuning
LLM_MAX_RETRIES=1
GROQ_TIMEOUT_SEC=30
OPENROUTER_TIMEOUT_SEC=30
```

### Run

```bash
uvicorn app.main:app --reload
```

Then open:
- **`http://127.0.0.1:8000/`** — browser UI
- **`http://127.0.0.1:8000/docs`** — interactive Swagger API docs

---

## Configuration

Retrieve or update chunking and retrieval settings via the `/config` endpoints. **Config is locked once any paper has been ingested** to prevent embedding dimension mismatches.

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 512 | Tokens per chunk |
| `chunk_overlap` | 64 | Overlapping tokens between consecutive chunks |
| `top_k_chunks` | 5 | Chunks returned per retrieval query |
| `top_n_papers` | 3 | Papers selected by the document selection agent |
| `similarity_threshold` | 0.70 | Minimum cosine similarity for chunk relevance |
| `embedding_dim` | 768 | Set automatically from the loaded model (SciBERT=768, MiniLM=384) |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for second-stage reranking |
| `rerank_top_n` | 12 | Number of initial candidates to rerank (0 disables) |
| `hybrid_alpha` | 0.7 | Dense vs BM25 weight (1.0 = dense-only, 0.0 = BM25-only) |

```bash
# Example: change chunk size before ingesting
curl -X PATCH http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"chunk_size": 300, "chunk_overlap": 30}'
```

---

## Using the Web UI

1. **Upload Papers** — drag PDFs into the file picker and click *Ingest PDFs*. Ingested papers appear in the *Ingested Papers* panel and are auto-selected.
2. **Select Papers** — check/uncheck papers to scope analysis. Use *Select All* / *Deselect All* for bulk operations.
3. **Run a Task** — choose a task type from the dropdown:
   - `analyse` — structured analysis with novelty scoring and comparative view
   - `review` — CoT peer review with strengths, weaknesses, suggestions
   - `citations` — extracted citations with type classification and LLM enrichment
   - `ask` — free-form Q&A with grounded citation references
   - `explain` — multi-level explanation (choose level from the second dropdown)
   - `gaps` — research gap detection across selected papers
   - `hypotheses` — testable hypothesis generation
4. **Inspect Results** — the *Output* panel shows a formatted readable view. Use *Collapse All / Expand All* to manage sections. The raw JSON is available under *Show raw JSON*.

---

## API Reference

All endpoints return JSON. Error responses follow the shape `{"code": "EXXX", "message": "..."}`.

### Ingestion

#### `POST /ingest`
Upload one or more PDF files for processing.

**Request:** `multipart/form-data`, field name `files[]`

**Response:**
```json
{
  "papers": [
    {"paper_id": "uuid", "filename": "paper.pdf", "raw_text": "..."}
  ]
}
```

---

### Papers

#### `GET /papers`
List all ingested papers with selection status.

```json
[{"paper_id": "uuid", "filename": "paper.pdf", "selected": true}]
```

#### `POST /papers/select`
Update paper selection state.

```json
// Request
{"paper_ids": ["uuid1", "uuid2"]}

// Response
{"selected_count": 2, "selected_papers": ["uuid1", "uuid2"]}
```

---

### Task (Unified Legacy Endpoint)

#### `POST /task`
Run any task in a single call. Prefer the dedicated endpoints below for new integrations.

```json
// Request
{
  "task": "analyse | review | citations | ask | explain",
  "question": "...",        // required for ask
  "level": "intermediate",  // for explain
  "paper_ids": ["uuid"],    // optional; falls back to UI selection
  "sections": ["method"]    // optional section filter
}
```

---

### Dedicated Endpoints

#### `POST /api/qa`
Answer a question using RAG over selected papers.

```json
// Request
{
  "question": "What optimiser was used?",
  "paper_ids": ["uuid"],          // optional
  "sections": ["method"],         // optional section filter
  "conversation_id": "session-1", // optional, enables multi-turn history
  "debug": false                  // set true for retrieval diagnostics
}

// Response
{
  "question": "...",
  "answer": "The authors used Adam [Excerpt 2] with lr=0.0001.",
  "context": [...],
  "grounded": true,
  "confidence": 0.83,
  "avg_relevance": 0.69,
  "selected_papers": ["uuid"],
  "cited_excerpts": [2],
  "cited_chunks": [...]
}
```

#### `DELETE /api/qa/conversation/{conversation_id}`
Clear multi-turn conversation history for a session.

---

#### `POST /api/citations`
Extract and classify citations from papers.

```json
// Response
{
  "citations": [
    {
      "raw_text": "(Vaswani et al., 2017)",
      "context": "This is consistent with prior work (Vaswani et al., 2017).",
      "type": "supporting",
      "insight": "...",
      "claim": "...",
      "relationship": "foundational",
      "canonical_key": "vaswani2017",
      "duplicate_of": null
    }
  ],
  "selected_papers": ["uuid"]
}
```

---

#### `POST /api/analysis`
Structured analysis with novelty scoring and comparative view.

```json
// Response (abbreviated)
{
  "analysis": {
    "analyses": [{"summary": "...", "methodology": "...", "novelty_score": 0.72, ...}],
    "comparison": "Paper 1 focuses on...",
    "insight_summary": "...",
    "novelty_scores": [{"paper_id": "...", "novelty_score": 0.72, "novelty_rationale": "..."}]
  }
}
```

---

#### `POST /api/review`
Chain-of-Thought peer review with cross-paper comparison.

```json
// Response
{
  "review": {
    "reviews": [
      {
        "paper_name": "paper.pdf",
        "strengths": ["..."],
        "weaknesses": ["..."],
        "suggestions": ["..."],
        "overall_assessment": "..."
      }
    ],
    "comparison": "Both papers target..."
  }
}
```

---

#### `POST /api/explain`
Multi-level explanation (see [Explanation Levels](#explanation-levels)).

```json
// Request
{"paper_ids": ["uuid"], "level": "intermediate"}

// Response
{
  "explanation": {"explanations": [...], "level": "intermediate", "count": 1},
  "selected_papers": ["uuid"],
  "level": "intermediate"
}
```

---

#### `POST /api/gaps`
Identify research gaps and missing experiments across selected papers.

```json
// Response
{
  "gaps": ["The papers do not evaluate on out-of-domain data..."],
  "missing_experiments": ["Ablation isolating positional encoding..."],
  "followup_directions": ["A follow-up combining both methods..."],
  "synthesis": "Full LLM prose...",
  "paper_count": 2
}
```

---

#### `POST /api/hypotheses`
Generate testable research hypotheses from selected papers.

```json
// Response
{
  "hypotheses": [
    {
      "title": "Hypothesis statement",
      "description": "...",
      "rationale": "...",
      "testability": "Experimental approach..."
    }
  ],
  "paper_count": 2
}
```

---

### Config & Health

#### `GET /config` — retrieve current settings
#### `PATCH /config` — update settings (locked after first ingestion)
#### `GET /health` — system status, paper count, chunk count, active config

---

## Explanation Levels

The `/api/explain` endpoint supports 7 levels tailored to different audiences and use cases:

| Level | Audience | Output |
|---|---|---|
| `beginner` | High-school / general public | Plain-language prose, everyday analogies, 3–4 paragraphs |
| `intermediate` | CS undergrad / practitioner | Structured sections: Problem / Method / Key Innovation / Results / Limitations |
| `expert` | ML researcher | Architecture, training strategy, novelty vs baselines, empirical evidence, ablations |
| `visual` | Anyone | ASCII architecture diagram annotated with extracted dimensions (hidden dim, heads, layers) |
| `training` | Practitioner | Side-by-side training vs inference flow diagram with hyperparameters |
| `pipeline` | Engineer | Layer-by-layer tensor shape and parameter count inference (transformer or CNN) |
| `components` | Engineer | Full component breakdown: attention mechanism, FFN, training config, datasets, ablations |

All levels use LLM generation when an API key is configured, with deterministic fallbacks when unavailable.

---

## Evaluation Harness

A quantitative evaluation script is included at `scripts/eval_qa.py`. It measures Q&A performance across three question categories (extractive, abstractive, yes/no) and produces ROUGE-1/2/L, Token F1, and Exact Match scores, plus an ablation table comparing three retrieval modes.

```bash
python scripts/eval_qa.py \
  --dataset data/eval_questions.json \
  --pdf-dir data/papers \
  --output-md reports/eval_results.md \
  --output-json reports/eval_results.json
```

**Dataset format** (`data/eval_questions.json`):
```json
{
  "papers": [{"path": "papers/paper1.pdf"}],
  "questions": [
    {
      "id": "q1",
      "question": "What optimiser was used?",
      "references": ["Adam"],
      "category": "extractive",
      "sections_hint": ["method"]
    }
  ]
}
```

**Options:**
- `--disable-no-retrieval` — skip LLM-only baseline (use when no API key is set)
- `--em-mode strict|balanced|relaxed` — Exact Match scoring mode (`balanced` recommended)
- `--request-delay N` — seconds between API calls to avoid rate limiting
- `--section-fallback-threshold 0.55` — relevance threshold below which section-aware retrieval falls back to flat retrieval

---

## Future Improvements

### Retrieval & Embeddings
- **Qdrant migration** — move from local ChromaDB to Qdrant for production-scale indexing and multi-node deployments.

### LLM Integration
- **Local model support** — add an Ollama provider so the system runs fully offline.

### Analysis Quality
- ✅ **Figure extraction** — vision-based extraction using GPT-4o/Gemini/Groq to extract structured data from charts and figures. [See docs](docs/FIGURE_EXTRACTION.md)
- **Reference resolution** — build a bibliography parser to resolve `[12]` and `(Smith et al., 2020)` to structured entries with DOIs.
- **Claim verification** — cross-check extracted metric claims against results tables for consistency.

### Scalability & Production
- **Persistent state** — replace in-memory `AppState` with a database-backed store (SQLite for single-node, PostgreSQL for multi-node).
- **Async task execution** — move long-running tasks (analysis, review) to a background task queue (Celery / ARQ) with WebSocket progress updates.
- **PDF pre-processing pipeline** — add OCR support (Tesseract / Docling) for scanned papers that yield no text from PyPDF2.

### UI & Developer Experience
- **Annotation layer** — highlight source sentences in the original PDF that correspond to cited excerpts.
- **REST SDK** — publish a typed Python client generated from the OpenAPI spec.
- **Docker image** — ship a `Dockerfile` and `docker-compose.yml` for one-command deployment.
