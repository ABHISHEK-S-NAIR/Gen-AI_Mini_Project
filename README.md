# PaperMind MVP Backend

Initial implementation of the PaperMind system from the SRD/SRC/Test Plan.

## Implemented now
- Multi-PDF ingestion endpoint with `paper_id` assignment
- PDF text extraction (PyPDF2)
- Section detection with basic academic-section synonyms
- Overlapping token chunking with metadata
- Deterministic local embedding engine (hash-based)
- In-memory vector database with metadata filtering
- Document Selection Agent (top-N papers by cosine similarity)
- Task router and initial service endpoints:
  - `analyse`
  - `review`
  - `citations`
  - `ask`
  - `explain`

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:
- `http://127.0.0.1:8000/` for the frontend control deck
- `http://127.0.0.1:8000/docs` for API docs

## API
- `POST /ingest` (multipart form, files[])
- `POST /task`
- `GET /config`
- `PATCH /config` (allowed only before first ingestion)
- `GET /health`
- `GET /papers`

## Notes
- This is an MVP foundation to start implementation quickly.
- LLM-powered generation is currently replaced with deterministic grounded templating to preserve traceability.
- Vector storage is in-memory for local development.
