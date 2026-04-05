import uuid

from fastapi import UploadFile

from app.config import settings
from app.core.errors import ERRORS
from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.chunker import chunk_sections
from app.services.embedding_engine import embed_texts, get_embedding_dim
from app.services.section_detector import detect_sections
from app.services.text_extractor import extract_text_from_pdf_bytes


async def ingest_files(files: list[UploadFile]) -> dict[str, object]:
    # Sync config dim to actual model dim on first call
    if settings.embedding_dim != get_embedding_dim():
        settings.embedding_dim = get_embedding_dim()
    
    ingested: list[IngestedPaper] = []

    for uploaded in files:
        if uploaded.content_type not in {"application/pdf", "application/x-pdf"}:
            return {"error": ERRORS["E001"].__dict__}

        pdf_bytes = await uploaded.read()
        raw_text = extract_text_from_pdf_bytes(pdf_bytes)
        if not raw_text:
            return {"error": ERRORS["E001"].__dict__}

        paper_id = str(uuid.uuid4())
        paper = IngestedPaper(paper_id=paper_id, filename=uploaded.filename or "unknown.pdf", raw_text=raw_text)
        state.papers[paper_id] = paper
        ingested.append(paper)

        sections = detect_sections(raw_text)
        if not any(sections[s].strip() for s in ("abstract", "intro", "method", "results", "conclusion")):
            return {"error": ERRORS["E002"].__dict__}
        state.sections[paper_id] = sections

        chunks = chunk_sections(
            paper_id=paper_id,
            sections=sections,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        if chunks:
            vectors = embed_texts([c.text for c in chunks], settings.embedding_dim)
            for c, v in zip(chunks, vectors, strict=True):
                c.embedding = v
            state.chunks[paper_id] = chunks
            state.vdb.upsert(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "paper_id": c.paper_id,
                        "section": c.section,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "embedding": c.embedding,
                    }
                    for c in chunks
                ]
            )

        abstract_text = sections.get("abstract", "")
        paper_seed = abstract_text if abstract_text.strip() else raw_text[:3000]
        state.paper_embeddings[paper_id] = embed_texts([paper_seed], settings.embedding_dim)[0]
        
        # Auto-select newly ingested papers
        state.selected_papers.add(paper_id)

    return {"papers": [p.model_dump() for p in ingested]}
