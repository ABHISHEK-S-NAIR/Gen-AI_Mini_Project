import uuid

from fastapi import UploadFile

from app.config import settings
from app.core.errors import ERRORS
from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.chunker import chunk_sections
from app.services.embedding_engine import embed_texts, get_embedding_dim
from app.services.figure_extractor import (
    build_figures_text,
    extract_figures_with_vision,
)
from app.services.section_detector import detect_sections
from app.services.text_extractor import build_tables_text, extract_pdf_content


async def ingest_files(files: list[UploadFile]) -> dict[str, object]:
    # Sync config dim to actual model dim on first call
    if settings.embedding_dim != get_embedding_dim():
        settings.embedding_dim = get_embedding_dim()

    ingested: list[IngestedPaper] = []

    for uploaded in files:
        if uploaded.content_type not in {"application/pdf", "application/x-pdf"}:
            return {"error": ERRORS["E001"].__dict__}

        pdf_bytes = await uploaded.read()
        raw_text, tables, figures = extract_pdf_content(pdf_bytes)
        if not raw_text and not tables and not figures:
            return {"error": ERRORS["E001"].__dict__}

        # Vision-based figure extraction if enabled
        if settings.enable_vision_extraction and figures:
            try:
                analyzed_figures = await extract_figures_with_vision(
                    pdf_bytes, figures, max_figures=settings.max_figures_per_paper
                )
                # Replace basic figure metadata with analyzed figures
                figures = analyzed_figures
            except ValueError as e:
                # API key not set - inform user but continue
                error_msg = str(e)
                if "API_KEY" in error_msg:
                    print(f"⚠️  Vision extraction skipped for {uploaded.filename}:")
                    print(f"    {error_msg}")
                    print(
                        f"    Set the API key to enable figure analysis: export {error_msg.split()[0]}='your_key'"
                    )
                else:
                    print(f"Vision extraction failed for {uploaded.filename}: {e}")
            except Exception as e:
                # Other errors - log and continue with basic figure metadata
                print(f"Vision extraction failed for {uploaded.filename}: {e}")

        paper_id = str(uuid.uuid4())
        paper = IngestedPaper(
            paper_id=paper_id,
            filename=uploaded.filename or "unknown.pdf",
            raw_text=raw_text,
            tables=tables,
            figures=figures,
        )
        state.add_paper(paper_id, paper)
        ingested.append(paper)

        sections = detect_sections(raw_text)

        # Build structured text from tables
        tables_text = build_tables_text(tables)

        # Build structured text from figures (if vision extraction was used)
        figures_text = (
            build_figures_text(figures) if settings.enable_vision_extraction else ""
        )

        # Combine tables and figures text
        extracted_content = "\n\n".join(filter(None, [tables_text, figures_text]))

        if extracted_content:
            if sections.get("results", "").strip():
                sections["results"] = (
                    sections["results"].rstrip() + "\n\n" + extracted_content
                )
            elif sections.get("method", "").strip():
                sections["method"] = (
                    sections["method"].rstrip() + "\n\n" + extracted_content
                )
            elif sections.get("intro", "").strip():
                sections["intro"] = (
                    sections["intro"].rstrip() + "\n\n" + extracted_content
                )
            else:
                sections["results"] = extracted_content
        if not any(
            sections[s].strip()
            for s in ("abstract", "intro", "method", "results", "conclusion")
        ):
            return {"error": ERRORS["E002"].__dict__}
        state.add_sections(paper_id, sections)

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
            state.add_chunks(paper_id, chunks)
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
        state.add_embedding(
            paper_id, embed_texts([paper_seed], settings.embedding_dim)[0]
        )

        # Auto-select newly ingested papers
        state.add_selected_paper(paper_id)

    return {"papers": [p.model_dump() for p in ingested]}
