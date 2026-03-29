import uuid

from app.models.schemas import Chunk, SectionName


def _tokenize(text: str) -> list[str]:
    return text.split()


def _detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def chunk_sections(
    paper_id: str,
    sections: dict[SectionName, str],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for section, text in sections.items():
        if not text.strip():
            continue

        tokens = _tokenize(text)
        if not tokens:
            continue

        step = max(1, chunk_size - overlap)
        for start in range(0, len(tokens), step):
            window = tokens[start : start + chunk_size]
            if not window:
                continue

            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    paper_id=paper_id,
                    section=section,
                    chunk_index=chunk_index,
                    text=_detokenize(window),
                )
            )
            chunk_index += 1

            if start + chunk_size >= len(tokens):
                break

    return chunks
