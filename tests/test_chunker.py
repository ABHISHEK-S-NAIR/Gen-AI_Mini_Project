from app.services.chunker import chunk_sections


def test_chunk_overlap_behavior() -> None:
    sections = {
        "abstract": " ".join([f"t{i}" for i in range(0, 1000)]),
        "intro": "",
        "method": "",
        "results": "",
        "conclusion": "",
        "other": "",
    }

    chunks = chunk_sections("p1", sections, chunk_size=512, overlap=64)
    assert len(chunks) == 3

    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()
    assert first_tokens[-64:] == second_tokens[:64]
