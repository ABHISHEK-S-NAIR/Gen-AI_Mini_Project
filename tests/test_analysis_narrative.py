from app.core.state import state
from app.models.schemas import Chunk
from app.services.analysis_service import analyse


def setup_function() -> None:
    state.clear()


def test_analysis_contains_mult_paragraph_narrative() -> None:
    pid = "p1"
    state.chunks[pid] = [
        Chunk(chunk_id="1", paper_id=pid, section="abstract", chunk_index=0, text="This paper studies sequence modeling."),
        Chunk(chunk_id="2", paper_id=pid, section="intro", chunk_index=1, text="The work compares recurrent and attention mechanisms."),
        Chunk(chunk_id="3", paper_id=pid, section="method", chunk_index=2, text="The method uses encoder-decoder with attention blocks."),
        Chunk(chunk_id="4", paper_id=pid, section="results", chunk_index=3, text="Results show strong translation performance."),
        Chunk(chunk_id="5", paper_id=pid, section="conclusion", chunk_index=4, text="The conclusion highlights scalability improvements."),
    ]

    result = analyse([pid])
    analyses = result["analyses"]
    assert len(analyses) == 1
    text = analyses[0]["analysis_text"]
    assert "\n\n" in text
    assert "This paper focuses" in text
