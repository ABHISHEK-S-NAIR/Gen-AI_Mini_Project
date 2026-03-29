from app.core.state import state
from app.models.schemas import Chunk, IngestedPaper
from app.services.analysis_service import analyse


def setup_function() -> None:
    state.clear()


def test_analysis_returns_structured_pipeline_output() -> None:
    pid = "p1"
    state.papers[pid] = IngestedPaper(paper_id=pid, filename="transformer.pdf", raw_text="dummy")
    state.sections[pid] = {
        "abstract": "We propose a Transformer model for sequence transduction.",
        "intro": "RNNs are sequential and slow. We propose a novel architecture based on attention.",
        "method": "The method uses multi-head self-attention and feed-forward layers.",
        "results": "The model reaches 28.4 BLEU on WMT14 En-De.",
        "conclusion": "However, ablation analysis is limited and left for future work.",
        "other": "",
    }
    state.chunks[pid] = [
        Chunk(chunk_id="1", paper_id=pid, section="intro", chunk_index=0, text="RNNs are sequential and slow."),
        Chunk(chunk_id="2", paper_id=pid, section="method", chunk_index=1, text="Uses self-attention."),
    ]

    out = analyse([pid])

    assert "reports" in out
    assert out["reports"]
    report = out["reports"][0]

    assert "paper_review" in report
    assert "explanation" in report
    assert "citation_insight" in report
    assert "structured_data" in report

    structured = report["structured_data"]
    for key in ("problem", "method", "key_technique", "results", "novelty", "limitations"):
        assert key in structured
        assert structured[key]

    assert "comparative_analysis" in out
    assert "insight_summary" in out
    assert "why_it_matters" in out
    assert "evolution" in out
