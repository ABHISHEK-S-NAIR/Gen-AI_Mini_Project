from app.core.state import state
from app.models.schemas import Chunk, IngestedPaper
from app.services.explanation_service import explain


def setup_function() -> None:
    state.clear()


def test_explanation_levels_are_distinct() -> None:
    pid = "p1"
    state.papers[pid] = IngestedPaper(paper_id=pid, filename="paper.pdf", raw_text="dummy")
    state.sections[pid] = {
        "abstract": "We propose a transformer model for machine translation.",
        "intro": "RNNs are sequential and slow.",
        "method": "The method uses self-attention with encoder-decoder blocks.",
        "results": "The model achieves 28.4 BLEU on WMT14.",
        "conclusion": "Future work includes stronger ablations.",
        "other": "",
    }

    beginner = explain(pid, "beginner")
    intermediate = explain(pid, "intermediate")
    expert = explain(pid, "expert")

    assert beginner["explanation"]
    assert intermediate["explanation"]
    assert expert["explanation"]

    assert beginner["explanation"] != intermediate["explanation"]
    assert intermediate["explanation"] != expert["explanation"]
    assert beginner["explanation"] != expert["explanation"]

    assert "metric" in intermediate["explanation"].lower() or "bleu" in intermediate["explanation"].lower()
    assert "architectural" in expert["explanation"].lower() or "architecture" in expert["explanation"].lower() or "training" in expert["explanation"].lower()


def test_visual_explanation_returns_diagram() -> None:
    pid = "p2"
    state.papers[pid] = IngestedPaper(paper_id=pid, filename="paper.pdf", raw_text="dummy")
    state.sections[pid] = {
        "abstract": "We use transformer for tasks.",
        "intro": "Background on transformers.",
        "method": "The method uses self-attention.",
        "results": "Good results.",
        "conclusion": "Future work.",
        "other": "",
    }
    state.chunks[pid] = [
        Chunk(chunk_id="1", paper_id=pid, section="other", chunk_index=0, text="Simple text."),
    ]

    visual = explain(pid, "visual")
    assert visual["level"] == "visual"
    assert visual["diagram"] is not None
    # Updated: New diagrams use Unicode arrows and box drawing characters
    assert "↓" in visual["diagram"] or "┌" in visual["diagram"] or "[" in visual["diagram"]
