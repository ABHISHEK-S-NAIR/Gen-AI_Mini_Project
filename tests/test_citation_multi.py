from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.citation_service import analyse_citations_for_papers


def setup_function() -> None:
    state.clear()


def test_multi_paper_citations_grouped_by_paper() -> None:
    state.papers["p1"] = IngestedPaper(
        paper_id="p1",
        filename="paper-one.pdf",
        raw_text="This is consistent with prior work (Alpha et al., 2020).",
    )
    state.papers["p2"] = IngestedPaper(
        paper_id="p2",
        filename="paper-two.pdf",
        raw_text="However, unlike (Beta, 2019), this approach scales better.",
    )

    result = analyse_citations_for_papers(["p1", "p2"])

    assert "papers" in result
    assert len(result["papers"]) == 2
    assert result["papers"][0]["paper_name"] == "paper-one.pdf"
    assert result["papers"][1]["paper_name"] == "paper-two.pdf"
    assert result["total_citations"] >= 2
