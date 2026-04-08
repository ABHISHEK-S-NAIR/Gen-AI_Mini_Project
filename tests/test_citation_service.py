from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.citation_service import analyse_citations


def setup_function() -> None:
    state.clear()


def test_citation_detects_author_year_and_numeric() -> None:
    paper_id = "p1"
    text = (
        "This method is consistent with prior work (Vaswani et al., 2017). "
        "Our setup follows common practice [12, 13]."
    )
    state.add_paper(paper_id, IngestedPaper(paper_id=paper_id, filename="a.pdf", raw_text=text))

    result = analyse_citations(paper_id)
    citations = result["citations"]

    assert len(citations) == 2
    assert any(c["raw_text"] == "(Vaswani et al., 2017)" for c in citations)
    assert any(c["raw_text"] == "[12, 13]" for c in citations)
    assert any(c["type"] == "supporting" for c in citations)


def test_citation_classifies_contrasting_sentence() -> None:
    paper_id = "p2"
    text = "However, unlike (Smith, 2020), our model avoids recurrence."
    state.add_paper(paper_id, IngestedPaper(paper_id=paper_id, filename="b.pdf", raw_text=text))

    result = analyse_citations(paper_id)
    citations = result["citations"]

    assert citations
    assert citations[0]["type"] == "contrasting"
    assert "contrast" in citations[0]["insight"].lower()


def test_citation_returns_e007_when_none_found() -> None:
    paper_id = "p3"
    state.add_paper(paper_id, IngestedPaper(paper_id=paper_id, filename="c.pdf", raw_text="No references here"))

    result = analyse_citations(paper_id)
    assert result["citations"] == []
    assert result["error"]["code"] == "E007"


def test_citation_does_not_overtrigger_contrasting_on_however_alone() -> None:
    paper_id = "p4"
    text = "However, we use DPR [26] as the retriever in our setup."
    state.add_paper(paper_id, IngestedPaper(paper_id=paper_id, filename="d.pdf", raw_text=text))

    result = analyse_citations(paper_id)
    citations = result["citations"]

    assert citations
    assert citations[0]["type"] != "contrasting"


def test_citation_context_cleans_metadata_noise() -> None:
    paper_id = "p5"
    text = (
        "BERT (Devlin et al., 2018) improves performance."
        "arXiv:1908.08345v1 [cs.CL] 23 Jul 2019 contact me@uni.edu https://example.org"
    )
    state.add_paper(paper_id, IngestedPaper(paper_id=paper_id, filename="e.pdf", raw_text=text))

    result = analyse_citations(paper_id)
    citations = result["citations"]

    assert citations
    context = citations[0]["context"]
    assert "arXiv:" not in context
    assert "@" not in context
    assert "http" not in context.lower()
