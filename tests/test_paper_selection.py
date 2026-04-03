from io import BytesIO

from fastapi.testclient import TestClient

from app.core.state import state
from app.main import app

client = TestClient(app)


def setup_function() -> None:
    """Clear state before each test."""
    state.clear()


def test_papers_endpoint_shows_selection_status() -> None:
    """Test that /papers endpoint includes selection status."""
    # Create a mock PDF
    pdf_content = b"%PDF-1.4 fake pdf content with abstract intro method results conclusion sections"
    files = {"files": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
    
    # Ingest a paper
    ingest_response = client.post("/ingest", files=files)
    assert ingest_response.status_code == 200
    
    # Get papers list
    response = client.get("/papers")
    assert response.status_code == 200
    papers = response.json()
    
    assert len(papers) == 1
    assert "paper_id" in papers[0]
    assert "filename" in papers[0]
    assert "selected" in papers[0]
    # Newly ingested papers should be auto-selected
    assert papers[0]["selected"] is True


def test_select_papers_endpoint() -> None:
    """Test updating paper selection."""
    # Create mock PDFs
    pdf_content = b"%PDF-1.4 fake pdf content with abstract intro method results conclusion sections"
    files = [
        ("files", ("paper1.pdf", BytesIO(pdf_content), "application/pdf")),
        ("files", ("paper2.pdf", BytesIO(pdf_content), "application/pdf")),
    ]
    
    # Ingest papers
    ingest_response = client.post("/ingest", files=files)
    assert ingest_response.status_code == 200
    
    # Get papers
    papers_response = client.get("/papers")
    papers = papers_response.json()
    paper_ids = [p["paper_id"] for p in papers]
    
    # Both should be selected by default
    assert all(p["selected"] for p in papers)
    
    # Deselect all
    select_response = client.post("/papers/select", json={"paper_ids": []})
    assert select_response.status_code == 200
    assert select_response.json()["selected_count"] == 0
    
    # Verify deselection
    papers_response = client.get("/papers")
    papers = papers_response.json()
    assert all(not p["selected"] for p in papers)
    
    # Select only first paper
    select_response = client.post("/papers/select", json={"paper_ids": [paper_ids[0]]})
    assert select_response.status_code == 200
    assert select_response.json()["selected_count"] == 1
    
    # Verify selection
    papers_response = client.get("/papers")
    papers = papers_response.json()
    selected_papers = [p for p in papers if p["selected"]]
    assert len(selected_papers) == 1
    assert selected_papers[0]["paper_id"] == paper_ids[0]


def test_select_invalid_paper_id() -> None:
    """Test that selecting invalid paper ID returns error."""
    response = client.post("/papers/select", json={"paper_ids": ["invalid-id"]})
    assert response.status_code == 404
    assert "E012" in response.json()["detail"]["code"]


def test_task_uses_selected_papers() -> None:
    """Test that tasks use selected papers when no paper_ids specified."""
    # This test would require mocking the actual task execution
    # For now, we verify the selection state is maintained
    pdf_content = b"%PDF-1.4 fake pdf content with abstract intro method results conclusion sections"
    files = [
        ("files", ("paper1.pdf", BytesIO(pdf_content), "application/pdf")),
        ("files", ("paper2.pdf", BytesIO(pdf_content), "application/pdf")),
    ]
    
    # Ingest papers
    client.post("/ingest", files=files)
    
    # Get papers
    papers_response = client.get("/papers")
    papers = papers_response.json()
    paper_ids = [p["paper_id"] for p in papers]
    
    # Select only first paper
    client.post("/papers/select", json={"paper_ids": [paper_ids[0]]})
    
    # Verify state.selected_papers has been updated
    assert len(state.selected_papers) == 1
    assert paper_ids[0] in state.selected_papers
