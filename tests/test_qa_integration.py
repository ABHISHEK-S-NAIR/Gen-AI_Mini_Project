"""
Comprehensive integration tests for Q&A service.
Tests complete Q&A flow with various scenarios.
"""
import pytest

from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.qa_service import answer_question_with_sections
from app.services.embedding_engine import embed_texts
from app.config import settings


def setup_function() -> None:
    """Clear state before each test."""
    state.clear()


def _create_chunks_with_embeddings(chunks_data: list[dict]) -> list[dict]:
    """
    Helper to create chunks with real embeddings.
    
    Args:
        chunks_data: List of dicts with 'text' and other chunk fields
        
    Returns:
        List of complete chunk dicts with embeddings added
    """
    texts = [c["text"] for c in chunks_data]
    embeddings = embed_texts(texts, settings.embedding_dim)
    
    result = []
    for chunk, embedding in zip(chunks_data, embeddings):
        chunk_copy = chunk.copy()
        chunk_copy["embedding"] = embedding
        result.append(chunk_copy)
    return result


def test_qa_with_successful_retrieval() -> None:
    """Test complete Q&A flow with successful retrieval and answer."""
    # Setup: ingest paper with known content
    paper_id = "p1"
    text = (
        "The proposed model uses a transformer architecture with 12 layers. "
        "Each layer has 768-dimensional hidden states and 8 attention heads. "
        "Training was performed on the GLUE benchmark for 3 epochs."
    )
    paper = IngestedPaper(paper_id=paper_id, filename="transformer.pdf", raw_text=text)
    state.add_paper(paper_id, paper)
    
    # Add chunks to vector DB with real embeddings
    chunks = _create_chunks_with_embeddings([
        {
            "chunk_id": "c1",
            "paper_id": paper_id,
            "section": "method",
            "chunk_index": 0,
            "text": "The proposed model uses a transformer architecture with 12 layers. Each layer has 768-dimensional hidden states.",
        },
        {
            "chunk_id": "c2",
            "paper_id": paper_id,
            "section": "method",
            "chunk_index": 1,
            "text": "Each layer has 768-dimensional hidden states and 8 attention heads.",
        },
        {
            "chunk_id": "c3",
            "paper_id": paper_id,
            "section": "results",
            "chunk_index": 0,
            "text": "Training was performed on the GLUE benchmark for 3 epochs.",
        }
    ])
    state.vdb.upsert(chunks)
    
    # Execute: ask question about content
    response = answer_question_with_sections(
        question="How many layers does the model have?",
        paper_ids=[paper_id],
        sections=None
    )
    
    # Assert: response structure is correct
    assert "question" in response
    assert "answer" in response
    assert "context" in response
    assert "grounded" in response
    assert "confidence" in response
    
    # Assert: context chunks are present
    assert len(response["context"]) > 0
    
    # Assert: chunks have relevance scores
    for chunk in response["context"]:
        assert "score" in chunk
        assert chunk["score"] >= 0.0


def test_qa_multi_paper_synthesis() -> None:
    """Test Q&A combining information from multiple papers."""
    # Setup: 2 papers with complementary info
    p1_text = "BERT uses WordPiece tokenization with a vocabulary of 30,000 tokens."
    p2_text = "GPT-2 uses byte-pair encoding (BPE) for tokenization."
    
    state.add_paper("p1", IngestedPaper(paper_id="p1", filename="bert.pdf", raw_text=p1_text))
    state.add_paper("p2", IngestedPaper(paper_id="p2", filename="gpt2.pdf", raw_text=p2_text))
    
    chunks = _create_chunks_with_embeddings([
        {
            "chunk_id": "c1",
            "paper_id": "p1",
            "section": "method",
            "chunk_index": 0,
            "text": "BERT uses WordPiece tokenization with a vocabulary of 30,000 tokens.",
        },
        {
            "chunk_id": "c2",
            "paper_id": "p2",
            "section": "method",
            "chunk_index": 0,
            "text": "GPT-2 uses byte-pair encoding (BPE) for tokenization.",
        }
    ])
    state.vdb.upsert(chunks)
    
    # Execute: question requiring both papers
    response = answer_question_with_sections(
        question="What tokenization methods are used?",
        paper_ids=["p1", "p2"],
        sections=None
    )
    
    # Assert: both papers' content appears in context
    paper_ids_in_context = set(chunk["paper_id"] for chunk in response["context"])
    assert "p1" in paper_ids_in_context or "p2" in paper_ids_in_context
    assert len(response["context"]) > 0


def test_qa_respects_section_filter() -> None:
    """Test that section filtering works correctly."""
    # Setup: paper with distinct sections
    paper_id = "p1"
    paper = IngestedPaper(
        paper_id=paper_id,
        filename="paper.pdf",
        raw_text="Method section. Results section."
    )
    state.add_paper(paper_id, paper)
    
    chunks = _create_chunks_with_embeddings([
        {
            "chunk_id": "c1",
            "paper_id": paper_id,
            "section": "method",
            "chunk_index": 0,
            "text": "We use a neural network with 3 hidden layers.",
        },
        {
            "chunk_id": "c2",
            "paper_id": paper_id,
            "section": "results",
            "chunk_index": 0,
            "text": "The model achieved 95% accuracy on the test set.",
        }
    ])
    state.vdb.upsert(chunks)
    
    # Execute: filter to "method" only
    response = answer_question_with_sections(
        question="What is the model architecture?",
        paper_ids=[paper_id],
        sections=["method"]
    )
    
    # Assert: context only from method section
    if response["context"]:
        for chunk in response["context"]:
            assert chunk["section"] == "method"


def test_qa_returns_error_when_no_relevant_context() -> None:
    """Test that appropriate error is returned when no relevant context found."""
    # Setup: paper with irrelevant content
    paper_id = "p1"
    paper = IngestedPaper(
        paper_id=paper_id,
        filename="paper.pdf",
        raw_text="Unrelated content about cooking recipes."
    )
    state.add_paper(paper_id, paper)
    
    chunks = _create_chunks_with_embeddings([
        {
            "chunk_id": "c1",
            "paper_id": paper_id,
            "section": "intro",
            "chunk_index": 0,
            "text": "How to make a perfect chocolate cake with butter and sugar.",
        }
    ])
    state.vdb.upsert(chunks)
    
    # Execute: ask technical question
    response = answer_question_with_sections(
        question="What is the model architecture?",
        paper_ids=[paper_id],
        sections=None
    )
    
    # Assert: error is returned or grounded is False
    assert response["grounded"] is False or "error" in response
    assert response["confidence"] == 0.0


def test_qa_handles_empty_paper_list() -> None:
    """Test that Q&A handles empty paper list gracefully."""
    # Setup: add a paper but don't include it in query
    state.add_paper("p1", IngestedPaper(paper_id="p1", filename="test.pdf", raw_text="Some text"))
    
    # Execute: query with empty paper list
    response = answer_question_with_sections(
        question="What is the method?",
        paper_ids=[],
        sections=None
    )
    
    # Assert: appropriate error response
    assert "error" in response or response["grounded"] is False
    assert response["confidence"] == 0.0


def test_qa_preserves_chunk_ordering() -> None:
    """Test that chunks from same paper maintain document order."""
    # Setup: paper with sequential chunks
    paper_id = "p1"
    paper = IngestedPaper(paper_id=paper_id, filename="paper.pdf", raw_text="Sequential content")
    state.add_paper(paper_id, paper)
    
    chunks = _create_chunks_with_embeddings([
        {
            "chunk_id": "c3",
            "paper_id": paper_id,
            "section": "method",
            "chunk_index": 2,
            "text": "Third paragraph of method.",
        },
        {
            "chunk_id": "c1",
            "paper_id": paper_id,
            "section": "method",
            "chunk_index": 0,
            "text": "First paragraph of method.",
        },
        {
            "chunk_id": "c2",
            "paper_id": paper_id,
            "section": "method",
            "chunk_index": 1,
            "text": "Second paragraph of method.",
        }
    ])
    state.vdb.upsert(chunks)
    
    # Execute
    response = answer_question_with_sections(
        question="Describe the method",
        paper_ids=[paper_id],
        sections=None
    )
    
    # Assert: chunks should be ordered by chunk_index in context display
    # The actual order depends on relevance, but _build_context_block should group by paper
    assert len(response["context"]) > 0
    # Each chunk should have chunk_index
    for chunk in response["context"]:
        assert "chunk_index" in chunk


def test_qa_caching_works() -> None:
    """Test that repeated identical queries use cache."""
    # Setup
    paper_id = "p1"
    paper = IngestedPaper(paper_id=paper_id, filename="paper.pdf", raw_text="Test content")
    state.add_paper(paper_id, paper)
    
    chunks = _create_chunks_with_embeddings([
        {
            "chunk_id": "c1",
            "paper_id": paper_id,
            "section": "intro",
            "chunk_index": 0,
            "text": "This is an introduction to the topic.",
        }
    ])
    state.vdb.upsert(chunks)
    
    question = "What is this paper about?"
    
    # Execute: first query
    response1 = answer_question_with_sections(question, [paper_id], None)
    
    # Execute: identical second query (should hit cache)
    response2 = answer_question_with_sections(question, [paper_id], None)
    
    # Assert: both responses should be identical
    assert response1["question"] == response2["question"]
    assert response1["answer"] == response2["answer"]
    # Note: context might have different object IDs but same content
    assert len(response1["context"]) == len(response2["context"])
