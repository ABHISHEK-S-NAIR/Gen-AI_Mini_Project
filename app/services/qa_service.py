import logging

from app.config import settings
from app.core.errors import ERRORS
from app.core.state import state
from app.services.embedding_engine import embed_texts
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

# Minimum cosine similarity score for context chunks to be considered relevant
RELEVANCE_THRESHOLD = 0.25  # Tuned for SciBERT embeddings

_SYSTEM_PROMPT = (
    "You are a research paper assistant. Answer questions strictly based on the "
    "provided excerpts from research papers. If the answer cannot be found in the "
    "excerpts, say so clearly — do not guess or add information not present in the context."
)


def _build_context_block(rows: list[dict]) -> str:
    """
    Build formatted context from retrieved chunks.
    Sorts chunks by paper and preserves document order within each paper.
    """
    # Group by paper_id for better organization
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["paper_id"]].append(row)
    
    # Sort within each group by chunk_index to preserve document order
    for paper_id in grouped:
        grouped[paper_id].sort(key=lambda r: r.get("chunk_index", 0))
    
    # Build formatted context
    parts = []
    excerpt_num = 1
    for paper_id, chunks in grouped.items():
        paper = state.papers.get(str(paper_id))
        paper_name = paper.filename if paper else "unknown"
        
        for row in chunks:
            score = row.get("score", 0.0)
            parts.append(
                f"[Excerpt {excerpt_num} | Paper: '{paper_name}' | "
                f"Section: {row['section']} | Relevance: {score:.2f}]\n{row['text']}"
            )
            excerpt_num += 1
    
    return "\n\n".join(parts)


def _ask_llm(question: str, context_block: str, rows: list[dict]) -> str:
    prompt = (
        f"Using only the excerpts below, answer the following question.\n\n"
        f"Question: {question}\n\n"
        f"Excerpts:\n{context_block}\n\n"
        f"Answer concisely. If the excerpts do not contain enough information, "
        f"say: 'The provided excerpts do not contain sufficient information to answer this question.'"
    )
    try:
        return call_llm(prompt, system=_SYSTEM_PROMPT, max_tokens=512, temperature=0.2)
    except Exception as e:
        logger.warning(f"LLM call failed in qa_service, falling back to raw context: {e}")
        # Graceful degradation: return truncated raw context
        raw = " ".join(str(r["text"]) for r in rows if "text" in r)
        return raw[:600]


def answer_question(question: str, paper_ids: list[str]) -> dict[str, object]:
    return answer_question_with_sections(question, paper_ids, sections=None)


def answer_question_with_sections(
    question: str,
    paper_ids: list[str],
    sections: list[str] | None,
) -> dict[str, object]:
    """
    Answer a question using RAG with relevance filtering.
    
    Retrieves semantically similar chunks, filters by relevance threshold,
    and generates an answer using LLM with grounding verification.
    """
    query_vec = embed_texts([question], settings.embedding_dim)[0]
    
    # Retrieve more candidates than needed to allow filtering
    candidate_rows = state.vdb.search(
        query_vec, settings.top_k_chunks * 2, paper_ids=paper_ids, sections=sections
    )

    if not candidate_rows:
        return {
            "question": question,
            "answer": "No content found in the selected papers.",
            "context": [],
            "grounded": False,
            "confidence": 0.0,
            "error": ERRORS["E005"].__dict__,
        }
    
    # Filter by relevance threshold
    relevant_rows = [r for r in candidate_rows if r.get("score", 0) >= RELEVANCE_THRESHOLD]
    
    if not relevant_rows:
        # No chunks meet relevance threshold
        best_score = max(r.get("score", 0) for r in candidate_rows)
        return {
            "question": question,
            "answer": (
                f"No sufficiently relevant context found in the selected papers. "
                f"Best match score was {best_score:.2f} (threshold: {RELEVANCE_THRESHOLD}). "
                f"Try rephrasing your question or selecting different papers."
            ),
            "context": [],
            "grounded": False,
            "confidence": 0.0,
            "error": ERRORS["E005"].__dict__,
        }
    
    # Take top k from relevant chunks
    rows = relevant_rows[:settings.top_k_chunks]
    
    context_block = _build_context_block(rows)
    answer = _ask_llm(question, context_block, rows)
    
    # Verify answer grounding
    avg_relevance = sum(r.get("score", 0) for r in rows) / len(rows)
    confidence = min(avg_relevance * 1.2, 1.0)  # Scale up slightly, cap at 1.0
    
    # Check if answer indicates insufficient information or LLM failure
    insufficient_indicators = [
        "do not contain sufficient",
        "cannot be found",
        "not enough information",
        "excerpts do not",
        "[LLM_UNAVAILABLE",  # LLM stub response
    ]
    is_grounded = not any(ind in answer for ind in insufficient_indicators)

    return {
        "question": question,
        "answer": answer,
        "context": rows,
        "grounded": is_grounded,
        "confidence": round(confidence, 2),
        "avg_relevance": round(avg_relevance, 2),
    }
