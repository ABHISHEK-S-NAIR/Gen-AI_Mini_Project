import logging

from app.config import settings
from app.core.errors import ERRORS
from app.core.state import state
from app.services.embedding_engine import embed_texts
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research paper assistant. Answer questions strictly based on the "
    "provided excerpts from research papers. If the answer cannot be found in the "
    "excerpts, say so clearly — do not guess or add information not present in the context."
)


def _build_context_block(rows: list[dict]) -> str:
    parts = []
    for i, row in enumerate(rows, 1):
        paper = state.papers.get(str(row["paper_id"]))
        paper_name = paper.filename if paper else "unknown"
        parts.append(
            f"[Excerpt {i} | Paper: '{paper_name}' | Section: {row['section']}]\n{row['text']}"
        )
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
    query_vec = embed_texts([question], settings.embedding_dim)[0]
    rows = state.vdb.search(
        query_vec, settings.top_k_chunks, paper_ids=paper_ids, sections=sections
    )

    if not rows:
        return {
            "question": question,
            "answer": "Insufficient context to answer this question from the selected papers.",
            "context": [],
            "grounded": False,
            "error": ERRORS["E005"].__dict__,
        }

    context_block = _build_context_block(rows)
    answer = _ask_llm(question, context_block, rows)

    return {
        "question": question,
        "answer": answer,
        "context": rows,
        "grounded": True,
    }
