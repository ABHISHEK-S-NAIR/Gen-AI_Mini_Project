from app.config import settings
from app.core.errors import ERRORS
from app.core.state import state
from app.services.embedding_engine import embed_texts


def answer_question(question: str, paper_ids: list[str]) -> dict[str, object]:
    query_vec = embed_texts([question], settings.embedding_dim)[0]
    rows = state.vdb.search(query_vec, settings.top_k_chunks, paper_ids=paper_ids)

    if not rows:
        return {
            "question": question,
            "answer": "Insufficient context to answer this question from the selected papers.",
            "context": [],
            "grounded": False,
            "error": ERRORS["E005"].__dict__,
        }

    context_text = " ".join(str(r["text"]) for r in rows)
    answer = context_text[:600]

    return {
        "question": question,
        "answer": answer,
        "context": rows,
        "grounded": True,
    }


def answer_question_with_sections(question: str, paper_ids: list[str], sections: list[str] | None) -> dict[str, object]:
    query_vec = embed_texts([question], settings.embedding_dim)[0]
    rows = state.vdb.search(query_vec, settings.top_k_chunks, paper_ids=paper_ids, sections=sections)

    if not rows:
        return {
            "question": question,
            "answer": "Insufficient context to answer this question from the selected papers.",
            "context": [],
            "grounded": False,
            "error": ERRORS["E005"].__dict__,
        }

    context_text = " ".join(str(r["text"]) for r in rows)
    answer = context_text[:600]

    return {
        "question": question,
        "answer": answer,
        "context": rows,
        "grounded": True,
    }
