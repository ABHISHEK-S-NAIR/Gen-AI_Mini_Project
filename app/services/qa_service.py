import logging
import re as _re

from app.config import settings
from app.core.errors import ERRORS
from app.core.state import state
from app.services.embedding_engine import embed_texts
from app.services.llm_client import call_llm, LLMUnavailableError
from app.utils.cache import DiskCache

logger = logging.getLogger(__name__)

# Minimum cosine similarity score for context chunks to be considered relevant
RELEVANCE_THRESHOLD = 0.35  # Tuned for SciBERT embeddings

# Cache for Q&A results (1 hour TTL)
_qa_cache = DiskCache(cache_dir="./cache/qa", max_age_seconds=3600)

_SYSTEM_PROMPT = (
    "You are a research paper assistant. Answer questions strictly based on the "
    "provided excerpts from research papers. "
    "When your answer draws on a specific excerpt, cite it inline using the format "
    "[Excerpt N] where N is the excerpt number shown in the context. "
    "If the answer cannot be found in the excerpts, say so clearly — "
    "do not guess or add information not present in the context."
)


def _lexical_jaccard(a: str, b: str) -> float:
    """Fast lexical similarity used for diversity selection."""
    ta = set(_re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(_re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _select_diverse_rows(rows: list[dict], k: int, diversity_lambda: float = 0.35) -> list[dict]:
    """
    Greedy MMR-like selection:
    maximize relevance while penalizing near-duplicate chunks.
    """
    if not rows or k <= 0:
        return []
    candidates = list(rows)
    selected: list[dict] = []

    # Seed with most relevant row.
    selected.append(candidates.pop(0))
    while candidates and len(selected) < k:
        best_idx = 0
        best_score = float("-inf")
        for i, cand in enumerate(candidates):
            relevance = float(cand.get("score", 0.0))
            redundancy = max(
                _lexical_jaccard(str(cand.get("text", "")), str(s.get("text", "")))
                for s in selected
            )
            mmr = relevance - diversity_lambda * redundancy
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(candidates.pop(best_idx))
    return selected


def _is_abstractive_question(question: str) -> bool:
    q = question.lower()
    markers = (
        "why",
        "how",
        "compare",
        "difference",
        "impact",
        "reason",
        "benefit",
        "tradeoff",
        "advantage",
        "disadvantage",
        "complementary",
    )
    return any(m in q for m in markers)


def _route_sections_for_question(question: str) -> list[str] | None:
    """Infer relevant paper sections from question intent."""
    q = question.lower()

    method_markers = (
        "how",
        "method",
        "approach",
        "architecture",
        "model",
        "algorithm",
        "technique",
        "implement",
        "design",
        "train",
    )
    results_markers = (
        "result",
        "performance",
        "accuracy",
        "score",
        "bleu",
        "rouge",
        "f1",
        "benchmark",
        "outperform",
        "beat",
        "achieve",
        "metric",
    )
    background_markers = (
        "why",
        "problem",
        "motivation",
        "background",
        "prior",
        "previous",
        "related",
        "challenge",
        "limitation",
    )
    contribution_markers = (
        "contribution",
        "novel",
        "new",
        "propose",
        "introduce",
        "different",
        "improve over",
    )

    if any(marker in q for marker in method_markers):
        return ["method", "intro"]
    if any(marker in q for marker in results_markers):
        return ["results", "conclusion"]
    if any(marker in q for marker in background_markers):
        return ["intro", "abstract"]
    if any(marker in q for marker in contribution_markers):
        return ["abstract", "intro", "conclusion"]
    return None


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
    max_excerpt_chars = 520
    for paper_id, chunks in grouped.items():
        paper = state.papers.get(str(paper_id))
        paper_name = paper.filename if paper else "unknown"
        
        for row in chunks:
            score = row.get("score", 0.0)
            chunk_text = str(row.get("text", ""))
            if len(chunk_text) > max_excerpt_chars:
                chunk_text = chunk_text[:max_excerpt_chars].rsplit(" ", 1)[0] + " ..."
            parts.append(
                f"[Excerpt {excerpt_num} | Paper: '{paper_name}' | "
                f"Section: {row['section']} | Relevance: {score:.2f}]\n{chunk_text}"
            )
            excerpt_num += 1
    
    return "\n\n".join(parts)


def _build_prompt(question: str, context_block: str, rows: list[dict], abstractive: bool = False) -> str:
    paper_count = len({row["paper_id"] for row in rows})

    if paper_count > 1:
        synthesis_instruction = (
            f"You have excerpts from {paper_count} different papers. "
            "Where the papers agree on a point, note the agreement explicitly. "
            "Where they disagree or take different approaches, highlight the contrast "
            "and name which paper takes which position."
        )
    else:
        synthesis_instruction = ""

    answer_style = (
        "Provide a synthesis answer in 2-4 sentences that explains relationships "
        "(cause/effect, contrasts, rationale) across excerpts."
        if abstractive
        else "Answer concisely with specific evidence from the excerpts."
    )

    return (
        f"Using only the excerpts below, answer the following question.\n"
        f"{synthesis_instruction}\n\n"
        f"Question: {question}\n\n"
        f"Excerpts:\n{context_block}\n\n"
        f"{answer_style} "
        f"If the excerpts do not contain enough information, say: "
        f"'The provided excerpts do not contain sufficient information to answer this question.'"
    )


def _ask_llm(
    question: str,
    context_block: str,
    rows: list[dict],
    messages: list[dict] | None = None,
    abstractive: bool = False,
) -> str:
    """
    Ask LLM to answer question based on context.
    Raises LLMUnavailableError if all providers fail.
    """
    prompt = _build_prompt(question, context_block, rows, abstractive=abstractive)
    return call_llm(
        prompt,
        system=_SYSTEM_PROMPT,
        max_tokens=320 if abstractive else 256,
        temperature=0.2,
        messages=messages,
    )


def _parse_grounded_answer(answer: str, rows: list[dict]) -> dict:
    """
    Parse [Excerpt N] tags from the LLM answer.
    Returns:
      {
        "answer": str,                      # original answer text unchanged
        "cited_excerpts": list[int],        # 1-based excerpt numbers referenced
        "cited_chunks": list[dict],         # the actual chunk rows for cited excerpts
      }
    """
    cited_numbers = [int(m) for m in _re.findall(r"\[Excerpt (\d+)\]", answer)]
    cited_numbers = sorted(set(n for n in cited_numbers if 1 <= n <= len(rows)))
    cited_chunks = [rows[n - 1] for n in cited_numbers]
    return {
        "answer": answer,
        "cited_excerpts": cited_numbers,
        "cited_chunks": cited_chunks,
    }


def answer_question(question: str, paper_ids: list[str]) -> dict[str, object]:
    return answer_question_with_sections(question, paper_ids, sections=None, conversation_id=None, debug=False)


def answer_question_with_sections(
    question: str,
    paper_ids: list[str],
    sections: list[str] | None,
    conversation_id: str | None = None,
    debug: bool = False,
) -> dict[str, object]:
    """
    Answer a question using RAG with relevance filtering and caching.
    
    Retrieves semantically similar chunks, filters by relevance threshold,
    and generates an answer using LLM with grounding verification.
    Results are cached for 1 hour to improve performance.
    """
    # Create cache key from question + papers + sections
    cache_key = f"qa:{question}:{sorted(paper_ids)}:{sorted(sections) if sections else 'all'}:{conversation_id or 'none'}"

    is_abstractive = _is_abstractive_question(question)
    relevance_threshold = RELEVANCE_THRESHOLD - 0.08 if is_abstractive else RELEVANCE_THRESHOLD
    retrieval_k = settings.top_k_chunks * 3 if is_abstractive else settings.top_k_chunks * 2

    routed_sections = None
    search_sections = sections
    if sections is None:
        routed_sections = _route_sections_for_question(question)
        search_sections = routed_sections

    debug_payload = {
        "all_candidates": [],
        "threshold_used": relevance_threshold,
        "candidates_above_threshold": 0,
        "abstractive_mode": is_abstractive,
        "routed_sections": routed_sections,
        "fallback_to_all_sections": False,
    }
    
    # Check cache first
    cached = _qa_cache.get(cache_key) if not debug else None
    if cached is not None:
        logger.info(f"Cache hit for question: {question[:50]}...")
        return cached
    
    query_vec = embed_texts([question], settings.embedding_dim)[0]
    
    # Retrieve more candidates than needed to allow filtering
    candidate_rows = state.vdb.search(
        query_vec, retrieval_k, paper_ids=paper_ids, sections=search_sections
    )

    if sections is None and routed_sections is not None:
        routed_relevant_count = sum(
            1 for row in candidate_rows if row.get("score", 0) >= relevance_threshold
        )
        if routed_relevant_count < settings.top_k_chunks:
            candidate_rows = state.vdb.search(
                query_vec, retrieval_k, paper_ids=paper_ids, sections=None
            )
            if debug:
                debug_payload["fallback_to_all_sections"] = True

    if debug:
        debug_payload["all_candidates"] = [
            {
                "chunk_id": r["chunk_id"],
                "score": round(r.get("score", 0), 4),
                "section": r["section"],
                "paper_id": r["paper_id"],
            }
            for r in candidate_rows
        ]

    if not candidate_rows:
        result = {
            "question": question,
            "answer": "No content found in the selected papers.",
            "context": [],
            "grounded": False,
            "confidence": 0.0,
            "cited_excerpts": [],
            "cited_chunks": [],
            "error": ERRORS["E005"].__dict__,
        }
        if debug:
            result["debug"] = debug_payload
        return result
    
    # Filter by relevance threshold
    relevant_rows = [r for r in candidate_rows if r.get("score", 0) >= relevance_threshold]

    if debug:
        debug_payload["candidates_above_threshold"] = len(relevant_rows)
    
    if not relevant_rows:
        # No chunks meet relevance threshold
        best_score = max(r.get("score", 0) for r in candidate_rows)
        result = {
            "question": question,
            "answer": (
                f"No sufficiently relevant context found in the selected papers. "
                f"Best match score was {best_score:.2f} (threshold: {relevance_threshold:.2f}). "
                f"Try rephrasing your question or selecting different papers."
            ),
            "context": [],
            "grounded": False,
            "confidence": 0.0,
            "cited_excerpts": [],
            "cited_chunks": [],
            "error": ERRORS["E005"].__dict__,
        }
        if debug:
            result["debug"] = debug_payload
        return result
    
    # Take top k from relevant chunks
    final_k = settings.top_k_chunks + 2 if is_abstractive else settings.top_k_chunks
    if is_abstractive:
        # Use diverse evidence for synthesis-style answers.
        rows = _select_diverse_rows(relevant_rows[: max(final_k * 3, final_k)], final_k)
    else:
        rows = relevant_rows[:final_k]
    
    context_block = _build_context_block(rows)
    prompt = _build_prompt(question, context_block, rows, abstractive=is_abstractive)

    messages = None
    if conversation_id:
        history = state.get_conversation(conversation_id)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": prompt},
        ]
    
    # Try to get LLM answer, with fallback on failure
    try:
        answer = _ask_llm(question, context_block, rows, messages=messages, abstractive=is_abstractive)
    except LLMUnavailableError as e:
        logger.error(f"LLM unavailable: {e}")
        # Return error with fallback to extracted relevant sentences
        result = {
            "question": question,
            "answer": "LLM service unavailable. Unable to generate answer.",
            "context": rows,
            "grounded": False,
            "confidence": 0.0,
            "cited_excerpts": [],
            "cited_chunks": [],
            "error": {
                "code": "E006",
                "message": "LLM service unavailable. Please configure GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY."
            },
            "fallback_context": _extract_top_sentences(rows, max_sentences=3),
        }
        if debug:
            result["debug"] = debug_payload
        return result

    if conversation_id:
        state.append_to_conversation(conversation_id, "user", question)
        state.append_to_conversation(conversation_id, "assistant", answer)

    parsed_answer = _parse_grounded_answer(answer, rows)
    cited_numbers = parsed_answer["cited_excerpts"]
    
    # Verify answer grounding
    avg_relevance = sum(r.get("score", 0) for r in rows) / len(rows)
    confidence = min(avg_relevance * 1.2, 1.0)  # Scale up slightly, cap at 1.0
    
    # Check if answer indicates insufficient information
    insufficient_indicators = [
        "do not contain sufficient",
        "cannot be found",
        "not enough information",
        "excerpts do not",
    ]
    answer_lower = answer.lower()
    is_grounded = (
        len(cited_numbers) > 0
        or not any(ind in answer_lower for ind in insufficient_indicators)
    )

    result = {
        "question": question,
        "answer": parsed_answer["answer"],
        "context": rows,
        "grounded": is_grounded,
        "confidence": round(confidence, 2),
        "avg_relevance": round(avg_relevance, 2),
        "cited_excerpts": parsed_answer["cited_excerpts"],
        "cited_chunks": parsed_answer["cited_chunks"],
    }

    if debug:
        result["debug"] = debug_payload
    
    # Cache the result
    if not debug:
        _qa_cache.set(cache_key, result)
    
    return result


def _extract_top_sentences(rows: list[dict], max_sentences: int = 3) -> str:
    """Extract most relevant sentences as fallback when LLM unavailable."""
    sentences = []
    for row in rows[:max_sentences]:
        text = row.get("text", "")
        # Take first sentence or 200 chars
        first_sent = text.split(". ")[0] if ". " in text else text[:200]
        sentences.append(first_sent)
    return " ... ".join(sentences)
