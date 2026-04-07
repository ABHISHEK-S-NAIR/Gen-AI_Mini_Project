"""
Automated peer review service using Chain-of-Thought + section-aware retrieval.

Approach (from ReAct / AutoRev papers):
1. RETRIEVE — pull the most review-relevant sections per paper
   (abstract, method, results, conclusion in that priority order)
2. REASON — use a CoT prompt that walks the LLM through each review
   dimension before producing final structured output
3. RESPOND — parse JSON output into the existing response schema

Fallback: if LLM is unavailable or JSON parsing fails, the old
deterministic strings are returned so the endpoint never crashes.
"""
import logging

from app.core.state import state
from app.services.llm_client import call_llm, call_llm_json
from app.services.structured_extraction_service import extract_structured_for_papers

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────

_REVIEW_SYSTEM = (
    "You are a rigorous academic peer reviewer with expertise in machine learning "
    "and NLP research. Your reviews are specific, evidence-based, and constructive. "
    "You never make generic comments — every point you raise is grounded in what the "
    "paper actually says or fails to say."
)

# ── Section retrieval ────────────────────────────────────────────────────────

_REVIEW_SECTIONS = ("abstract", "intro", "method", "results", "conclusion")
_SECTION_WORD_LIMIT = 500   # words per section fed into the prompt


def _retrieve_review_context(paper_id: str) -> str:
    """
    Pull the most review-relevant sections from the paper and format them
    as a labelled context block. This is the 'retrieve' step from ReAct.
    """
    sections = state.sections.get(paper_id, {})
    parts = []
    for sec in _REVIEW_SECTIONS:
        text = sections.get(sec, "").strip()
        if text:
            # Truncate long sections to keep prompt within token limits
            words = text.split()
            truncated = " ".join(words[:_SECTION_WORD_LIMIT])
            if len(words) > _SECTION_WORD_LIMIT:
                truncated += " [... truncated]"
            parts.append(f"=== {sec.upper()} ===\n{truncated}")
    return "\n\n".join(parts) if parts else "No section content available."


# ── Chain-of-Thought review prompt ──────────────────────────────────────────

def _build_cot_review_prompt(paper_name: str, context: str, paper_id: str) -> str:
    """
    Build a Chain-of-Thought prompt that walks the LLM through each review
    dimension before producing the final structured JSON output.
    This implements the ReAct pattern: reason step by step, then act (output).
    """
    return f"""You are reviewing the research paper: "{paper_name}"

Here are the key sections of the paper:

{context}

Think through this review step by step before writing your final output.

STEP 1 — PROBLEM & NOVELTY
What problem does this paper address? Is the motivation clearly stated?
Is the proposed contribution genuinely novel compared to prior work mentioned?

STEP 2 — METHODOLOGY SOUNDNESS  
Is the proposed method technically sound and well-described?
Are there gaps, unjustified design choices, or missing details in the method section?

STEP 3 — EXPERIMENTAL RIGOR
Are the experiments sufficient to support the claims?
Are baselines appropriate? Are metrics reported clearly with numbers?
Is there evidence of ablation studies or error analysis?

STEP 4 — CLARITY & PRESENTATION
Is the paper well-structured and clearly written?
Are there sections that are confusing, incomplete, or poorly motivated?

STEP 5 — FINAL STRUCTURED REVIEW
Based on your analysis above, produce your final peer review.

Respond ONLY with this JSON (no markdown fences, no extra text):
{{
  "paper_id": "{paper_id}",
  "paper_name": "{paper_name}",
  "strengths": [
    "specific strength grounded in paper content",
    "another specific strength",
    "third strength if applicable"
  ],
  "weaknesses": [
    "specific weakness grounded in paper content",
    "another specific weakness"
  ],
  "suggestions": [
    "specific actionable suggestion tied to a weakness",
    "another specific suggestion"
  ],
  "overall_assessment": "One paragraph summary of the paper quality, contribution, and recommendation (accept/revise/reject with reasoning)."
}}"""


# ── Single paper review ──────────────────────────────────────────────────────

def _review_single_paper(paper_id: str, structured: dict) -> dict[str, object]:
    """
    Review a single paper using CoT prompting over retrieved sections.
    Falls back to deterministic output if LLM is unavailable.
    """
    paper = state.papers.get(paper_id)
    paper_name = paper.filename if paper else structured.get("title", "unknown.pdf")

    context = _retrieve_review_context(paper_id)
    prompt = _build_cot_review_prompt(paper_name, context, paper_id)

    result = call_llm_json(prompt, system=_REVIEW_SYSTEM, max_tokens=900)

    # Validate the parsed result has required keys
    required_keys = {"strengths", "weaknesses", "suggestions"}
    if result and required_keys.issubset(result.keys()):
        # Ensure all values are lists of strings
        for key in ("strengths", "weaknesses", "suggestions"):
            if not isinstance(result[key], list):
                result[key] = [str(result[key])]
        result.setdefault("paper_id", paper_id)
        result.setdefault("paper_name", paper_name)
        result.setdefault("overall_assessment", "")
        return result

    # Fallback: deterministic output if JSON parse failed or LLM unavailable
    logger.warning(f"LLM review failed for {paper_id}, using deterministic fallback.")
    metrics_str = ", ".join(structured.get("metrics", [])[:3]) or "no explicit metrics"
    return {
        "paper_id": paper_id,
        "paper_name": paper_name,
        "strengths": [
            f"Method articulation is concrete: {structured['proposed_method']}",
            f"Core technical lever is identifiable: {structured['core_technique']}",
            f"Result reporting includes measurable signal: {metrics_str}",
        ],
        "weaknesses": [
            f"Extracted limitations indicate unresolved issues: {structured['limitations']}",
            "Generalization and robustness evidence is not explicitly rich in extracted text.",
        ],
        "suggestions": [
            "Provide targeted ablations that isolate the unique contribution from confounding factors.",
            "Expand failure-case and boundary-condition reporting to support deployment confidence.",
        ],
        "overall_assessment": "",
    }


# ── Multi-paper comparison ───────────────────────────────────────────────────

def _compare_papers(reviews: list[dict]) -> str | None:
    """
    Generate a comparative review across multiple papers using LLM.
    Falls back to a template string if LLM is unavailable.
    """
    if len(reviews) < 2:
        return None

    names = [r.get("paper_name", f"Paper {i+1}") for i, r in enumerate(reviews)]
    strengths_block = "\n".join(
        f"{names[i]} strengths: {'; '.join(r.get('strengths', [])[:2])}"
        for i, r in enumerate(reviews)
    )
    weaknesses_block = "\n".join(
        f"{names[i]} weaknesses: {'; '.join(r.get('weaknesses', [])[:2])}"
        for i, r in enumerate(reviews)
    )

    prompt = f"""You have reviewed {len(reviews)} research papers. Write a comparative analysis.

{strengths_block}

{weaknesses_block}

In 3-4 sentences:
1. What is the shared research theme or problem area?
2. How do their methodological approaches differ?
3. Which paper makes a stronger empirical case and why?

Be specific. Reference paper names directly."""

    try:
        return call_llm(prompt, system=_REVIEW_SYSTEM, max_tokens=300, temperature=0.3)
    except Exception as e:
        logger.warning(f"Comparison LLM call failed: {e}")
        # Deterministic fallback
        if len(reviews) >= 2:
            return (
                f"Comparative Review: {names[0]} and {names[1]} address related problems "
                f"with differing methodological emphasis. "
                f"A detailed comparison requires LLM availability."
            )
        return None


# ── Public entry point ───────────────────────────────────────────────────────

def review(paper_ids: list[str]) -> dict[str, object]:
    """
    Main review function. Called by task_router for the 'review' task.
    Response shape is preserved exactly from the original implementation.
    """
    structured_items = extract_structured_for_papers(paper_ids)
    reviews = [
        _review_single_paper(pid, structured)
        for pid, structured in zip(paper_ids, structured_items)
    ]
    comparison = _compare_papers(reviews)
    return {"reviews": reviews, "comparison": comparison}
