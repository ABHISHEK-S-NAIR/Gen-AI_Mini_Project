"""
Digest service: generates a 1-page executive summary across all
ingested papers in the current session.
"""
import logging

from app.core.state import state
from app.services.llm_client import LLMUnavailableError, call_llm
from app.services.structured_extraction_service import extract_structured_data

logger = logging.getLogger(__name__)

_DIGEST_SYSTEM = (
    "You are a senior research analyst writing a concise executive digest "
    "for an academic audience. You are precise, avoid filler phrases, "
    "and always ground observations in specific paper content."
)

_SECTION_WORD_LIMIT = 150


def generate_digest(paper_ids: list[str]) -> dict[str, object]:
    """
    Generate a 1-page executive digest across all provided papers.

    Returns:
    {
      "digest": str,
      "key_themes": list[str],
      "notable_results": list[str],
      "open_questions": list[str],
      "paper_count": int,
      "paper_titles": list[str],
      "error": dict | None,
    }
    """
    if not paper_ids:
        return _digest_error("No papers provided.")

    # Build structured summaries for each paper
    paper_blocks = []
    paper_titles = []
    for paper_id in paper_ids:
        paper = state.papers.get(paper_id)
        sections = state.sections.get(paper_id, {})
        structured = extract_structured_data(paper_id)
        filename = paper.filename if paper else paper_id
        paper_titles.append(filename)

        def get_section(name: str) -> str:
            text = sections.get(name, "").strip()
            return " ".join(text.split()[:_SECTION_WORD_LIMIT])

        metrics_str = (
            ", ".join(structured.get("metrics", [])[:3])
            if structured.get("metrics")
            else "not reported"
        )

        paper_blocks.append(
            f"Paper: {filename}\n"
            f"Problem:    {structured['problem']}\n"
            f"Method:     {structured['proposed_method']}\n"
            f"Technique:  {structured['core_technique']}\n"
            f"Results:    {structured['results']}\n"
            f"Metrics:    {metrics_str}\n"
            f"Novelty:    {structured['novelty']}\n"
            f"Abstract:   {get_section('abstract')}"
        )

    papers_block = "\n\n---\n\n".join(paper_blocks)
    n = len(paper_ids)

    prompt = (
        f"Write a 1-page executive digest covering {n} research paper(s) "
        "analyzed in this session.\n\n"
        "Structure your digest with exactly these four sections:\n\n"
        "OVERVIEW\n"
        f"A 2-3 sentence summary of what these {n} paper(s) are collectively about "
        "and what makes them significant.\n\n"
        "KEY THEMES (bullet list of 3-5 themes)\n"
        "The shared or contrasting themes across the papers. "
        "Name specific papers where relevant.\n\n"
        "NOTABLE RESULTS (bullet list of 3-5 results)\n"
        "The most important empirical findings. Include specific numbers where available.\n\n"
        "OPEN QUESTIONS (bullet list of 2-4 questions)\n"
        "What these papers leave unanswered. What should the field tackle next?\n\n"
        "Rules:\n"
        "- Be specific - reference papers by filename\n"
        "- Include concrete numbers from results where available\n"
        "- Do not pad with generic observations\n"
        "- Write the overview in prose; use bullet lists for the three list sections\n\n"
        f"Papers:\n{papers_block}"
    )

    try:
        digest = call_llm(
            prompt,
            system=_DIGEST_SYSTEM,
            max_tokens=900,
            temperature=0.2,
        )

        themes, results, questions = _parse_digest_sections(digest)

        return {
            "digest": digest,
            "key_themes": themes,
            "notable_results": results,
            "open_questions": questions,
            "paper_count": n,
            "paper_titles": paper_titles,
            "error": None,
        }

    except LLMUnavailableError as e:
        logger.warning(f"Digest LLM unavailable: {e}")
        return _digest_error(
            "LLM unavailable. Configure GROQ_API_KEY, GEMINI_API_KEY, "
            "or OPENROUTER_API_KEY to use this feature."
        )
    except Exception as e:
        logger.error(f"Digest generation failed: {e}")
        return _digest_error(str(e))


def _parse_digest_sections(digest: str) -> tuple[list[str], list[str], list[str]]:
    """
    Parse bullet items from KEY THEMES, NOTABLE RESULTS, OPEN QUESTIONS
    sections of the digest. Returns three lists.
    """
    import re

    def extract_section(heading: str) -> list[str]:
        pattern = re.compile(
            rf"{heading}.*?\n(.*?)(?=\n[A-Z]{{3}}|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(digest)
        if not match:
            return []
        block = match.group(1)
        items = []
        for line in block.splitlines():
            line = line.strip()
            if re.match(r"^[\-\*•\d]\s*\.?\s+", line):
                item = re.sub(r"^[\-\*•\d]\s*\.?\s+", "", line).strip()
                if item:
                    items.append(item)
        return items

    return (
        extract_section("KEY THEMES"),
        extract_section("NOTABLE RESULTS"),
        extract_section("OPEN QUESTIONS"),
    )


def _digest_error(message: str) -> dict[str, object]:
    return {
        "digest": "",
        "key_themes": [],
        "notable_results": [],
        "open_questions": [],
        "paper_count": 0,
        "paper_titles": [],
        "error": {"code": "E017", "message": message},
    }
