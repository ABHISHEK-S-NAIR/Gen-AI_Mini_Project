"""
Synthesis service: research gap finder and hypothesis generator.
Both operate across all selected papers and require 2+ papers minimum.
Gap finder identifies unanswered questions; hypothesis generator proposes
forward-looking testable hypotheses.
"""
import logging

from app.core.state import state
from app.services.llm_client import LLMUnavailableError, call_llm

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM = (
    "You are an expert research analyst with deep knowledge of academic literature. "
    "Your analysis is specific, grounded in the provided paper content, "
    "and avoids generic observations."
)

_SECTION_WORD_LIMIT = 250  # words per section per paper fed into prompts


def _build_paper_summaries(paper_ids: list[str]) -> list[dict[str, str]]:
    """
    Build a compact representation of each paper for synthesis prompts.
    Uses abstract + conclusion as the most information-dense sections.
    Falls back to intro + results if those are empty.
    Returns a list of dicts with keys: paper_id, filename, abstract,
    conclusion, method, results.
    """
    summaries = []
    for paper_id in paper_ids:
        paper = state.papers.get(paper_id)
        sections = state.sections.get(paper_id, {})

        def get_section(name: str) -> str:
            text = sections.get(name, "").strip()
            words = text.split()
            return " ".join(words[:_SECTION_WORD_LIMIT])

        summaries.append(
            {
                "paper_id": paper_id,
                "filename": paper.filename if paper else paper_id,
                "abstract": get_section("abstract") or get_section("intro"),
                "conclusion": get_section("conclusion") or get_section("results"),
                "method": get_section("method"),
                "results": get_section("results"),
            }
        )
    return summaries


def _format_papers_block(summaries: list[dict[str, str]]) -> str:
    """Format paper summaries into a numbered block for LLM prompts."""
    parts = []
    for idx, s in enumerate(summaries, start=1):
        parts.append(
            f"--- Paper {idx}: {s['filename']} ---\n"
            f"Abstract/Intro:\n{s['abstract']}\n\n"
            f"Methods:\n{s['method']}\n\n"
            f"Results/Conclusion:\n{s['conclusion'] or s['results']}"
        )
    return "\n\n".join(parts)


def find_research_gaps(paper_ids: list[str]) -> dict[str, object]:
    """
    Identify unanswered research questions and missing experiments
    across the provided papers.

    Returns:
    {
      "gaps": list[str],           # each gap as a standalone sentence
      "missing_experiments": list[str],
      "followup_directions": list[str],
      "synthesis": str,            # full LLM prose response
      "paper_count": int,
      "error": dict | None,
    }
    """
    if not paper_ids:
        return _gap_error("No papers provided.")

    summaries = _build_paper_summaries(paper_ids)
    papers_block = _format_papers_block(summaries)
    n = len(paper_ids)

    prompt = (
        f"You have read {n} research paper(s). Based strictly on what these "
        "papers say and what they leave unaddressed, answer the following:\n\n"
        "1. UNANSWERED QUESTIONS (list 3-5)\n"
        "   What research questions do these papers raise but not fully answer? "
        "   Be specific - reference the papers by name.\n\n"
        "2. MISSING EXPERIMENTS (list 2-4)\n"
        "   What experiments or evaluations are conspicuously absent? "
        "   What would a rigorous reviewer demand?\n\n"
        "3. FOLLOW-UP DIRECTIONS (list 2-3)\n"
        "   What would make a strong follow-up paper? What gap in the "
        "   literature would it fill?\n\n"
        "Format your response with the three numbered sections above. "
        "Under each section use a simple numbered list. "
        "Be specific - avoid generic statements that could apply to any paper.\n\n"
        f"Papers:\n{papers_block}"
    )

    try:
        synthesis = call_llm(
            prompt,
            system=_SYNTHESIS_SYSTEM,
            max_tokens=800,
            temperature=0.3,
        )

        gaps, missing, followup = _parse_gap_sections(synthesis)

        return {
            "gaps": gaps,
            "missing_experiments": missing,
            "followup_directions": followup,
            "synthesis": synthesis,
            "paper_count": n,
            "error": None,
        }

    except LLMUnavailableError as e:
        logger.warning(f"Gap finder LLM unavailable: {e}")
        return _gap_error(
            "LLM unavailable. Configure GROQ_API_KEY, GEMINI_API_KEY, "
            "or OPENROUTER_API_KEY to use this feature."
        )
    except Exception as e:
        logger.error(f"Gap finder failed: {e}")
        return _gap_error(str(e))


def _parse_gap_sections(synthesis: str) -> tuple[list[str], list[str], list[str]]:
    """
    Parse the structured LLM response into three separate lists.
    Each section is identified by its number (1., 2., 3.) or heading keyword.
    Within each section, lines starting with a digit or dash become list items.
    Falls back to putting the full text in gaps[0] if parsing fails.
    """
    import re

    sections = re.split(
        r"\n\s*(?:\d\.\s+(?:UNANSWERED|MISSING|FOLLOW)|(?:UNANSWERED|MISSING|FOLLOW))",
        synthesis,
        flags=re.IGNORECASE,
    )

    def extract_items(text: str) -> list[str]:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^[\d\-\*•]\s*\.?\s+", line):
                item = re.sub(r"^[\d\-\*•]\s*\.?\s+", "", line).strip()
                if item:
                    items.append(item)
        return items if items else ([text.strip()] if text.strip() else [])

    if len(sections) >= 4:
        return (
            extract_items(sections[1]),
            extract_items(sections[2]),
            extract_items(sections[3]),
        )
    if len(sections) == 1:
        return ([synthesis.strip()], [], [])
    all_items = extract_items(synthesis)
    third = max(1, len(all_items) // 3)
    return all_items[:third], all_items[third : 2 * third], all_items[2 * third :]


def _gap_error(message: str) -> dict[str, object]:
    return {
        "gaps": [],
        "missing_experiments": [],
        "followup_directions": [],
        "synthesis": "",
        "paper_count": 0,
        "error": {"code": "E014", "message": message},
    }


def generate_hypotheses(paper_ids: list[str]) -> dict[str, object]:
    """
    Propose novel, testable research hypotheses based on the methods
    and results of the provided papers.

    Returns:
    {
      "hypotheses": list[dict],   # each: {title, description, rationale, testability}
      "synthesis": str,           # full LLM prose response
      "paper_count": int,
      "error": dict | None,
    }
    """
    if not paper_ids:
        return _hypothesis_error("No papers provided.")

    summaries = _build_paper_summaries(paper_ids)
    papers_block = _format_papers_block(summaries)
    n = len(paper_ids)

    prompt = (
        f"You have read {n} research paper(s). Based on their methods and "
        "results, propose exactly 3 novel research hypotheses that:\n"
        "  - Are forward-looking and could be tested with currently available tools\n"
        "  - Build directly on the findings in these papers\n"
        "  - Are specific enough to be falsifiable\n\n"
        "For each hypothesis, provide:\n"
        "  HYPOTHESIS [N]: <one-sentence hypothesis statement>\n"
        "  Description: <2-3 sentences explaining the idea>\n"
        "  Rationale: <why this follows from the papers' findings>\n"
        "  How to test: <concrete experimental approach in 1-2 sentences>\n\n"
        "Be specific - use the papers' own terminology. "
        "Do not propose hypotheses that are already addressed in the papers.\n\n"
        f"Papers:\n{papers_block}"
    )

    try:
        synthesis = call_llm(
            prompt,
            system=_SYNTHESIS_SYSTEM,
            max_tokens=900,
            temperature=0.4,
        )

        hypotheses = _parse_hypotheses(synthesis)

        return {
            "hypotheses": hypotheses,
            "synthesis": synthesis,
            "paper_count": n,
            "error": None,
        }

    except LLMUnavailableError as e:
        logger.warning(f"Hypothesis generator LLM unavailable: {e}")
        return _hypothesis_error(
            "LLM unavailable. Configure GROQ_API_KEY, GEMINI_API_KEY, "
            "or OPENROUTER_API_KEY to use this feature."
        )
    except Exception as e:
        logger.error(f"Hypothesis generator failed: {e}")
        return _hypothesis_error(str(e))


def _parse_hypotheses(synthesis: str) -> list[dict[str, str]]:
    """
    Parse structured hypothesis blocks from LLM output.
    Each block starts with HYPOTHESIS [N]: and has Description/Rationale/
    How to test sub-fields. Returns list of dicts with keys:
    title, description, rationale, testability.
    Falls back to a single dict with the full synthesis if parsing fails.
    """
    import re

    blocks = re.split(r"\n\s*HYPOTHESIS\s+\[?\d\]?\s*:", synthesis, flags=re.IGNORECASE)
    hypotheses = []

    for block in blocks[1:]:  # skip text before first hypothesis
        lines = block.strip().splitlines()
        title = lines[0].strip() if lines else ""

        def extract_field(keyword: str) -> str:
            pattern = re.compile(
                rf"{keyword}\s*:\s*(.+?)(?=\n\s*(?:Description|Rationale|How to test|HYPOTHESIS)|$)",
                re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(block)
            return match.group(1).strip() if match else ""

        hypotheses.append(
            {
                "title": title,
                "description": extract_field("Description"),
                "rationale": extract_field("Rationale"),
                "testability": extract_field("How to test"),
            }
        )

    if not hypotheses:
        return [
            {
                "title": "Generated hypothesis",
                "description": synthesis,
                "rationale": "",
                "testability": "",
            }
        ]

    return hypotheses[:3]  # cap at 3


def _hypothesis_error(message: str) -> dict[str, object]:
    return {
        "hypotheses": [],
        "synthesis": "",
        "paper_count": 0,
        "error": {"code": "E015", "message": message},
    }
