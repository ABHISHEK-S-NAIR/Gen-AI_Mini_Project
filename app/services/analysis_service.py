import os

from app.core.state import state
from app.services.citation_service import analyse_citations_for_papers
from app.services.llm_client import call_llm_json
from app.services.structured_extraction_service import extract_structured_for_papers


_ANALYSIS_SYSTEM = (
    "You are a precise research-paper analyst. "
    "Use only provided text and return strict JSON without markdown fences."
)


def _trim_words(text: str, max_words: int = 180) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _diagram_for(technique: str) -> str:
    t = technique.lower()
    if "self-attention" in t or "transformer" in t:
        return "[Input Tokens] --> [Token Embedding] --> [Multi-Head Self-Attention] --> [Feed-Forward Layers] --> [Task Output]"
    if "mlm" in t:
        return "[Input Tokens] --> [Masking Step] --> [Bidirectional Encoder] --> [MLM/NSP Objectives] --> [Fine-Tuned Output]"
    if "retrieval" in t:
        return "[Question/Input] --> [Retriever] --> [Relevant Documents] --> [Generator] --> [Grounded Output]"
    return "[Input] --> [Model Encoding] --> [Learning Module] --> [Prediction]"


def _citation_insight_for(citations: list[dict[str, str]]) -> str:
    counts = {"supporting": 0, "contrasting": 0, "neutral": 0}
    for c in citations:
        ctype = c.get("type", "neutral")
        if ctype in counts:
            counts[ctype] += 1
    total = sum(counts.values())
    if total == 0:
        return "No citation evidence available to infer positioning."

    distribution = (
        f"Supporting={counts['supporting']}, Contrasting={counts['contrasting']}, Neutral={counts['neutral']}."
    )
    if counts["contrasting"] > counts["supporting"] and counts["contrasting"] >= counts["neutral"]:
        implication = "The paper positions itself by challenging prior approaches, signaling strong motivation pressure."
    elif counts["supporting"] >= counts["contrasting"] and counts["supporting"] > 0:
        implication = "The paper is positioned with validation-oriented references, strengthening confidence in its direction."
    else:
        implication = "The paper relies more on background references, so novelty positioning depends on method/results clarity."
    return f"Distribution: {distribution} Implication: {implication}"


def _comparison_block(structured_items: list[dict[str, str]]) -> str:
    """
    Generate a structured comparison of multiple papers using LLM.
    Falls back to a deterministic template if LLM is unavailable or
    fewer than 2 papers are provided.
    """
    if len(structured_items) < 2:
        return "Single-paper run: comparison not applicable."

    # Build a compact summary of each paper for the prompt
    paper_summaries = []
    for idx, s in enumerate(structured_items, start=1):
        metrics_str = (
            ", ".join(s.get("metrics", [])[:3])
            if s.get("metrics")
            else "not reported"
        )
        paper_summaries.append(
            f"Paper {idx}: {s.get('title', '')}\n"
            f"  Problem:    {s.get('problem', '')}\n"
            f"  Method:     {s.get('proposed_method', '')}\n"
            f"  Technique:  {s.get('core_technique', '')}\n"
            f"  Results:    {s.get('results', '')}\n"
            f"  Metrics:    {metrics_str}\n"
            f"  Novelty:    {s.get('novelty', '')}\n"
            f"  Limitation: {s.get('limitations', '')}"
        )

    papers_block = "\n\n".join(paper_summaries)
    n = len(structured_items)

    prompt = (
        f"You are comparing {n} research papers. "
        "Write a focused comparative analysis structured into exactly "
        "four short paragraphs with these headings:\n\n"
        "1. Problem framing - How do the papers frame their research "
        "problems? Do they target the same gap or different ones?\n"
        "2. Methodology - How do their technical approaches differ? "
        "Use each paper's own terminology.\n"
        "3. Evaluation rigor - Compare the strength of their experimental "
        "evidence. Which paper has stronger empirical support and why?\n"
        "4. Practical impact - Which contribution is more likely to be "
        "adopted in practice and why?\n\n"
        "Be specific. Name each paper by its title. "
        "Do not use bullet points - write in prose.\n\n"
        f"Papers:\n{papers_block}"
    )

    system = (
        "You are a rigorous academic analyst. "
        "Write clear, evidence-based comparative analysis in prose. "
        "Never use generic filler phrases."
    )

    try:
        from app.services.llm_client import call_llm
        result = call_llm(prompt, system=system, max_tokens=600, temperature=0.2)
        if result and "[LLM_UNAVAILABLE" not in result:
            return result
    except Exception:
        pass

    # Deterministic fallback
    first = structured_items[0] if structured_items else {}
    second = structured_items[1] if len(structured_items) > 1 else {}
    lines = ["Comparison (LLM unavailable - template fallback)"]
    for idx, s in enumerate(structured_items, start=1):
        lines.append(f"Paper {idx}: {s.get('title', '')}")
        lines.append(f"  Architecture: {s.get('architecture', '')}")
        lines.append(f"  Approach: {s.get('proposed_method', '')}")
    lines.append(
        f"Key difference: Paper 1 uses {first.get('core_technique', '')} "
        f"with {first.get('learning_strategy', '')}, while Paper 2 uses "
        f"{second.get('core_technique', '')} with {second.get('learning_strategy', '')}."
    )
    return "\n".join(lines)


def _score_novelty(structured: dict[str, str]) -> dict[str, object]:
    """
    Ask the LLM to score the novelty of this paper's core contribution
    on a 0.0-1.0 scale relative to the prior work it cites.

    Returns:
      {
        "novelty_score": float,      # 0.0-1.0, rounded to 2 dp
        "novelty_rationale": str,    # one sentence justification
      }

    Falls back to {"novelty_score": None, "novelty_rationale": ""}
    if LLM is unavailable or response is unparseable.
    """
    prompt = (
        "You are evaluating the novelty of a research paper's core contribution.\n\n"
        "Paper details:\n"
        f"  Title:      {structured.get('title', '')}\n"
        f"  Problem:    {structured.get('problem', '')}\n"
        f"  Technique:  {structured.get('core_technique', '')}\n"
        f"  Novelty:    {structured.get('novelty', '')}\n"
        f"  Prior work: {structured.get('limitations', '')}\n\n"
        "Task: On a scale from 0.0 to 1.0, how novel is the core contribution "
        "of this paper compared to the prior work it cites?\n\n"
        "Scoring guide:\n"
        "  0.0-0.2  Incremental improvement on a well-known method\n"
        "  0.3-0.5  Meaningful extension with some new ideas\n"
        "  0.6-0.8  Significant new approach or technique\n"
        "  0.9-1.0  Foundational new idea with broad impact potential\n\n"
        "Return ONLY a JSON object with exactly two keys:\n"
        '  "novelty_score": <float between 0.0 and 1.0>,\n'
        '  "novelty_rationale": <one sentence justification, max 25 words>\n\n'
        "No markdown fences. No extra text."
    )

    system = (
        "You are a precise academic evaluator. "
        "Return only valid JSON with a float and a string. No commentary."
    )

    try:
        result = call_llm_json(prompt, system=system, max_tokens=120)

        if not isinstance(result, dict):
            raise ValueError("Not a dict")

        score = result.get("novelty_score")
        rationale = result.get("novelty_rationale", "")

        # Validate score is a float in [0, 1]
        score = float(score)
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Score out of range: {score}")

        return {
            "novelty_score": round(score, 2),
            "novelty_rationale": str(rationale).strip(),
        }

    except Exception:
        return {"novelty_score": None, "novelty_rationale": ""}


def _insight_summary(
    structured_items: list[dict[str, str]],
    analyses: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    if not structured_items:
        return {
            "insight_summary": "No structured data available.",
            "why_it_matters": "No direct implication available.",
            "evolution": "No evolution path available.",
        }

    if len(structured_items) == 1:
        s = structured_items[0]
        metrics_str = ", ".join(s.get('metrics', [])[:3]) if s.get('metrics') else "no explicit metrics"
        analysis_item = analyses[0] if analyses else {}
        key_idea = str(analysis_item.get("key_idea", "") or s.get("novelty", "")).strip()
        summary = str(analysis_item.get("summary", "") or s.get("problem", "")).strip()
        return {
            "insight_summary": f"The core contribution is {key_idea or 'not clearly specified in the paper'}.",
            "why_it_matters": (
                f"It matters because it targets {summary or 'the stated research problem'} "
                f"with measurable signal {metrics_str}."
            ),
            "evolution": f"Prior baselines -> {s['title']} -> follow-up methods in the same task family.",
        }

    s1 = structured_items[0]
    s2 = structured_items[1]
    return {
        "insight_summary": (
            f"{s1['title']} and {s2['title']} contribute differently via "
            f"{s1['contribution_type']} vs {s2['contribution_type']} advances."
        ),
        "why_it_matters": "Together they show complementary routes to improve performance and adoption in real systems.",
        "evolution": f"{s1['title']} -> {s2['title']} -> broader LLM-era optimization and scaling work.",
    }


def _get_paper_full_text(paper_id: str) -> str:
    """Concatenate all non-empty sections of a paper into one string."""
    sections = state.sections.get(paper_id, {})
    parts = []
    for sec in ("abstract", "intro", "method", "results", "conclusion"):
        text = sections.get(sec, "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _summarize_paper_fields(paper_id: str, structured: dict) -> dict[str, str]:
    """
    Generate real summarized text for the four analysis fields.
    Uses section-specific text as input to the summarizer wherever available,
    falling back to structured extraction strings if a section is empty.
    """
    sections = state.sections.get(paper_id, {})

    # Use raw section text as input when available; fall back to structured strings
    abstract_text = sections.get("abstract", "").strip()
    method_text = sections.get("method", "").strip()
    results_text = sections.get("results", "").strip()
    intro_text = sections.get("intro", "").strip()
    conclusion_text = sections.get("conclusion", "").strip()

    # Results fallback helper text
    metrics_str = ", ".join(structured.get('metrics', [])[:3]) if structured.get('metrics') else ""

    fallback = {
        "summary": f"The paper addresses {structured['problem']} and proposes {structured['proposed_method']}.",
        "methodology": f"The method is organized around {structured['proposed_method']} with a {structured['learning_strategy']} strategy.",
        "key_idea": f"The central idea is {structured['novelty']}." or structured.get("novelty", ""),
        "results": f"Reported results indicate {structured['results']} with metric evidence {metrics_str}.",
    }

    # Keep prompt compact to stay under provider token/request limits.
    compact_abstract = _trim_words(abstract_text, 160)
    compact_intro = _trim_words(intro_text, 140)
    compact_method = _trim_words(method_text, 220)
    compact_results = _trim_words(results_text, 180)
    compact_conclusion = _trim_words(conclusion_text, 120)

    prompt = (
        "Analyze this paper content and return a JSON object with exactly these keys: "
        '"summary", "methodology", "key_idea", "results".\n\n'
        "Requirements:\n"
        "- Each value should be concise (1-3 sentences).\n"
        "- Methodology must describe concrete procedure, not generic wording.\n"
        "- If a detail is unclear from text, say 'not clearly specified in the paper'.\n"
        "- Use only given content.\n\n"
        f"Extracted fields:\n"
        f"problem={structured.get('problem', '')}\n"
        f"proposed_method={structured.get('proposed_method', '')}\n"
        f"core_technique={structured.get('core_technique', '')}\n"
        f"learning_strategy={structured.get('learning_strategy', '')}\n"
        f"novelty={structured.get('novelty', '')}\n"
        f"results={structured.get('results', '')}\n"
        f"metrics={structured.get('metrics', [])}\n\n"
        f"[ABSTRACT]\n{compact_abstract or 'N/A'}\n\n"
        f"[INTRO]\n{compact_intro or 'N/A'}\n\n"
        f"[METHOD]\n{compact_method or 'N/A'}\n\n"
        f"[RESULTS]\n{compact_results or 'N/A'}\n\n"
        f"[CONCLUSION]\n{compact_conclusion or 'N/A'}"
    )

    try:
        if not (
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        ):
            return fallback

        response = call_llm_json(prompt, system=_ANALYSIS_SYSTEM, max_tokens=500)
        if not isinstance(response, dict):
            return fallback

        out = {
            "summary": str(response.get("summary", "")).strip() or fallback["summary"],
            "methodology": str(response.get("methodology", "")).strip() or fallback["methodology"],
            "key_idea": str(response.get("key_idea", "")).strip() or fallback["key_idea"],
            "results": str(response.get("results", "")).strip() or fallback["results"],
        }
        return out
    except Exception:
        return fallback



def analyse(paper_ids: list[str]) -> dict[str, object]:
    structured_items = extract_structured_for_papers(paper_ids)
    citation_payload = analyse_citations_for_papers(paper_ids)
    citation_map = {
        p.get("paper_id"): p.get("citations", [])
        for p in citation_payload.get("papers", [])
        if isinstance(p, dict)
    }

    analyses: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for s in structured_items:
        # NEW — real summarization
        summarized = _summarize_paper_fields(s["paper_id"], s)
        novelty_data = _score_novelty(s)
        summary = summarized["summary"]
        methodology = summarized["methodology"]
        key_idea = summarized["key_idea"]
        results = summarized["results"]
        metrics_str = ", ".join(s.get('metrics', [])[:3]) if s.get('metrics') else "no explicit metrics"

        analyses.append(
            {
                "paper_id": s["paper_id"],
                "paper_name": s["title"],
                "summary": summary,
                "methodology": methodology,
                "key_idea": key_idea,
                "results": results,
                "novelty_score": novelty_data["novelty_score"],
                "novelty_rationale": novelty_data["novelty_rationale"],
                "analysis_text": (
                    f"This paper focuses on: {summary}\n\n"
                    f"Method: {methodology}\n\n"
                    f"Results: {results}"
                ),
                "citation_insight": _citation_insight_for(citation_map.get(s["paper_id"], [])),
            }
        )

        reports.append(
            {
                "paper_id": s["paper_id"],
                "paper_name": s["title"],
                "paper_review": {
                    "strengths": [
                        f"Method description: {s['proposed_method']}",
                        f"Technique anchor: {s['core_technique']}",
                    ],
                    "weaknesses": [f"Limitation evidence: {s['limitations']}"],
                    "suggestions": ["Add ablations and stronger robustness analysis."],
                },
                "explanation": {
                    "beginner": f"The paper solves {s['problem']} using {s['core_technique']}.",
                    "intermediate": f"Method: {s['proposed_method']}. Result: {s['results']} ({metrics_str}).",
                    "expert": (
                        f"Architectural/training interpretation: {s['architecture']} with {s['learning_strategy']}; "
                        f"novelty = {s['novelty']}"
                    ),
                },
                "citation_insight": _citation_insight_for(citation_map.get(s["paper_id"], [])),
                "structured_data": {
                    "problem": s["problem"],
                    "method": s["proposed_method"],
                    "key_technique": s["core_technique"],
                    "results": s["results"],
                    "novelty": s["novelty"],
                    "limitations": s["limitations"],
                    "novelty_score": novelty_data["novelty_score"],
                    "novelty_rationale": novelty_data["novelty_rationale"],
                },
            }
        )

    insight = _insight_summary(structured_items, analyses=analyses)
    return {
        "structured_data": [
            {
                "paper_id": s["paper_id"],
                "title": s["title"],
                "problem": s["problem"],
                "proposed_method": s["proposed_method"],
                "core_technique": s["core_technique"],
                "results": s["results"],
                "novelty": s["novelty"],
                "limitations": s["limitations"],
            }
            for s in structured_items
        ],
        "analyses": analyses,
        "comparison": _comparison_block(structured_items),
        "comparative_analysis": _comparison_block(structured_items),
        "reports": reports,
        "insight_summary": insight["insight_summary"],
        "why_it_matters": insight["why_it_matters"],
        "evolution": insight["evolution"],
        "novelty_scores": [
            {
                "paper_id": s["paper_id"],
                "title": s["title"],
                "novelty_score": a["novelty_score"],
                "novelty_rationale": a["novelty_rationale"],
            }
            for s, a in zip(structured_items, analyses)
        ],
    }
