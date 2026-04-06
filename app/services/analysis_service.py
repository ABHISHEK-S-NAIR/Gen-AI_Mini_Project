from app.core.state import state
from app.services.citation_service import analyse_citations_for_papers
from app.services.structured_extraction_service import extract_structured_for_papers
from app.services.summarization_engine import summarize, summarize_multiple


def _sanitize_analysis_text(text: str) -> str:
    lowered = text
    forbidden = {
        "strength": "",
        "weakness": "",
        "suggestion": "",
        "better": "",
        "good": "",
        "limitation": "",
    }
    for bad, repl in forbidden.items():
        lowered = lowered.replace(bad, repl).replace(bad.capitalize(), repl)
    return " ".join(lowered.split())


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
    if len(structured_items) < 2:
        return "Single-paper run: comparison not applicable."

    lines = ["Comparison"]
    for idx, s in enumerate(structured_items, start=1):
        lines.append(f"Paper {idx}:")
        lines.append(f"- Architecture: {s['architecture']}")
        lines.append(f"- Approach: {s['proposed_method']}")

    first = structured_items[0]
    second = structured_items[1]
    lines.append(
        "Key Difference: "
        f"Paper 1 emphasizes {first['core_technique']} with {first['learning_strategy']}, "
        f"while Paper 2 emphasizes {second['core_technique']} with {second['learning_strategy']}."
    )
    return "\n".join(lines)


def _insight_summary(structured_items: list[dict[str, str]]) -> dict[str, str]:
    if not structured_items:
        return {
            "insight_summary": "No structured data available.",
            "why_it_matters": "No direct implication available.",
            "evolution": "No evolution path available.",
        }

    if len(structured_items) == 1:
        s = structured_items[0]
        metrics_str = ", ".join(s.get('metrics', [])[:3]) if s.get('metrics') else "no explicit metrics"
        return {
            "insight_summary": f"The core contribution is {s['novelty']}",
            "why_it_matters": f"It matters because it targets {s['problem']} with measurable signal {metrics_str}.",
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

    # Prepare all inputs for batch summarization
    inputs = []
    fallbacks = []
    
    # Summary: abstract + intro give the best overview
    summary_input = " ".join(filter(None, [abstract_text, intro_text]))
    if not summary_input:
        summary_input = f"{structured['problem']} {structured['proposed_method']}"
    inputs.append(summary_input)
    fallbacks.append(_sanitize_analysis_text(
        f"The paper addresses {structured['problem']} and proposes {structured['proposed_method']}."
    ))

    # Methodology: method section is the ground truth
    methodology_input = method_text or f"{structured['core_technique']} {structured['learning_strategy']}"
    inputs.append(methodology_input)
    fallbacks.append(_sanitize_analysis_text(
        f"The method is organized around {structured['core_technique']} with a {structured['learning_strategy']} strategy."
    ))

    # Key idea: abstract is usually the most distilled statement of novelty
    key_idea_input = abstract_text or structured['novelty']
    if len(key_idea_input.split()) > 30:
        inputs.append(key_idea_input)
        fallbacks.append(_sanitize_analysis_text(f"The central idea is {structured['novelty']}."))
    else:
        # Short enough, don't summarize
        inputs.append("")  # empty placeholder
        fallbacks.append(key_idea_input)

    # Results: results section directly
    metrics_str = ", ".join(structured.get('metrics', [])[:3]) if structured.get('metrics') else ""
    results_input = " ".join(filter(None, [results_text, conclusion_text]))
    if not results_input:
        results_input = f"{structured['results']} {metrics_str}"
    inputs.append(results_input)
    fallbacks.append(_sanitize_analysis_text(f"Reported results indicate {structured['results']} with metric evidence {metrics_str}."))
    
    # Batch summarize all at once
    try:
        summaries = summarize_multiple(inputs)
        # Use fallbacks for empty results or failures
        summary = summaries[0] if summaries[0] else fallbacks[0]
        methodology = summaries[1] if summaries[1] else fallbacks[1]
        key_idea = summaries[2] if summaries[2] else fallbacks[2]
        results_summary = summaries[3] if summaries[3] else fallbacks[3]
    except Exception as e:
        # Complete fallback to template strings
        summary = fallbacks[0]
        methodology = fallbacks[1]
        key_idea = fallbacks[2]
        results_summary = fallbacks[3]

    return {
        "summary": summary,
        "methodology": methodology,
        "key_idea": key_idea,
        "results": results_summary,
    }



def analyse(paper_ids: list[str]) -> dict[str, object]:
    structured_items = extract_structured_for_papers(paper_ids)
    citation_payload = analyse_citations_for_papers(paper_ids)
    citation_map = {
        p.get("paper_id"): p.get("citations", [])
        for p in citation_payload.get("papers", [])
        if isinstance(p, dict)
    }

    analyses: list[dict[str, str]] = []
    reports: list[dict[str, object]] = []
    for s in structured_items:
        # NEW — real summarization
        summarized = _summarize_paper_fields(s["paper_id"], s)
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
                "analysis_text": (
                    f"This paper focuses on: {summary}\n\n"
                    f"Method: {methodology}\n\n"
                    f"Results: {results}"
                ),
                "text_diagram": _diagram_for(s["core_technique"]),
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
                },
            }
        )

    insight = _insight_summary(structured_items)
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
    }
