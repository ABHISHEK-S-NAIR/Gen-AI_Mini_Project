from app.services.structured_extraction_service import extract_structured_for_papers


def _diagram_for(technique: str) -> str:
    t = technique.lower()
    if "self-attention" in t or "transformer" in t:
        return "[Input] --> [Embedding] --> [Self-Attention Blocks] --> [Task Head] --> [Output]"
    if "retrieval" in t:
        return "[Input/Query] --> [Retriever] --> [Context Selection] --> [Generator] --> [Grounded Output]"
    if "mlm" in t:
        return "[Input Tokens] --> [Mask Tokens] --> [Bidirectional Encoder] --> [MLM/NSP Objectives] --> [Fine-tuned Model]"
    return "[Input] --> [Encoder/Backbone] --> [Learning Mechanism] --> [Prediction]"


def explain(paper_id: str, level: str) -> dict[str, object]:
    items = extract_structured_for_papers([paper_id])
    if not items:
        return {"paper_id": paper_id, "level": level, "explanation": "No data available.", "diagram": None}

    item = items[0]

    beginner = (
        f"This paper tries to solve {item['problem']}. "
        f"It does this by using {item['core_technique']} in a simple pipeline. "
        f"The main result says {item['results']}."
    )

    intermediate = (
        f"Problem statement: {item['problem']}\n"
        f"Proposed method: {item['proposed_method']}\n"
        f"Technical mechanism: {item['core_technique']} ({item['learning_strategy']})\n"
        f"Result and metric: {item['results']} ({item['metric']})\n"
        f"Novel element: {item['novelty']}"
    )

    expert = (
        f"Technical interpretation: the architectural design follows {item['architecture']} and operationalizes "
        f"{item['core_technique']} as the primary training and optimization lever. The contribution type is "
        f"{item['contribution_type']} with explicit novelty claim: {item['novelty']}. Empirical signal is "
        f"{item['results']} measured by {item['metric']}. Known constraints are {item['limitations']}."
    )

    if level == "beginner":
        return {"paper_id": paper_id, "paper_name": item["title"], "level": "beginner", "explanation": beginner, "diagram": None}
    if level == "intermediate":
        return {
            "paper_id": paper_id,
            "paper_name": item["title"],
            "level": "intermediate",
            "explanation": intermediate,
            "diagram": None,
        }
    if level == "expert":
        return {"paper_id": paper_id, "paper_name": item["title"], "level": "expert", "explanation": expert, "diagram": None}

    return {
        "paper_id": paper_id,
        "paper_name": item["title"],
        "level": "visual",
        "explanation": f"Visual flow for {item['title']} using {item['core_technique']}.",
        "diagram": _diagram_for(item["core_technique"]),
    }
