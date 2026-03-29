from app.services.structured_extraction_service import extract_structured_for_papers


def _review_for_item(item: dict[str, str]) -> dict[str, object]:
    strengths = [
        f"Method articulation is concrete: {item['proposed_method']}",
        f"Core technical lever is identifiable: {item['core_technique']}",
        f"Result reporting includes measurable signal: {item['metric']}",
    ]

    weaknesses = [
        f"Extracted limitations indicate unresolved issues: {item['limitations']}",
        "Generalization and robustness evidence is not explicitly rich in extracted text.",
    ]

    suggestions = [
        "Provide targeted ablations that isolate the unique contribution from confounding factors.",
        "Expand failure-case and boundary-condition reporting to support deployment confidence.",
    ]

    return {
        "paper_id": item["paper_id"],
        "paper_name": item["title"],
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
    }


def _comparison(items: list[dict[str, str]]) -> str | None:
    if len(items) < 2:
        return None

    first = items[0]
    second = items[1]
    return (
        "Comparative Review: "
        f"{first['title']} is centered on {first['core_technique']} with {first['learning_strategy']}, while "
        f"{second['title']} is centered on {second['core_technique']} with {second['learning_strategy']}. "
        "The main review distinction is strength in technical framing versus breadth of evaluative evidence."
    )


def review(paper_ids: list[str]) -> dict[str, object]:
    items = extract_structured_for_papers(paper_ids)
    reviews = [_review_for_item(item) for item in items]
    return {"reviews": reviews, "comparison": _comparison(items)}
