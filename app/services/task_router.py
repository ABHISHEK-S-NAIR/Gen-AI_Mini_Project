from app.services.analysis_service import analyse
from app.services.citation_service import analyse_citations_for_papers
from app.services.explanation_service import explain
from app.services.qa_service import answer_question_with_sections
from app.services.review_service import review


def route_task(
    task: str,
    selected_papers: list[str],
    question: str | None,
    level: str | None,
    sections: list[str] | None,
) -> dict[str, object]:
    if task == "analyse":
        return analyse(selected_papers)
    if task == "review":
        return review(selected_papers)
    if task == "citations":
        if not selected_papers:
            return {"citations": []}
        return analyse_citations_for_papers(selected_papers)
    if task == "ask":
        return answer_question_with_sections(question or "", selected_papers, sections)
    if task == "explain":
        if not selected_papers:
            return {"explanations": [], "level": level or "intermediate"}
        return explain(selected_papers, level or "intermediate")

    return {"error": {"code": "E000", "message": "UNKNOWN_TASK"}}
