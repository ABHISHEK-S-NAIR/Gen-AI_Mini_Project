from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings, update_settings
from app.core.errors import ERRORS
from app.core.state import state
from app.models.schemas import (
    AnalysisRequest, AnalysisResponse,
    CitationRequest, CitationResponse,
    ConfigResponse, ConfigUpdateRequest,
    ExplanationRequest, ExplanationResponse,
    QARequest, QAResponse,
    ReviewRequest, ReviewResponse,
    TaskRequest, TaskResponse,
)
from app.services.doc_selection_agent import select_documents
from app.services.input_handler import ingest_files
from app.services.output_formatter import format_task_output
from app.services.task_router import route_task

app = FastAPI(title="PaperMind MVP", version="0.1.0")
app.mount("/web", StaticFiles(directory="app/web"), name="web")


@app.get("/")
def ui_root() -> FileResponse:
    return FileResponse("app/web/index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "papers": len(state.papers),
        "chunks": sum(len(v) for v in state.chunks.values()),
        "config": settings.model_dump(),
    }


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail=ERRORS["E001"].__dict__)

    result = await ingest_files(files)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/task", response_model=TaskResponse)
def run_task(req: TaskRequest) -> TaskResponse:
    """
    Legacy unified task endpoint. 
    Prefer using dedicated endpoints: /api/qa, /api/citations, etc.
    """
    if not state.papers:
        raise HTTPException(status_code=400, detail={"code": "E008", "message": "NO_PAPERS_INGESTED"})

    if req.task == "ask" and not (req.question or req.query):
        raise HTTPException(status_code=400, detail={"code": "E011", "message": "QUESTION_REQUIRED_FOR_ASK"})

    selected = req.paper_ids or []
    if not selected:
        # If no paper_ids specified in request, use selected papers from UI
        if state.selected_papers:
            selected = list(state.selected_papers)
        else:
            # Fall back to document selection agent
            intent_text = req.question or req.query or req.task
            dsa = select_documents(intent_text, state.paper_embeddings)
            selected = dsa.selected_papers
            if not selected:
                raise HTTPException(status_code=404, detail=ERRORS["E004"].__dict__)

    result = route_task(
        task=req.task,
        selected_papers=selected,
        question=(req.question or req.query) if req.task == "ask" else None,
        level=req.level if req.task == "explain" else None,
        sections=req.sections,
    )

    formatted = format_task_output(req.task, selected, result)
    return TaskResponse(task=req.task, selected_papers=selected, result=formatted)


def _get_selected_papers(paper_ids: list[str] | None) -> list[str]:
    """Helper to resolve paper selection with fallback to state."""
    if paper_ids:
        # Validate provided paper IDs
        invalid_ids = [pid for pid in paper_ids if pid not in state.papers]
        if invalid_ids:
            raise HTTPException(
                status_code=404,
                detail={"code": "E012", "message": f"Invalid paper IDs: {invalid_ids}"}
            )
        return paper_ids
    
    # Use selected papers from state
    if state.selected_papers:
        return list(state.selected_papers)
    
    raise HTTPException(
        status_code=400,
        detail={"code": "E013", "message": "No papers selected. Provide paper_ids or select papers first."}
    )


@app.post("/api/qa", response_model=QAResponse)
def answer_question(req: QARequest) -> QAResponse:
    """
    Dedicated Q&A endpoint with specific validation.
    Answer questions about research papers using RAG.
    """
    if not state.papers:
        raise HTTPException(status_code=400, detail={"code": "E008", "message": "NO_PAPERS_INGESTED"})
    
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail={"code": "E011", "message": "QUESTION_REQUIRED"})
    
    selected = _get_selected_papers(req.paper_ids)
    
    from app.services.qa_service import answer_question_with_sections
    result = answer_question_with_sections(
        req.question,
        selected,
        req.sections,
        conversation_id=req.conversation_id,
        debug=req.debug,
    )
    
    return QAResponse(
        question=result["question"],
        answer=result["answer"],
        context=result["context"],
        grounded=result["grounded"],
        confidence=result.get("confidence"),
        avg_relevance=result.get("avg_relevance"),
        selected_papers=selected,
        cited_excerpts=result.get("cited_excerpts", []),
        cited_chunks=result.get("cited_chunks", []),
    )


@app.delete("/api/qa/conversation/{conversation_id}")
def clear_qa_conversation(conversation_id: str) -> dict[str, bool]:
    state.clear_conversation(conversation_id)
    return {"cleared": True}


@app.post("/api/citations", response_model=CitationResponse)
def analyze_citations(req: CitationRequest) -> CitationResponse:
    """
    Dedicated citation analysis endpoint.
    Extract and analyze citations from research papers.
    """
    if not state.papers:
        raise HTTPException(status_code=400, detail={"code": "E008", "message": "NO_PAPERS_INGESTED"})
    
    selected = _get_selected_papers(req.paper_ids)
    
    from app.services.citation_service import analyse_citations_for_papers
    result = analyse_citations_for_papers(selected)
    
    return CitationResponse(
        citations=result.get("all_citations", []),
        selected_papers=selected,
    )


@app.post("/api/analysis", response_model=AnalysisResponse)
def analyze_papers(req: AnalysisRequest) -> AnalysisResponse:
    """
    Dedicated analysis endpoint.
    Generate comprehensive analysis of research papers.
    """
    if not state.papers:
        raise HTTPException(status_code=400, detail={"code": "E008", "message": "NO_PAPERS_INGESTED"})
    
    selected = _get_selected_papers(req.paper_ids)
    
    from app.services.analysis_service import analyse
    result = analyse(selected)
    
    return AnalysisResponse(
        analysis=result,
        selected_papers=selected,
    )


@app.post("/api/review", response_model=ReviewResponse)
def review_papers(req: ReviewRequest) -> ReviewResponse:
    """
    Dedicated peer review endpoint.
    Generate peer review feedback for research papers.
    """
    if not state.papers:
        raise HTTPException(status_code=400, detail={"code": "E008", "message": "NO_PAPERS_INGESTED"})
    
    selected = _get_selected_papers(req.paper_ids)
    
    from app.services.review_service import review
    result = review(selected)
    
    return ReviewResponse(
        review=result,
        selected_papers=selected,
    )


@app.post("/api/explain", response_model=ExplanationResponse)
def explain_papers(req: ExplanationRequest) -> ExplanationResponse:
    """
    Dedicated explanation endpoint.
    Generate explanations of research papers at different expertise levels.
    """
    if not state.papers:
        raise HTTPException(status_code=400, detail={"code": "E008", "message": "NO_PAPERS_INGESTED"})
    
    selected = _get_selected_papers(req.paper_ids)
    
    from app.services.explanation_service import explain
    result = explain(selected, req.level)
    
    return ExplanationResponse(
        explanation=result,
        selected_papers=selected,
        level=req.level,
    )


@app.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(**settings.model_dump())


@app.patch("/config", response_model=ConfigResponse)
def patch_config(req: ConfigUpdateRequest) -> ConfigResponse:
    if state.papers:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "E009",
                "message": "CONFIG_LOCKED_AFTER_INGESTION",
            },
        )

    payload = req.model_dump(exclude_none=True)
    try:
        updated = update_settings(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "E010", "message": str(exc)}) from exc

    return ConfigResponse(**updated.model_dump())


@app.get("/papers")
def list_papers() -> list[dict[str, object]]:
    return [
        {
            "paper_id": p.paper_id,
            "filename": p.filename,
            "selected": p.paper_id in state.selected_papers,
        }
        for p in state.papers.values()
    ]


@app.post("/papers/select")
def update_paper_selection(req: dict[str, list[str]]) -> dict[str, object]:
    """Update the selection state of papers.
    
    Args:
        req: Dictionary with 'paper_ids' list of paper IDs to select
    
    Returns:
        Dictionary with updated selection state
    """
    paper_ids = req.get("paper_ids", [])
    
    # Validate all paper IDs exist
    invalid_ids = [pid for pid in paper_ids if pid not in state.papers]
    if invalid_ids:
        raise HTTPException(
            status_code=404,
            detail={"code": "E012", "message": f"Invalid paper IDs: {invalid_ids}"}
        )
    
    # Update selected papers
    state.set_selected_papers(paper_ids)
    
    return {
        "selected_count": len(state.selected_papers),
        "selected_papers": list(state.selected_papers),
    }
