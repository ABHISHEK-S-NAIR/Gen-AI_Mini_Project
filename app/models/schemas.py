from typing import Literal

from pydantic import BaseModel, Field


SectionName = Literal["abstract", "intro", "method", "results", "conclusion", "other"]
TaskType = Literal["analyse", "review", "citations", "ask", "explain"]
ExplainLevel = Literal["beginner", "intermediate", "expert", "visual", "training", "pipeline", "components"]


class IngestedPaper(BaseModel):
    paper_id: str
    filename: str
    raw_text: str


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    section: SectionName
    chunk_index: int
    text: str
    embedding: list[float] = Field(default_factory=list)


class DSAResult(BaseModel):
    selected_papers: list[str]
    topic_groups: list[dict[str, object]]


class TaskRequest(BaseModel):
    task: TaskType
    query: str | None = None
    question: str | None = None
    level: ExplainLevel | None = None
    paper_ids: list[str] | None = None
    sections: list[SectionName] | None = None


class TaskResponse(BaseModel):
    task: TaskType
    selected_papers: list[str]
    result: dict[str, object]


class QAResult(BaseModel):
    question: str
    answer: str
    context: list[Chunk]
    grounded: bool


class ApiError(BaseModel):
    code: str
    message: str


class ConfigUpdateRequest(BaseModel):
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    top_k_chunks: int | None = None
    top_n_papers: int | None = None
    similarity_threshold: float | None = None


class ConfigResponse(BaseModel):
    chunk_size: int
    chunk_overlap: int
    top_k_chunks: int
    top_n_papers: int
    similarity_threshold: float
    embedding_dim: int


# Dedicated request/response models for split endpoints
class QARequest(BaseModel):
    question: str
    paper_ids: list[str] | None = None
    sections: list[SectionName] | None = None
    conversation_id: str | None = None
    debug: bool = False


class QAResponse(BaseModel):
    question: str
    answer: str
    context: list[dict]
    grounded: bool
    confidence: float | None = None
    avg_relevance: float | None = None
    selected_papers: list[str]
    cited_excerpts: list[int] = Field(default_factory=list)
    cited_chunks: list[dict] = Field(default_factory=list)


class CitationRequest(BaseModel):
    paper_ids: list[str] | None = None


class CitationResponse(BaseModel):
    citations: list[dict]
    selected_papers: list[str]


class AnalysisRequest(BaseModel):
    paper_ids: list[str] | None = None


class AnalysisResponse(BaseModel):
    analysis: dict[str, object]
    selected_papers: list[str]


class ReviewRequest(BaseModel):
    paper_ids: list[str] | None = None


class ReviewResponse(BaseModel):
    review: dict[str, object]
    selected_papers: list[str]


class ExplanationRequest(BaseModel):
    paper_ids: list[str] | None = None
    level: ExplainLevel = "intermediate"


class ExplanationResponse(BaseModel):
    explanation: dict[str, object]
    selected_papers: list[str]
    level: ExplainLevel
