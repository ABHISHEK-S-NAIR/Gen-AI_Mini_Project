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
