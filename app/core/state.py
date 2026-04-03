from app.models.schemas import Chunk, IngestedPaper
from app.services.vector_db import InMemoryVectorDB


class AppState:
    def __init__(self) -> None:
        self.papers: dict[str, IngestedPaper] = {}
        self.sections: dict[str, dict[str, str]] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        self.paper_embeddings: dict[str, list[float]] = {}
        self.vdb = InMemoryVectorDB()
        self.selected_papers: set[str] = set()  # Track selected paper IDs

    def clear(self) -> None:
        self.papers.clear()
        self.sections.clear()
        self.chunks.clear()
        self.paper_embeddings.clear()
        self.vdb = InMemoryVectorDB()
        self.selected_papers.clear()


state = AppState()
