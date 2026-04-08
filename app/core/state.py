import threading
from typing import Optional

from app.models.schemas import Chunk, IngestedPaper
from app.services.vector_db import InMemoryVectorDB


class AppState:
    """Thread-safe application state using RLock for concurrent access."""
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._papers: dict[str, IngestedPaper] = {}
        self._sections: dict[str, dict[str, str]] = {}
        self._chunks: dict[str, list[Chunk]] = {}
        self._paper_embeddings: dict[str, list[float]] = {}
        self._vdb = InMemoryVectorDB()
        self._selected_papers: set[str] = set()

    @property
    def papers(self) -> dict[str, IngestedPaper]:
        """Thread-safe access to papers dictionary."""
        with self._lock:
            return self._papers.copy()
    
    def get_paper(self, paper_id: str) -> Optional[IngestedPaper]:
        """Thread-safe get single paper."""
        with self._lock:
            return self._papers.get(paper_id)
    
    def add_paper(self, paper_id: str, paper: IngestedPaper) -> None:
        """Thread-safe add paper."""
        with self._lock:
            self._papers[paper_id] = paper
    
    def remove_paper(self, paper_id: str) -> None:
        """Thread-safe remove paper."""
        with self._lock:
            self._papers.pop(paper_id, None)

    @property
    def sections(self) -> dict[str, dict[str, str]]:
        """Thread-safe access to sections dictionary."""
        with self._lock:
            return self._sections.copy()
    
    def get_sections(self, paper_id: str) -> Optional[dict[str, str]]:
        """Thread-safe get sections for a paper."""
        with self._lock:
            return self._sections.get(paper_id)
    
    def add_sections(self, paper_id: str, sections: dict[str, str]) -> None:
        """Thread-safe add sections."""
        with self._lock:
            self._sections[paper_id] = sections

    @property
    def chunks(self) -> dict[str, list[Chunk]]:
        """Thread-safe access to chunks dictionary."""
        with self._lock:
            return self._chunks.copy()
    
    def get_chunks(self, paper_id: str) -> Optional[list[Chunk]]:
        """Thread-safe get chunks for a paper."""
        with self._lock:
            return self._chunks.get(paper_id)
    
    def add_chunks(self, paper_id: str, chunks: list[Chunk]) -> None:
        """Thread-safe add chunks."""
        with self._lock:
            self._chunks[paper_id] = chunks

    @property
    def paper_embeddings(self) -> dict[str, list[float]]:
        """Thread-safe access to embeddings dictionary."""
        with self._lock:
            return self._paper_embeddings.copy()
    
    def add_embedding(self, paper_id: str, embedding: list[float]) -> None:
        """Thread-safe add embedding."""
        with self._lock:
            self._paper_embeddings[paper_id] = embedding

    @property
    def vdb(self) -> InMemoryVectorDB:
        """Direct access to VectorDB (VectorDB itself should be thread-safe)."""
        return self._vdb

    @property
    def selected_papers(self) -> set[str]:
        """Thread-safe access to selected papers set."""
        with self._lock:
            return self._selected_papers.copy()
    
    def add_selected_paper(self, paper_id: str) -> None:
        """Thread-safe add to selected papers."""
        with self._lock:
            self._selected_papers.add(paper_id)
    
    def remove_selected_paper(self, paper_id: str) -> None:
        """Thread-safe remove from selected papers."""
        with self._lock:
            self._selected_papers.discard(paper_id)
    
    def toggle_paper_selection(self, paper_id: str) -> bool:
        """Thread-safe toggle paper selection. Returns new state."""
        with self._lock:
            if paper_id in self._selected_papers:
                self._selected_papers.remove(paper_id)
                return False
            else:
                self._selected_papers.add(paper_id)
                return True
    
    def set_selected_papers(self, paper_ids: list[str]) -> None:
        """Thread-safe bulk set selected papers."""
        with self._lock:
            self._selected_papers = set(paper_ids)

    def clear(self) -> None:
        """Thread-safe clear all state."""
        with self._lock:
            self._papers.clear()
            self._sections.clear()
            self._chunks.clear()
            self._paper_embeddings.clear()
            self._vdb = InMemoryVectorDB()
            self._selected_papers.clear()


state = AppState()
