from collections import defaultdict

from app.config import settings
from app.models.schemas import DSAResult
from app.services.embedding_engine import cosine_similarity, embed_texts


def select_documents(query: str, paper_embeddings: dict[str, list[float]]) -> DSAResult:
    if not query.strip() or not paper_embeddings:
        return DSAResult(selected_papers=[], topic_groups=[])

    query_vec = embed_texts([query], settings.embedding_dim)[0]
    scored: list[tuple[str, float]] = []
    for paper_id, emb in paper_embeddings.items():
        scored.append((paper_id, cosine_similarity(query_vec, emb)))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [pid for pid, s in scored if s >= settings.similarity_threshold][: settings.top_n_papers]
    if not selected:
        selected = [pid for pid, _ in scored[: settings.top_n_papers]]

    groups: dict[str, list[str]] = defaultdict(list)
    for pid in selected:
        groups[pid[:1] or "g"].append(pid)

    topic_groups = [{"label": f"topic_{k}", "paper_ids": v} for k, v in groups.items()]
    return DSAResult(selected_papers=selected, topic_groups=topic_groups)
