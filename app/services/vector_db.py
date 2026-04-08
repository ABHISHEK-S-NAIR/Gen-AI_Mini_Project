from app.services.embedding_engine import cosine_similarity


class InMemoryVectorDB:
    def __init__(self) -> None:
        self._rows: list[dict[str, object]] = []

    def upsert(self, rows: list[dict[str, object]]) -> None:
        self._rows.extend(rows)

    def search(
        self,
        query_vector: list[float],
        k: int,
        paper_ids: list[str] | None = None,
        sections: list[str] | None = None,
    ) -> list[dict[str, object]]:
        filtered = self._rows
        if paper_ids:
            filtered = [r for r in filtered if r["paper_id"] in paper_ids]
        if sections:
            filtered = [r for r in filtered if r["section"] in sections]

        scored = []
        for row in filtered:
            score = cosine_similarity(query_vector, row["embedding"])
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Include similarity score in each returned row
        results = []
        for score, row in scored[:k]:
            row_with_score = dict(row)  # Create a copy to avoid mutating original
            row_with_score["score"] = score
            results.append(row_with_score)
        return results

    def all_rows(self) -> list[dict[str, object]]:
        return list(self._rows)
