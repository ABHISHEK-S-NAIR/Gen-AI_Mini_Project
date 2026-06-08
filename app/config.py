from pydantic import BaseModel


class Settings(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_chunks: int = 5
    top_n_papers: int = 3
    similarity_threshold: float = 0.70
    embedding_dim: int = 768  # Updated: matches SciBERT output dim (fallback MiniLM uses 384, auto-detected)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 12
    hybrid_alpha: float = 0.7
    enable_vision_extraction: bool = (
        True  # Enable vision-based figure extraction (set to False to disable)
    )
    max_figures_per_paper: int = 10  # Maximum number of figures to analyze per paper
    llm_provider: str = (
        "groq"  # Vision model provider (groq, openai, gemini) - Groq is free and fast
    )


settings = Settings()


def update_settings(values: dict[str, int | float | str]) -> Settings:
    if "chunk_size" in values and int(values["chunk_size"]) <= 0:
        raise ValueError("chunk_size must be > 0")
    if "chunk_overlap" in values and int(values["chunk_overlap"]) < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if "chunk_size" in values and "chunk_overlap" in values:
        if int(values["chunk_overlap"]) >= int(values["chunk_size"]):
            raise ValueError("chunk_overlap must be smaller than chunk_size")
    if "top_k_chunks" in values and int(values["top_k_chunks"]) <= 0:
        raise ValueError("top_k_chunks must be > 0")
    if "top_n_papers" in values and int(values["top_n_papers"]) <= 0:
        raise ValueError("top_n_papers must be > 0")
    if "similarity_threshold" in values:
        threshold = float(values["similarity_threshold"])
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
    if "hybrid_alpha" in values:
        alpha = float(values["hybrid_alpha"])
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("hybrid_alpha must be in [0, 1]")
    if "rerank_top_n" in values and int(values["rerank_top_n"]) < 0:
        raise ValueError("rerank_top_n must be >= 0")

    for key, value in values.items():
        setattr(settings, key, value)

    return settings
