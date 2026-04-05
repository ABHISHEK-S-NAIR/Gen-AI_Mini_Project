"""
Embedding engine using SciBERT for scientific text.
Primary model: allenai/scibert_scivocab_uncased (768-dim)
Fallback model: all-MiniLM-L6-v2 (384-dim) — used automatically if SciBERT fails to load.
"""
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_PRIMARY_MODEL_NAME = "allenai/scibert_scivocab_uncased"
_FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_active_model_name: str = ""
_active_dim: int = 768


def _get_model() -> SentenceTransformer:
    global _model, _active_model_name, _active_dim
    if _model is not None:
        return _model

    # Try primary (SciBERT)
    try:
        logger.info(f"Loading embedding model: {_PRIMARY_MODEL_NAME}")
        _model = SentenceTransformer(_PRIMARY_MODEL_NAME)
        _active_model_name = _PRIMARY_MODEL_NAME
        _active_dim = 768
        logger.info(f"Loaded {_PRIMARY_MODEL_NAME} successfully (dim={_active_dim})")
        return _model
    except Exception as e:
        logger.warning(f"Failed to load {_PRIMARY_MODEL_NAME}: {e}. Falling back to {_FALLBACK_MODEL_NAME}.")

    # Fallback (MiniLM)
    try:
        _model = SentenceTransformer(_FALLBACK_MODEL_NAME)
        _active_model_name = _FALLBACK_MODEL_NAME
        _active_dim = 384
        logger.info(f"Loaded fallback model {_FALLBACK_MODEL_NAME} (dim={_active_dim})")
        return _model
    except Exception as e:
        raise RuntimeError(
            f"Could not load any embedding model. "
            f"Tried {_PRIMARY_MODEL_NAME} and {_FALLBACK_MODEL_NAME}. Last error: {e}"
        )


def get_embedding_dim() -> int:
    """Returns the actual embedding dimension of the loaded model."""
    _get_model()  # ensure model is loaded
    return _active_dim


def embed_texts(texts: list[str], dim: int) -> list[list[float]]:
    """
    Embed a list of texts using the loaded SentenceTransformer model.
    The `dim` parameter is accepted for API compatibility but ignored —
    actual dimensionality is determined by the loaded model.
    Empty or whitespace-only strings are returned as zero vectors.
    """
    if not texts:
        return []

    model = _get_model()

    # Handle empty strings — encode them separately as zero vectors
    results: list[list[float]] = []
    non_empty_indices: list[int] = []
    non_empty_texts: list[str] = []

    for i, text in enumerate(texts):
        if not text.strip():
            results.append([0.0] * _active_dim)
        else:
            results.append([])  # placeholder
            non_empty_indices.append(i)
            non_empty_texts.append(text)

    if non_empty_texts:
        embeddings = model.encode(
            non_empty_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        for idx, embedding in zip(non_empty_indices, embeddings):
            results[idx] = embedding.tolist()

    return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
