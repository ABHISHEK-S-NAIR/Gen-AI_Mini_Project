import hashlib

import numpy as np


def _hash_vector(text: str, dim: int) -> np.ndarray:
    if not text.strip():
        return np.zeros(dim, dtype=np.float32)

    out = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        out[idx] += 1.0

    norm = np.linalg.norm(out)
    if norm > 0:
        out = out / norm
    return out


def embed_texts(texts: list[str], dim: int) -> list[list[float]]:
    return [_hash_vector(text, dim).tolist() for text in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
