import re

from app.core.state import state

BACKGROUND_CUES = (
    "previous work",
    "has been established",
    "rnn models",
    "earlier models",
    "prior models",
    "background",
    "related work",
)


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.?!])\s+", normalized)
    return [p.strip() for p in parts if p.strip()]


def _strip_noise(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\([^\)]*\d{4}[^\)]*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;,")


def _phrase(text: str, max_words: int = 16) -> str:
    cleaned = _strip_noise(text)
    words = cleaned.split()
    if not words:
        return ""
    return " ".join(words[:max_words])


def _non_background_sentence(text: str) -> str:
    for sent in _sentences(text):
        lowered = sent.lower()
        if any(cue in lowered for cue in BACKGROUND_CUES):
            continue
        return sent
    return _sentences(text)[0] if _sentences(text) else ""


def _find_metric(text: str) -> str:
    m = re.search(r"\b\d+(?:\.\d+)?\s*(?:%|BLEU|ROUGE|F1|accuracy|EM)\b", text, re.IGNORECASE)
    return m.group(0) if m else "no explicit metric"


def _infer_core_technique(text: str) -> str:
    lowered = text.lower()
    rules = [
        ("self-attention", "self-attention"),
        ("transformer", "transformer architecture"),
        ("masked language model", "MLM + NSP"),
        ("mlm", "MLM + NSP"),
        ("retrieval", "retrieval-augmented generation"),
        ("graph neural", "graph-guided learning"),
        ("lstm", "recurrent sequence modeling"),
    ]
    for key, value in rules:
        if key in lowered:
            return value
    return "task-specific neural modeling"


def extract_structured_data(paper_id: str) -> dict[str, str]:
    paper = state.papers.get(paper_id)
    sections = state.sections.get(paper_id, {})

    title = paper.filename if paper else "unknown.pdf"
    abstract = sections.get("abstract", "")
    intro = sections.get("intro", "")
    method = sections.get("method", "")
    results = sections.get("results", "")
    conclusion = sections.get("conclusion", "")

    if not any([abstract, intro, method, results, conclusion]):
        fallback = " ".join(c.text for c in state.chunks.get(paper_id, [])[:8])
        abstract = fallback
        intro = fallback
        method = fallback
        results = fallback
        conclusion = fallback

    method_src = _non_background_sentence(method) or _non_background_sentence(intro)
    problem_src = _non_background_sentence(" ".join([abstract, intro]))
    result_src = _non_background_sentence(" ".join([results, conclusion]))

    novelty_src = ""
    for sent in _sentences(" ".join([intro, conclusion, abstract])):
        l = sent.lower()
        if any(k in l for k in ("we propose", "we introduce", "novel", "first", "new architecture")):
            novelty_src = sent
            break
    if not novelty_src:
        novelty_src = method_src

    limitation_src = ""
    for sent in _sentences(" ".join([conclusion, results, intro])):
        l = sent.lower()
        if any(k in l for k in ("limitation", "future work", "fails", "challenging", "however")):
            limitation_src = sent
            break
    if not limitation_src:
        limitation_src = "limited evaluation details"

    full_text = " ".join([abstract, intro, method, results, conclusion])
    core_technique = _infer_core_technique(full_text)
    metric = _find_metric(full_text)

    architecture = "transformer-based" if "transformer" in full_text.lower() else "non-transformer neural"
    learning_strategy = "pretraining" if "pre-train" in full_text.lower() or "pretrain" in full_text.lower() else "task-specific training"
    contribution_type = "architectural" if "architecture" in full_text.lower() else "methodological"

    return {
        "paper_id": paper_id,
        "title": title,
        "problem": _phrase(problem_src) or "problem not clearly extracted",
        "proposed_method": _phrase(method_src) or "method not clearly extracted",
        "core_technique": core_technique,
        "results": _phrase(result_src) or "results not clearly extracted",
        "novelty": _phrase(novelty_src) or "novelty not clearly extracted",
        "limitations": _phrase(limitation_src) or "limitations not clearly extracted",
        "metric": metric,
        "architecture": architecture,
        "learning_strategy": learning_strategy,
        "contribution_type": contribution_type,
    }


def extract_structured_for_papers(paper_ids: list[str]) -> list[dict[str, str]]:
    return [extract_structured_data(pid) for pid in paper_ids]
