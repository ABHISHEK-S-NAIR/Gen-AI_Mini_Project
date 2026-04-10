import re

from app.core.errors import ERRORS
from app.core.state import state


# Comprehensive citation patterns covering common academic formats
PATTERNS = [
    # Author-year with parentheses: (Smith, 2020) or (Smith et al., 2020)
    re.compile(r"\([A-Z][a-z]+(?:\s+et al\.?)?,\s*\d{4}[a-z]?\)"),
    
    # Narrative citation: Smith et al. (2020) or Jones (2021)
    re.compile(r"\b[A-Z][a-z]+(?:\s+et al\.?)?\s+\(\d{4}[a-z]?\)"),
    
    # Multiple citations: (Smith 2020; Jones 2021) or (Smith, 2020; Jones, 2021)
    re.compile(r"\([A-Z][a-z]+(?:\s+et al\.?)?,?\s+\d{4}(?:\s*;\s*[A-Z][a-z]+(?:\s+et al\.?)?,?\s+\d{4})+\)"),
    
    # Author-year without parentheses: Smith et al 2020 or Smith and Jones 2020
    re.compile(r"\b[A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+|\s+et al\.?)?(?:\s+\()?(\d{4}[a-z]?)(?:\))?"),
    
    # Numeric citations: [12] or [1, 2, 3] or [5-8]
    re.compile(r"\[(\d+(?:\s*,\s*\d+|-\d+)*)\]"),
    
    # Superscript-style (captured as regular text): ^1,2,3^
    re.compile(r"\^(\d+(?:,\s*\d+)*)\^"),
]

# Legacy patterns kept for backward compatibility
AUTHOR_YEAR_PATTERN = PATTERNS[0]
NUMERIC_PATTERN = PATTERNS[4]

SENTENCE_PATTERN = re.compile(r"(?<=[.?!])\s+(?=(?:[A-Z]|\d+\s+[A-Z]))")

SUPPORTING_KEYWORDS = (
    "consistent with",
    "in line with",
    "builds on",
    "based on",
    "as shown by",
    "similar to",
    "confirm",
    "supports",
    "supported by",
)

CONTRASTING_KEYWORDS = (
    "in contrast",
    "contrary to",
    "unlike",
    "whereas",
    "differs from",
    "differ from",  # Added verb form
    "fails to",
    "fail to",  # Added plural form
    "failed to",
    "limited by",
    "as opposed to",
    "outperforms",
    "outperform",
    "superior to",
    "better than",
    "advantages over",
    "advantage over",
)

# Weak signals that need reinforcement from other keywords
WEAK_CONTRAST_WORDS = ("however", "but")

ABBREVIATION_MAP = {
    "et al.": "et al<prd>",
    "e.g.": "e<prd>g<prd>",
    "i.e.": "i<prd>e<prd>",
    "fig.": "fig<prd>",
    "eq.": "eq<prd>",
}

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
ARXIV_PATTERN = re.compile(r"arXiv:\S+", re.IGNORECASE)
SECTION_HEADING_INLINE_PATTERN = re.compile(r"\s\d+(?:\.\d+)*\s+[A-Z][A-Za-z-]{2,}")


def _default_canonical_key(raw_text: str) -> str:
    lowered = (raw_text or "").lower().strip()
    cleaned = re.sub(r"[^a-z0-9]+", "", lowered)
    return cleaned or lowered or "unknown"


def _normalize_text(text: str) -> str:
    # Common PDF artifacts cleanup: soft hyphenation, ligatures, and line breaks.
    text = text.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    # If punctuation is immediately followed by a heading token, insert a split space.
    text = re.sub(r"([.?!])(?=(?:[A-Z]|\d+\s+[A-Z]))", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _protect_abbreviations(text: str) -> str:
    out = text
    for src, dst in ABBREVIATION_MAP.items():
        out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    return out


def _restore_abbreviations(text: str) -> str:
    return text.replace("<prd>", ".")


def _compact_sentence(sentence: str, max_len: int = 420) -> str:
    s = re.sub(r"\s+", " ", sentence).strip()

    # Remove common metadata noise from extracted PDF text.
    s = EMAIL_PATTERN.sub("", s)
    s = URL_PATTERN.sub("", s)
    s = ARXIV_PATTERN.sub("", s)

    # Remove trailing inline section heading if sentence got merged (e.g., '. 2 Background').
    heading_match = SECTION_HEADING_INLINE_PATTERN.search(s)
    if heading_match and heading_match.start() > 40:
        s = s[: heading_match.start()].strip()

    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_len:
        return s
    return s[:max_len].rstrip() + "..."


def _split_sentences(text: str) -> list[str]:
    protected = _protect_abbreviations(text)
    parts = re.split(SENTENCE_PATTERN, protected)
    return [_restore_abbreviations(p.strip()) for p in parts if p.strip()]


def _extract_citation_strings(sentence: str) -> list[str]:
    """
    Extract all citation strings from a sentence using comprehensive patterns.
    Handles author-year, narrative, and numeric citation formats.
    """
    citations: list[str] = []
    seen = set()
    
    for pattern in PATTERNS:
        for match in pattern.finditer(sentence):
            citation = match.group(0).strip()
            if citation not in seen:
                citations.append(citation)
                seen.add(citation)
    
    return citations


def _classify_sentence(sentence: str) -> str:
    """
    Classify citation function in a sentence.
    
    Handles complex cases like:
    - Negation after supporting keywords ("similar to X, but fails...")
    - Weak contrast words (but/however) that need reinforcement
    - Multiple competing signals
    
    Priority: contrasting > supporting > neutral
    """
    lowered = sentence.lower()
    
    # Check for strong contrasting signals
    has_strong_contrast = any(k in lowered for k in CONTRASTING_KEYWORDS)
    
    # Check for weak contrast signals (but/however)
    has_weak_contrast = any(k in lowered for k in WEAK_CONTRAST_WORDS)
    
    # Check for supporting signals
    has_support = any(k in lowered for k in SUPPORTING_KEYWORDS)
    
    # Weak contrast words only count if paired with support/contrast keywords
    # E.g., "similar to X, but fails" = contrasting
    # E.g., "However, we use X" = neutral (just transition word)
    if has_weak_contrast and (has_support or has_strong_contrast):
        # Check if weak contrast comes after support (indicates negation)
        support_positions = [lowered.find(k) for k in SUPPORTING_KEYWORDS if k in lowered]
        weak_positions = [lowered.find(k) for k in WEAK_CONTRAST_WORDS if k in lowered]
        
        if support_positions and weak_positions:
            earliest_support = min(support_positions)
            earliest_weak = min(weak_positions)
            
            # "similar to X, but fails" - weak contrast after support
            if earliest_weak > earliest_support:
                return "contrasting"
    
    # Strong contrast always wins
    if has_strong_contrast:
        return "contrasting"
    
    # Supporting without negation
    if has_support:
        return "supporting"
    
    return "neutral"


def _build_insight(citation_text: str, citation_type: str) -> str:
    if citation_type == "supporting":
        return f"{citation_text} is used to reinforce the current claim with prior evidence."
    if citation_type == "contrasting":
        return f"{citation_text} is used to contrast the current claim against prior work or assumptions."
    return f"{citation_text} is referenced as related background without strong support/contrast language."


def analyse_citations(paper_id: str) -> dict[str, object]:
    paper = state.papers.get(paper_id)
    if not paper:
        return {"citations": [], "error": ERRORS["E007"].__dict__}

    text = _normalize_text(paper.raw_text)
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for sentence in _split_sentences(text):
        raw_cites = _extract_citation_strings(sentence)
        if not raw_cites:
            continue

        citation_type = _classify_sentence(sentence)
        context = _compact_sentence(sentence)
        for raw_cite in raw_cites:
            key = (raw_cite, context)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "raw_text": raw_cite,
                    "context": context,
                    "type": citation_type,
                    "insight": _build_insight(raw_cite, citation_type),
                }
            )

    if not citations:
        return {"citations": [], "error": ERRORS["E007"].__dict__}

    citations = deduplicate_citations(citations)
    citations = enrich_citations_with_llm(citations)
    return {"citations": citations}


def deduplicate_citations(citations: list[dict]) -> list[dict]:
    """
    Embeds each citation's raw_text and clusters citations whose cosine
    similarity > 0.9 as the same work. Within each cluster, keeps the
    citation with the longest context string as the canonical one, and
    adds a "duplicate_of" key (pointing to the canonical raw_text) on
    all others. Citations with no near-neighbour get duplicate_of=None.

    Works on the full list regardless of size.
    Falls back to returning the original list unchanged if embedding fails.
    """
    if len(citations) < 2:
        return [{**dict(c), "duplicate_of": c.get("duplicate_of", None)} for c in citations]

    try:
        from app.config import settings
        from app.services.embedding_engine import cosine_similarity, embed_texts

        texts = [c.get("raw_text", "") for c in citations]
        vectors = embed_texts(texts, settings.embedding_dim)

        parent = list(range(len(citations)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                if cosine_similarity(vectors[i], vectors[j]) > 0.9:
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(len(citations)):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        result = [dict(c) for c in citations]
        for members in clusters.values():
            if len(members) == 1:
                result[members[0]]["duplicate_of"] = None
                continue
            canonical_idx = max(members, key=lambda i: len(citations[i].get("context", "")))
            canonical_raw = citations[canonical_idx].get("raw_text", "")
            for idx in members:
                if idx == canonical_idx:
                    result[idx]["duplicate_of"] = None
                else:
                    result[idx]["duplicate_of"] = canonical_raw

        return result

    except Exception:
        return [{**dict(c), "duplicate_of": c.get("duplicate_of", None)} for c in citations]


def enrich_citations_with_llm(citations: list[dict]) -> list[dict]:
    """
    Takes the top 20 extracted citations and uses LLM to enrich each with:
      - claim: what specific claim this citation supports or challenges
      - relationship: one of "foundational" | "incremental" | "contradicting"
      - canonical_key: a normalised author-year string used for deduplication
        e.g. "(Vaswani et al., 2017)" and "[1]" both map to "vaswani2017"
        If not determinable, use the raw_text lowercased stripped of punctuation.

    Returns the same list with three new keys added to each dict.
    Falls back gracefully — if LLM call fails or JSON parse fails,
    return the original list with claim="", relationship="neutral",
    canonical_key=raw_text.lower() for each item.
    """
    if not citations:
        return citations

    def with_defaults(items: list[dict]) -> list[dict]:
        out = []
        for c in items:
            base = dict(c)
            base.setdefault("claim", "")
            base.setdefault("relationship", base.get("type", "neutral"))
            base.setdefault("canonical_key", _default_canonical_key(base.get("raw_text", "")))
            out.append(base)
        return out

    top = citations[:20]

    numbered = "\n".join(
        f"{i+1}. raw_text={c.get('raw_text', '')} | context={str(c.get('context', ''))[:120]}"
        for i, c in enumerate(top)
    )

    prompt = (
        "You are analyzing citations extracted from a research paper.\n\n"
        "For each numbered citation below, return a JSON array where each element has:\n"
        '  "index": (1-based integer matching the input number),\n'
        '  "claim": (one sentence - the specific claim this citation supports or challenges,\n'
        '            inferred from its context. Empty string if unclear.),\n'
        '  "relationship": (exactly one of: "foundational", "incremental", "contradicting"),\n'
        '  "canonical_key": (a short normalised key like "vaswani2017" or "smith2020b"\n'
        '                    for deduplication. Use lastname+year, lowercase, no spaces.\n'
        '                    If numeric like [1], use "ref1". If unknown, use "unknown".)\n\n'
        "Return ONLY the JSON array. No markdown fences. No extra text.\n\n"
        f"Citations:\n{numbered}"
    )

    system = (
        "You are a precise academic citation analyst. "
        "Return only valid JSON arrays, no commentary."
    )

    try:
        from app.services.llm_client import call_llm_json

        enrichments = call_llm_json(prompt, system=system, max_tokens=1200)
        if not isinstance(enrichments, list):
            raise ValueError("Expected a JSON array")

        enrichment_map = {e["index"]: e for e in enrichments if isinstance(e, dict) and "index" in e}

        enriched = []
        for i, c in enumerate(citations):
            base = dict(c)
            if i < 20 and (i + 1) in enrichment_map:
                e = enrichment_map[i + 1]
                rel = e.get("relationship", base.get("type", "neutral"))
                if rel not in ("foundational", "incremental", "contradicting", "neutral"):
                    rel = base.get("type", "neutral")
                base["claim"] = e.get("claim", "")
                base["relationship"] = rel
                base["canonical_key"] = e.get("canonical_key", _default_canonical_key(base.get("raw_text", "")))
            else:
                base.setdefault("claim", "")
                base.setdefault("relationship", base.get("type", "neutral"))
                base.setdefault("canonical_key", _default_canonical_key(base.get("raw_text", "")))
            enriched.append(base)

        return enriched

    except Exception:
        fallback = []
        for c in citations:
            base = dict(c)
            base.setdefault("claim", "")
            base.setdefault("relationship", "neutral")
            base.setdefault("canonical_key", _default_canonical_key(base.get("raw_text", "")))
            fallback.append(base)
        return fallback


def analyse_citations_for_papers(paper_ids: list[str]) -> dict[str, object]:
    papers_output: list[dict[str, object]] = []
    total = 0

    for paper_id in paper_ids:
        paper = state.papers.get(paper_id)
        paper_name = paper.filename if paper else "unknown.pdf"
        single = analyse_citations(paper_id)
        citations = single.get("citations", [])
        total += len(citations)

        papers_output.append(
            {
                "paper_id": paper_id,
                "paper_name": paper_name,
                "citations": citations,
                "error": single.get("error"),
            }
        )

    return {
        "papers": papers_output,
        "all_citations": [c for p in papers_output for c in p.get("citations", [])],
        "total_citations": total,
    }
