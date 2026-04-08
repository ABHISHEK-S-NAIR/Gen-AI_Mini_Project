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
    "however",  # Moved from weak contrast
    "but",  # Moved from weak contrast
)

WEAK_CONTRAST_WORDS = ()  # Deprecated - moved to main CONTRASTING_KEYWORDS

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
    - Multiple competing signals
    - Weak indicators that need reinforcement
    
    Priority: contrasting > supporting > neutral
    """
    lowered = sentence.lower()
    
    # Check for strong contrasting signals first
    has_contrast = any(k in lowered for k in CONTRASTING_KEYWORDS)
    
    # Check for supporting signals
    has_support = any(k in lowered for k in SUPPORTING_KEYWORDS)
    
    # If both signals present, look for negation patterns that override support
    if has_support and has_contrast:
        # Patterns like "similar to X, but/however they fail"
        # Check if contrast word comes after support word
        support_positions = [lowered.find(k) for k in SUPPORTING_KEYWORDS if k in lowered]
        contrast_positions = [lowered.find(k) for k in CONTRASTING_KEYWORDS if k in lowered]
        
        if support_positions and contrast_positions:
            earliest_support = min(support_positions)
            earliest_contrast = min(contrast_positions)
            
            # If contrast comes after support, it's likely a negation/limitation
            if earliest_contrast > earliest_support:
                return "contrasting"
    
    # Standard classification
    if has_contrast:
        return "contrasting"
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

    return {"citations": citations}


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
        "total_citations": total,
    }
