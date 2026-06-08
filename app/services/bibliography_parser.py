import json
import logging
import re
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\barxiv:\s*\S+", re.IGNORECASE)
_CROSSREF_CACHE: dict[str, str | None] = {}


def _clean_entry(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n;,")


def _parse_numeric_group(text: str) -> list[int]:
    numbers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = [p.strip() for p in part.split("-") if p.strip()]
            if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
                start, end = int(bounds[0]), int(bounds[1])
                if start <= end:
                    numbers.extend(range(start, end + 1))
                else:
                    numbers.extend(range(end, start + 1))
                continue
        if part.isdigit():
            numbers.append(int(part))
    return sorted(set(numbers))


def _resolve_author_year(raw: str, entries: list[dict]) -> dict | None:
    year_match = _YEAR_RE.search(raw)
    year = year_match.group(0) if year_match else ""
    author_match = re.search(r"\b([A-Z][a-z]+)\b", raw)
    author = author_match.group(1) if author_match else ""

    candidates = []
    for entry in entries:
        entry_year = str(entry.get("year") or "")
        authors = entry.get("authors") or []
        first_author = ""
        if authors:
            first_author = re.split(r"\s+", str(authors[0]))[0]

        if year and entry_year and year != entry_year:
            continue
        if author and first_author:
            if author.lower() != first_author.lower():
                continue
        candidates.append(entry)

    if candidates:
        return candidates[0]
    if year:
        for entry in entries:
            if str(entry.get("year") or "") == year:
                return entry
    return None


def split_entries(references_text: str) -> list[str]:
    if not references_text or not references_text.strip():
        return []

    text = references_text.replace("\r", "").strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return []

    numeric_pattern = re.compile(r"^\s*(?:\[(\d+)\]|(\d{1,3})\.)\s+\S+")
    entries: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        joined = _clean_entry(" ".join(buffer))
        if joined:
            entries.append(joined)
        buffer = []

    for line in lines:
        if not line:
            if buffer:
                buffer.append(" ")
            continue
        if numeric_pattern.match(line):
            flush()
            buffer.append(line)
            continue
        if buffer:
            buffer.append(line)
        else:
            buffer.append(line)

    flush()

    if len(entries) >= 2:
        return entries

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) >= 2:
        return [_clean_entry(b) for b in blocks]

    heur_entries: list[str] = []
    buf: list[str] = []
    author_start = re.compile(r"^[A-Z][A-Za-z'\-]+,?\s+[A-Z]")
    for line in lines:
        if not line:
            continue
        if buf and author_start.match(line) and _YEAR_RE.search(" ".join(buf)):
            joined = _clean_entry(" ".join(buf))
            if joined:
                heur_entries.append(joined)
            buf = [line]
        else:
            buf.append(line)

    if buf:
        joined = _clean_entry(" ".join(buf))
        if joined:
            heur_entries.append(joined)

    return heur_entries if heur_entries else [_clean_entry(text)]


def parse_entry(entry_text: str) -> dict:
    raw = _clean_entry(entry_text)
    if not raw:
        return {"raw": "", "title": "", "authors": [], "year": "", "venue": "", "doi": "", "arxiv": "", "key": ""}

    raw_no_marker = re.sub(r"^\s*(?:\[\d+\]|\d{1,3}\.)\s+", "", raw).strip()

    year_match = _YEAR_RE.search(raw_no_marker)
    year = year_match.group(0) if year_match else ""

    doi_match = _DOI_RE.search(raw_no_marker)
    doi = doi_match.group(0).rstrip(".;,") if doi_match else ""

    arxiv_match = _ARXIV_RE.search(raw_no_marker)
    arxiv = arxiv_match.group(0) if arxiv_match else ""

    authors_part = ""
    rest = raw_no_marker
    if year_match:
        split_idx = raw_no_marker.find(year)
        if split_idx > 0:
            authors_part = raw_no_marker[:split_idx].strip(" .,;")
            rest = raw_no_marker[split_idx + len(year):].strip(" .,")
    else:
        period_idx = raw_no_marker.find(".")
        if 0 < period_idx < 160:
            authors_part = raw_no_marker[:period_idx]
            rest = raw_no_marker[period_idx + 1:].strip()

    authors = _parse_authors(authors_part)

    title = ""
    venue = ""
    if rest:
        quote_match = re.search(r"[\"“”](.+?)[\"“”]", rest)
        if quote_match:
            title = quote_match.group(1).strip()
            venue = rest.replace(quote_match.group(0), "").strip(" .,")
        else:
            sentence_split = re.split(r"\.(\s+|$)", rest, maxsplit=1)
            title = sentence_split[0].strip(" .,") if sentence_split else ""
            venue = rest[len(sentence_split[0]):].strip(" .,") if sentence_split else ""

    key = _build_key(authors, year, title)
    return {"raw": raw, "title": title, "authors": authors, "year": year, "venue": venue, "doi": doi, "arxiv": arxiv, "key": key}


def _parse_authors(authors_part: str) -> list[str]:
    if not authors_part:
        return []
    cleaned = authors_part.replace("&", "and")
    cleaned = re.sub(r"\bet al\.?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    if not cleaned:
        return []
    if ";" in cleaned:
        parts = [p.strip() for p in cleaned.split(";") if p.strip()]
    elif " and " in cleaned:
        parts = [p.strip() for p in cleaned.split(" and ") if p.strip()]
    else:
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    return parts


def _build_key(authors: list[str], year: str, title: str) -> str:
    first = re.sub(r"[^A-Za-z]", "", authors[0]).lower() if authors else "unknown"
    year_key = year or "nodate"
    title_key = re.sub(r"[^a-z0-9]+", "", title.lower())[:12] if title else ""
    return f"{first}{year_key}{title_key}" if title_key else f"{first}{year_key}"


def resolve_citation(raw_cite: str, entries: list[dict]) -> dict | list[dict] | None:
    if not raw_cite or not entries:
        return None

    raw = raw_cite.strip()
    numeric_match = re.match(r"^\[(.+)\]$", raw)
    if numeric_match:
        numbers = _parse_numeric_group(numeric_match.group(1))
        resolved = [entries[n - 1] for n in numbers if 1 <= n <= len(entries)]
        if not resolved:
            return None
        return resolved[0] if len(resolved) == 1 else resolved

    if ";" in raw and _YEAR_RE.search(raw):
        parts = [p.strip() for p in raw.strip("() ").split(";") if p.strip()]
        resolved_parts = [_resolve_author_year(part, entries) for part in parts]
        resolved_parts = [r for r in resolved_parts if r]
        if resolved_parts:
            return resolved_parts[0] if len(resolved_parts) == 1 else resolved_parts

    return _resolve_author_year(raw, entries)


def enrich_missing_doi(entry: dict, timeout_sec: int = 10) -> dict:
    if not entry or entry.get("doi"):
        return entry

    title = str(entry.get("title") or "").strip()
    authors = entry.get("authors") or []
    if not title:
        return entry

    cache_key = _crossref_cache_key(title, authors)
    if cache_key in _CROSSREF_CACHE:
        cached = _CROSSREF_CACHE[cache_key]
        if cached:
            updated = dict(entry)
            updated["doi"] = cached
            return updated
        return entry

    try:
        query = {"query.title": title, "rows": 1}
        if authors:
            query["query.author"] = str(authors[0])

        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(query)
        req = urllib.request.Request(url, headers={"User-Agent": "PaperMind/1.0 (bibliography resolver)"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload = resp.read()

        data = json.loads(payload)
        items = data.get("message", {}).get("items", [])
        if not items:
            _CROSSREF_CACHE[cache_key] = None
            return entry

        doi = items[0].get("DOI") or ""
        if doi:
            _CROSSREF_CACHE[cache_key] = doi
            updated = dict(entry)
            updated["doi"] = doi
            return updated
        _CROSSREF_CACHE[cache_key] = None
        return entry
    except Exception as exc:
        logger.debug(f"Crossref lookup failed: {exc}")
        return entry


def _crossref_cache_key(title: str, authors: list[str]) -> str:
    base = title.lower().strip()
    first_author = str(authors[0]).lower().strip() if authors else ""
    return f"{base}|{first_author}"
