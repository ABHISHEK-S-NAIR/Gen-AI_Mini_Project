"""
arXiv recommender service.
Extracts keywords from ingested papers and queries the arXiv API to
suggest related papers. No LLM required - pure API integration.
arXiv API docs: https://arxiv.org/help/api/user-manual
"""
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from app.services.structured_extraction_service import extract_structured_data

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
_ARXIV_NS = "http://www.w3.org/2005/Atom"
_DEFAULT_MAX_RESULTS = 5
_REQUEST_TIMEOUT = 10  # seconds


def _build_query(structured: dict[str, str]) -> str:
    """
    Build an arXiv search query string from extracted structured data.
    Uses core_technique, problem, and dataset names as keywords.
    Keeps the query short (max 8 terms) to avoid overly narrow searches.
    """
    terms = []

    # Core technique is the most distinctive signal
    technique = structured.get("core_technique", "")
    if technique and technique != "task-specific neural modeling":
        # Take first 4 words only to keep query broad
        terms.extend(technique.split()[:4])

    # Add up to 2 dataset names - these are highly distinctive
    datasets = structured.get("datasets", [])
    terms.extend(datasets[:2])

    # Extract 2-3 noun phrases from problem statement
    problem = structured.get("problem", "")
    if problem and not problem.endswith("not clearly extracted"):
        # Take first 3 meaningful words, skip stop words
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
            "we",
            "our",
            "their",
            "of",
            "in",
            "on",
            "to",
            "for",
            "with",
            "and",
            "or",
            "but",
            "not",
        }
        words = [
            w
            for w in re.sub(r"[^\w\s]", "", problem).split()
            if w.lower() not in stopwords and len(w) > 3
        ]
        terms.extend(words[:3])

    # Deduplicate preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique_terms.append(term)

    query_terms = unique_terms[:8]
    if not query_terms:
        return ""

    return " ".join(query_terms)


def _fetch_arxiv(query: str, max_results: int) -> list[dict[str, str]]:
    """
    Call the arXiv API and parse the Atom feed response.
    Returns a list of dicts with keys: arxiv_id, title, authors,
    abstract, url, published.
    Returns [] on any network or parse error.
    """
    if not query.strip():
        return []

    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )

    url = f"{ARXIV_API_BASE}?{params}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "PaperMind/1.0 (research tool)"},
        )
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            xml_data = resp.read()
    except Exception as e:
        logger.warning(f"arXiv API request failed: {e}")
        return []

    try:
        root = ET.fromstring(xml_data)
        ns = {"atom": _ARXIV_NS}
        entries = root.findall("atom:entry", ns)

        results = []
        for entry in entries:
            def get(tag: str) -> str:
                el = entry.find(f"atom:{tag}", ns)
                return el.text.strip() if el is not None and el.text else ""

            arxiv_id_url = get("id")
            arxiv_id = arxiv_id_url.split("/abs/")[-1] if "/abs/" in arxiv_id_url else arxiv_id_url

            authors = [
                name_el.text.strip()
                for author in entry.findall("atom:author", ns)
                for name_el in [author.find("atom:name", ns)]
                if name_el is not None and name_el.text
            ]

            abstract = get("summary")
            abstract = re.sub(r"\s+", " ", abstract).strip()
            if len(abstract) > 300:
                abstract = abstract[:297] + "..."

            results.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": get("title"),
                    "authors": authors[:4],
                    "abstract": abstract,
                    "url": arxiv_id_url,
                    "published": get("published")[:10],
                }
            )

        return results

    except ET.ParseError as e:
        logger.warning(f"arXiv XML parse error: {e}")
        return []


def recommend_papers(paper_id: str, max_results: int = 5) -> dict[str, object]:
    """
    Given a paper_id of an already-ingested paper, query arXiv for
    related papers based on extracted keywords.

    Returns:
    {
      "paper_id": str,
      "query_used": str,
      "recommendations": list[dict],
      "count": int,
      "error": dict | None,
    }
    """
    try:
        max_results = int(max_results) if max_results else _DEFAULT_MAX_RESULTS
        if max_results <= 0:
            max_results = _DEFAULT_MAX_RESULTS

        structured = extract_structured_data(paper_id)
        query = _build_query(structured)

        if not query:
            return {
                "paper_id": paper_id,
                "query_used": "",
                "recommendations": [],
                "count": 0,
                "error": {
                    "code": "E016",
                    "message": "Could not extract keywords from paper.",
                },
            }

        recommendations = _fetch_arxiv(query, max_results)

        return {
            "paper_id": paper_id,
            "query_used": query,
            "recommendations": recommendations,
            "count": len(recommendations),
            "error": None,
        }

    except Exception as e:
        logger.error(f"arXiv recommender failed for {paper_id}: {e}")
        return {
            "paper_id": paper_id,
            "query_used": "",
            "recommendations": [],
            "count": 0,
            "error": {"code": "E016", "message": str(e)},
        }


def recommend_for_papers(paper_ids: list[str], max_results: int = 5) -> dict[str, object]:
    """
    Run recommend_papers for each paper_id and merge results.
    Deduplicates recommendations by arxiv_id across papers.

    Returns:
    {
      "papers": list[dict],
      "all_recommendations": list[dict],
      "total_unique": int,
    }
    """
    papers_output = []
    seen_ids: dict[str, int] = {}
    all_recs: dict[str, dict] = {}

    for paper_id in paper_ids:
        result = recommend_papers(paper_id, max_results)
        papers_output.append(result)
        for rec in result.get("recommendations", []):
            aid = rec["arxiv_id"]
            seen_ids[aid] = seen_ids.get(aid, 0) + 1
            all_recs[aid] = rec

    sorted_recs = sorted(
        all_recs.values(),
        key=lambda r: seen_ids[r["arxiv_id"]],
        reverse=True,
    )

    return {
        "papers": papers_output,
        "all_recommendations": sorted_recs,
        "total_unique": len(sorted_recs),
    }
