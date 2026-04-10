"""
Section detector for academic PDFs.
Handles real-world heading formats from PyPDF2-extracted text:
  - Bare words:           "Abstract", "Introduction", "Method"
  - Numbered:             "1. Introduction", "2.3 Methodology", "3 Results"
  - Roman numerals:       "I. Introduction", "IV. EXPERIMENTAL RESULTS"
  - Section symbol:       "§3 Method", "§ 3. Approach"
  - ALL CAPS:             "INTRODUCTION", "RELATED WORK AND BACKGROUND"
  - Prefixed variants:    "Proposed Method", "Our Approach", "System Design"
  - Common synonyms:      "Experiments", "Evaluation", "Discussion", "Ablation"
"""
import re
from app.models.schemas import SectionName


# ── Heading patterns per section ─────────────────────────────────────────────
#
# Each entry is a list of regex fragments that match the *content* of a heading
# line after the optional numeric/roman/symbol prefix is stripped.
# Patterns are tried in order; first match wins.

_SECTION_PATTERNS: dict[SectionName, list[str]] = {
    "abstract": [
        r"abstract",
    ],
    "intro": [
        r"introduction",
        r"intro\b",
        r"background\s+and\s+motivation",
        r"motivation\s+and\s+background",
        r"overview",
        r"problem\s+statement",
    ],
    "method": [
        r"method(?:ology|s)?",
        r"approach(?:es)?",
        r"proposed\s+(?:method|model|framework|approach|architecture|system)",
        r"our\s+(?:method|approach|model|framework|system)",
        r"model\s+(?:design|architecture|description)",
        r"system\s+(?:design|description|overview)",
        r"technical\s+approach",
        r"architecture",
        r"framework",
    ],
    "results": [
        r"(?:experimental\s+)?results?",
        r"experiments?",
        r"evaluation(?:s)?",
        r"empirical\s+(?:evaluation|results|study)",
        r"performance\s+(?:evaluation|analysis)",
        r"benchmarks?",
        r"ablation\s+(?:study|studies|analysis)",
        r"quantitative\s+(?:results|evaluation)",
        r"analysis",
    ],
    "conclusion": [
        r"conclusion(?:s)?",
        r"concluding\s+remarks?",
        r"discussion(?:s)?",
        r"summary",
        r"future\s+work",
        r"limitations?\s+and\s+future.*",
        r"conclusion\s+and\s+future.*",
    ],
}

# Matches the optional prefix before the heading content:
#   "1."  "2.3"  "1.2.3"  →  decimal numbering
#   "I."  "IV."  "ii."    →  Roman numerals (up to 4 chars to avoid false positives)
#   "§3"  "§ 3." "§3.2"   →  section symbol
# Roman numerals MUST be followed by a period and optional space to distinguish
# from words starting with I, V, X (like "Introduction", "Vocabulary", "XOR")
_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?:\d+\.?)+\s*"           # decimal: "1.", "2.3", "3.1.2"
    r"|[IVXivx]{1,4}\.\s*"      # roman:   "I.", "IV.", "ii." (period required)
    r"|§\s*\d*\.?\s*"           # symbol:  "§3", "§ 3."
    r")?"
)

# A heading line should be short — real headings are rarely more than 10 words.
# Lines longer than this are body text that accidentally matched.
_MAX_HEADING_WORDS = 10

# Compiled patterns cache
_COMPILED: dict[SectionName, list[re.Pattern]] = {
    section: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for section, pats in _SECTION_PATTERNS.items()
}


def _llm_detect_sections(raw_text: str) -> dict[SectionName, str] | None:
    """
    Fallback section detector using LLM. Only called when regex heading
    detection finds nothing.

    Strategy:
      1. Send the first 3000 characters to the LLM (enough to cover most
         title + abstract + early intro, which is where heading markers
         usually appear even in non-standard papers).
      2. Ask the LLM to return a JSON object mapping section names to the
         character offset where that section BEGINS in the FULL raw_text.
      3. Use those offsets to slice raw_text into sections.
      4. Return None if the LLM call fails, JSON is malformed, or no
         offsets are found — the caller falls back to the "other" bucket.

    Returns a dict with keys: abstract, intro, method, results, conclusion, other
    All values are strings (empty string if section not found).
    """
    try:
        from app.services.llm_client import call_llm_json

        sample = raw_text[:3000]

        prompt = (
            "You are analyzing the beginning of a research paper (first ~3000 characters).\n\n"
            "Identify where each of the following sections begins in the text below. "
            "For each section you can find, return its CHARACTER OFFSET (integer) "
            "- the index into the text where that section's content starts "
            "(just after the heading line).\n\n"
            "Return ONLY a JSON object with these exact keys (omit keys you cannot find):\n"
            '  "abstract", "intro", "method", "results", "conclusion"\n\n'
            "Values must be integers (character offsets). "
            "Return ONLY the JSON object. No markdown fences. No explanation.\n\n"
            "Example output: {\"abstract\": 142, \"intro\": 891, \"method\": 2103}\n\n"
            f"Paper text:\n{sample}"
        )

        system = (
            "You are a precise document structure analyzer. "
            "Return only valid JSON objects with integer values. No extra text."
        )

        offsets = call_llm_json(prompt, system=system, max_tokens=200)

        if not isinstance(offsets, dict) or not offsets:
            return None

        valid_keys = {"abstract", "intro", "method", "results", "conclusion"}
        clean: dict[str, int] = {}
        for k, v in offsets.items():
            if k in valid_keys:
                try:
                    offset = int(v)
                    if 0 <= offset < len(raw_text):
                        clean[k] = offset
                except (TypeError, ValueError):
                    continue

        if not clean:
            return None

        sorted_sections = sorted(clean.items(), key=lambda x: x[1])

        result: dict[SectionName, str] = {
            "abstract": "",
            "intro": "",
            "method": "",
            "results": "",
            "conclusion": "",
            "other": "",
        }

        for i, (section_name, start_offset) in enumerate(sorted_sections):
            end_offset = (
                sorted_sections[i + 1][1]
                if i + 1 < len(sorted_sections)
                else len(raw_text)
            )
            result[section_name] = raw_text[start_offset:end_offset].strip()

        first_offset = sorted_sections[0][1]
        if first_offset > 0:
            result["other"] = raw_text[:first_offset].strip()

        if not any(result[k].strip() for k in valid_keys):
            return None

        return result

    except Exception:
        return None


def _classify_line(line: str) -> SectionName | None:
    """
    Try to classify a single line as a section heading.
    Returns the SectionName if matched, None otherwise.
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Reject lines that are clearly body text (too long)
    if len(stripped.split()) > _MAX_HEADING_WORDS:
        return None

    # Strip the numeric/roman/symbol prefix to get the heading content
    content = _PREFIX_RE.sub("", stripped).strip()
    if not content:
        return None

    for section, patterns in _COMPILED.items():
        for pat in patterns:
            # Use fullmatch for exact matches only - prevents body text
            # containing keywords like "propose" or "method" from being
            # misclassified as headings
            if pat.fullmatch(content):
                return section

    return None


def detect_sections(raw_text: str) -> dict[SectionName, str]:
    """
    Parse raw PDF text into labelled sections.

    Strategy:
    1. Walk every line and attempt to classify it as a section heading.
    2. Record boundary positions where headings are found.
    3. Slice the text between consecutive boundaries to get section content.
    4. If no boundaries are found at all, dump everything into 'other'
       so downstream services still receive text to work with.
    5. If only some sections are found, missing ones remain empty strings.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    boundaries: list[tuple[int, SectionName]] = []

    for i, line in enumerate(lines):
        section = _classify_line(line)
        if section is not None:
            # Avoid duplicate consecutive detections of the same section
            # (some PDFs repeat the heading on a new page)
            if boundaries and boundaries[-1][1] == section:
                continue
            boundaries.append((i, section))

    # No headings found — return everything as 'other' so nothing is lost
    if not boundaries:
        # Try LLM fallback before giving up
        llm_result = _llm_detect_sections(raw_text)
        if llm_result is not None:
            return llm_result
        # LLM unavailable or failed — return everything as 'other'
        return {
            "abstract": "",
            "intro": "",
            "method": "",
            "results": "",
            "conclusion": "",
            "other": "\n".join(lines),
        }

    boundaries.sort(key=lambda x: x[0])
    sections: dict[SectionName, str] = {
        "abstract": "",
        "intro": "",
        "method": "",
        "results": "",
        "conclusion": "",
        "other": "",
    }

    for idx, (start_idx, section_name) in enumerate(boundaries):
        end_idx = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        content = "\n".join(lines[start_idx + 1 : end_idx]).strip()
        # If the same section heading appears twice (e.g. page repeat),
        # append rather than overwrite so no content is lost
        if sections[section_name]:
            sections[section_name] += "\n" + content
        else:
            sections[section_name] = content

    return sections
