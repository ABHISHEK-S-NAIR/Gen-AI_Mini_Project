import re

from app.models.schemas import SectionName


SECTION_ALIASES: dict[SectionName, tuple[str, ...]] = {
    "abstract": ("abstract",),
    "intro": ("introduction", "intro"),
    "method": ("methodology", "methods", "method", "approach"),
    "results": ("results", "experiments", "evaluation"),
    "conclusion": ("conclusion", "conclusions", "discussion"),
    "other": tuple(),
}


def detect_sections(raw_text: str) -> dict[SectionName, str]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    boundaries: list[tuple[int, SectionName]] = []

    for i, line in enumerate(lines):
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        for section, aliases in SECTION_ALIASES.items():
            if section == "other":
                continue
            if normalized in aliases:
                boundaries.append((i, section))
                break

    if not boundaries:
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
        sections[section_name] = "\n".join(lines[start_idx + 1 : end_idx]).strip()

    return sections
