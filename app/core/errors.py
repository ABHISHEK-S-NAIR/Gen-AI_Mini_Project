from dataclasses import dataclass


@dataclass(frozen=True)
class AppError:
    code: str
    message: str


ERRORS = {
    "E001": AppError("E001", "PARSE_FAILURE"),
    "E002": AppError("E002", "NO_SECTIONS_FOUND"),
    "E003": AppError("E003", "EMBED_FAILURE"),
    "E004": AppError("E004", "NO_RELEVANT_PAPERS"),
    "E005": AppError("E005", "INSUFFICIENT_CONTEXT"),
    "E006": AppError("E006", "LLM_TIMEOUT"),
    "E007": AppError("E007", "CITATION_NONE"),
}
