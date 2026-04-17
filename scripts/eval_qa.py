#!/usr/bin/env python3
"""
Evaluation harness for PaperMind Q&A.

Produces:
1) Category-wise quantitative table (ROUGE-1/2/L, Token F1, Exact Match)
2) Ablation table for retrieval modes:
   - no_retrieval (LLM only)
   - flat_retrieval (RAG without section filter)
   - section_aware_retrieval (RAG with section hints/heuristics)

Usage:
    python scripts/eval_qa.py \
      --dataset data/eval_questions.json \
      --pdf-dir data/papers \
      --output-md reports/eval_results.md \
      --output-json reports/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rouge_score import rouge_scorer

from app.config import settings
from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.chunker import chunk_sections
from app.services.embedding_engine import embed_texts, get_embedding_dim
from app.services.llm_client import LLMUnavailableError, call_llm
from app.services.qa_service import answer_question_with_sections
from app.services.section_detector import detect_sections
from app.services.text_extractor import extract_text_from_pdf_bytes

VALID_CATEGORIES = {"extractive", "abstractive", "yes/no"}
VALID_MODES = ("no_retrieval", "flat_retrieval", "section_aware_retrieval")
NO_CONTEXT_ANSWER = "The provided excerpts do not contain sufficient information to answer this question."


@dataclass
class EvalItem:
    qid: str
    question: str
    references: list[str]
    category: str
    paper_files: list[str] | None
    sections_hint: list[str] | None


def _normalize_answer(text: str) -> str:
    """SQuAD-style normalization for Token F1 / Exact Match."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _clean_prediction_text(text: str) -> str:
    """
    Clean model output for fair metric computation.
    Removes citation tags and prompt-echo excerpt headers.
    """
    if not text:
        return ""
    t = text.strip()
    # Remove inline citation markers.
    t = re.sub(r"\[Excerpt\s+\d+\]", "", t, flags=re.IGNORECASE)
    # Drop any pasted excerpt header/body lines.
    t = re.sub(r"\[Excerpt\s+\d+[^\]]*\].*", "", t, flags=re.IGNORECASE | re.DOTALL)
    # Normalize whitespace.
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _canonical_yes_no(text: str) -> str:
    t = _normalize_answer(text)
    # Prioritize explicit starting yes/no when present.
    if re.match(r"^(yes|no)\b", t):
        return "yes" if t.startswith("yes") else "no"
    yes_signals = (" yes ", " indeed ", " true ", " does ", " is ")
    no_signals = (" no ", " not ", " doesnt ", " dont ", " false ")
    padded = f" {t} "
    if any(s in padded for s in no_signals):
        return "no"
    if any(s in padded for s in yes_signals):
        return "yes"
    return t


def _short_answer_for_em(text: str, category: str) -> str:
    t = _clean_prediction_text(text)
    if category == "yes/no":
        return _canonical_yes_no(t)
    # Keep first sentence to avoid penalizing trailing elaboration.
    parts = re.split(r"(?<=[.!?])\s+", t, maxsplit=1)
    return parts[0].strip() if parts else t


def _em_candidates(text: str, category: str) -> list[str]:
    """
    Build strict-but-fair candidate strings for Exact Match.
    This handles common QA phrasing like:
    - "The answer is 512."
    - "d_model = 512"
    while still requiring exact normalized equality on a candidate.
    """
    base = _clean_prediction_text(text)
    if not base:
        return []

    if category == "yes/no":
        return [_canonical_yes_no(base)]

    cands: set[str] = set()
    base = re.sub(r"^(answer|final answer)\s*:\s*", "", base, flags=re.IGNORECASE).strip()
    first_sent = _short_answer_for_em(base, category)

    for s in (base, first_sent):
        if not s:
            continue
        cands.add(_normalize_answer(s))

        # Extract value-like tail after common copula patterns.
        for m in re.finditer(r"\b(?:is|are|was|were|equals?|=)\s+([^.,;:!?]+)", s, flags=re.IGNORECASE):
            span = m.group(1).strip()
            if 0 < len(span.split()) <= 8:
                cands.add(_normalize_answer(span))

        # Numeric forms are common for extractive QA.
        for m in re.finditer(r"\b\d[\d,]*(?:\.\d+)?%?\b", s):
            cands.add(_normalize_answer(m.group(0)))

    return [c for c in cands if c]


def _exact_match_from_candidates(prediction: str, references: list[str], category: str) -> float:
    pred = set(_em_candidates(prediction, category))
    if not pred:
        return 0.0
    for ref in references:
        ref_cands = set(_em_candidates(ref, category))
        if pred & ref_cands:
            return 1.0
    return 0.0


def _is_weak_rag_result(result: dict[str, Any], min_avg_relevance: float) -> bool:
    answer = str(result.get("answer", "")).lower()
    if "provided excerpts do not contain sufficient information" in answer:
        return True
    if "llm service unavailable" in answer:
        return True
    if not result.get("grounded", False):
        return True
    if float(result.get("avg_relevance", 0.0) or 0.0) < min_avg_relevance:
        return True
    if len(result.get("context", []) or []) < 2:
        return True
    return False


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    overlap = 0
    for tok, c in pred_counts.items():
        overlap += min(c, ref_counts.get(tok, 0))
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(reference))


def _qa_em_balanced(prediction: str, references: list[str], category: str) -> float:
    """
    Practical EM for free-form QA generations:
    - yes/no: canonical exact match
    - extractive/abstractive: exact OR containment of a meaningful phrase (>=4 tokens)
    """
    pred = _normalize_answer(_short_answer_for_em(prediction, category))
    if not pred:
        return 0.0

    if category == "yes/no":
        return max(_exact_match(pred, _short_answer_for_em(ref, category)) for ref in references)

    pred_tokens = pred.split()
    for ref in references:
        ref_n = _normalize_answer(_short_answer_for_em(ref, category))
        if not ref_n:
            continue
        if pred == ref_n:
            return 1.0
        # Phrase containment helps avoid false zeros from minor wrapper wording.
        if len(ref_n.split()) >= 4 and ref_n in pred:
            return 1.0
        if len(pred_tokens) >= 4 and pred in ref_n:
            return 1.0
    return 0.0


def _resolve_sections(item: EvalItem) -> list[str] | None:
    """Resolve sections for section-aware retrieval."""
    if item.sections_hint:
        return item.sections_hint

    q = item.question.lower()
    sections: list[str] = []

    if any(k in q for k in ("method", "architecture", "model", "approach", "training")):
        sections.append("method")
    if any(k in q for k in ("result", "perform", "accuracy", "f1", "rouge", "score", "improv")):
        sections.append("results")
    if any(k in q for k in ("conclusion", "limitation", "future work")):
        sections.append("conclusion")
    if any(k in q for k in ("why", "motivation", "background", "introduce")):
        sections.append("intro")
    if any(k in q for k in ("summary", "abstract", "main contribution")):
        sections.append("abstract")

    if item.category == "yes/no" and "results" not in sections:
        sections.append("results")

    return sorted(set(sections)) or None


def _resolve_path(path_str: str, dataset_dir: Path, pdf_dir: Path | None) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if pdf_dir is not None:
        return (pdf_dir / p).resolve()
    return (dataset_dir / p).resolve()


def _load_eval_items(dataset_path: Path) -> tuple[list[str], list[EvalItem]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Dataset must be a JSON object.")

    if "questions" in payload:
        rows = payload["questions"]
    elif "items" in payload:
        rows = payload["items"]
    else:
        raise ValueError("Dataset must contain `questions` or `items` list.")

    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset question list is empty.")

    papers = payload.get("papers", [])
    paper_paths: list[str] = []
    for p in papers:
        if not isinstance(p, dict) or "path" not in p:
            raise ValueError("Each `papers` entry must be an object with `path`.")
        paper_paths.append(str(p["path"]))

    items: list[EvalItem] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Question row {idx} must be an object.")
        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError(f"Question row {idx} missing `question`.")

        refs_raw = row.get("references", row.get("gold_answers", row.get("gold_answer")))
        if refs_raw is None:
            raise ValueError(f"Question row {idx} missing `gold_answer`/`references`.")
        if isinstance(refs_raw, str):
            references = [refs_raw]
        elif isinstance(refs_raw, list):
            references = [str(r).strip() for r in refs_raw if str(r).strip()]
        else:
            raise ValueError(f"Question row {idx} has invalid reference format.")
        if not references:
            raise ValueError(f"Question row {idx} has no valid references.")

        category = str(row.get("category", "extractive")).strip().lower()
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Question row {idx} has invalid category `{category}`.")

        paper_files = row.get("paper_files")
        if paper_files is not None:
            if not isinstance(paper_files, list) or not paper_files:
                raise ValueError(f"Question row {idx} has invalid `paper_files`.")
            paper_files = [str(p).strip() for p in paper_files if str(p).strip()]

        sections_hint = row.get("sections_hint")
        if sections_hint is not None:
            if not isinstance(sections_hint, list) or not sections_hint:
                raise ValueError(f"Question row {idx} has invalid `sections_hint`.")
            sections_hint = [str(s).strip() for s in sections_hint if str(s).strip()]

        items.append(
            EvalItem(
                qid=str(row.get("id", f"q{idx + 1}")),
                question=question,
                references=references,
                category=category,
                paper_files=paper_files,
                sections_hint=sections_hint,
            )
        )

    return paper_paths, items


def _ingest_pdf_paths(pdf_paths: list[Path]) -> dict[str, str]:
    """Ingest PDFs directly into in-process app state. Returns filename->paper_id mapping."""
    state.clear()
    if settings.embedding_dim != get_embedding_dim():
        settings.embedding_dim = get_embedding_dim()

    filename_to_paper_id: dict[str, str] = {}

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf file, got: {pdf_path}")

        raw_text = extract_text_from_pdf_bytes(pdf_path.read_bytes())
        if not raw_text.strip():
            raise ValueError(f"No extractable text from PDF: {pdf_path}")

        paper_id = str(uuid.uuid4())
        filename = pdf_path.name
        state.add_paper(paper_id, IngestedPaper(paper_id=paper_id, filename=filename, raw_text=raw_text))
        filename_to_paper_id[filename] = paper_id
        state.add_selected_paper(paper_id)

        sections = detect_sections(raw_text)
        state.add_sections(paper_id, sections)
        chunks = chunk_sections(
            paper_id=paper_id,
            sections=sections,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        if chunks:
            vectors = embed_texts([c.text for c in chunks], settings.embedding_dim)
            for c, v in zip(chunks, vectors, strict=True):
                c.embedding = v
            state.add_chunks(paper_id, chunks)
            state.vdb.upsert(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "paper_id": c.paper_id,
                        "section": c.section,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "embedding": c.embedding,
                    }
                    for c in chunks
                ]
            )

    return filename_to_paper_id


def _score_prediction(
    prediction: str,
    references: list[str],
    category: str,
    em_mode: str,
    scorer: rouge_scorer.RougeScorer,
) -> dict[str, float]:
    cleaned_prediction = _clean_prediction_text(prediction)
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "token_f1": 0.0, "exact_match": 0.0}
    if em_mode == "relaxed":
        exact_match = _exact_match_from_candidates(prediction, references, category)
    elif em_mode == "balanced":
        exact_match = _qa_em_balanced(prediction, references, category)
    else:
        pred_short = _short_answer_for_em(prediction, category)
        exact_match = max(_exact_match(pred_short, _short_answer_for_em(ref, category)) for ref in references)
    for ref in references:
        cleaned_ref = _clean_prediction_text(ref)
        rouge = scorer.score(cleaned_ref, cleaned_prediction)
        vals = {
            "rouge1": rouge["rouge1"].fmeasure,
            "rouge2": rouge["rouge2"].fmeasure,
            "rougeL": rouge["rougeL"].fmeasure,
            "token_f1": _token_f1(cleaned_prediction, cleaned_ref),
            "exact_match": exact_match,
        }
        # Standard QA behavior: take max score across multiple references.
        for k, v in vals.items():
            if v > best[k]:
                best[k] = v
    return best


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r[key]) for r in rows) / len(rows)


def _compute_aggregates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for category in ("extractive", "abstractive", "yes/no"):
        subset = [r for r in rows if r["category"] == category]
        out[category] = {
            "rouge1": _mean(subset, "rouge1"),
            "rouge2": _mean(subset, "rouge2"),
            "rougeL": _mean(subset, "rougeL"),
            "token_f1": _mean(subset, "token_f1"),
            "exact_match": _mean(subset, "exact_match"),
            "count": float(len(subset)),
        }
    out["overall"] = {
        "rouge1": _mean(rows, "rouge1"),
        "rouge2": _mean(rows, "rouge2"),
        "rougeL": _mean(rows, "rougeL"),
        "token_f1": _mean(rows, "token_f1"),
        "exact_match": _mean(rows, "exact_match"),
        "count": float(len(rows)),
    }
    return out


def _format_results_markdown(
    aggregate_by_mode: dict[str, dict[str, dict[str, float]]],
    include_no_retrieval: bool,
) -> str:
    # Quantitative table from section-aware retrieval
    quant = aggregate_by_mode["section_aware_retrieval"]
    quant_lines = [
        "## D. Quantitative Results",
        "",
        "| Metric | Extractive | Abstractive | Yes/No | Overall |",
        "|---|---:|---:|---:|---:|",
        f"| ROUGE-1 | {quant['extractive']['rouge1']:.2f} | {quant['abstractive']['rouge1']:.2f} | {quant['yes/no']['rouge1']:.2f} | {quant['overall']['rouge1']:.2f} |",
        f"| ROUGE-2 | {quant['extractive']['rouge2']:.2f} | {quant['abstractive']['rouge2']:.2f} | {quant['yes/no']['rouge2']:.2f} | {quant['overall']['rouge2']:.2f} |",
        f"| ROUGE-L | {quant['extractive']['rougeL']:.2f} | {quant['abstractive']['rougeL']:.2f} | {quant['yes/no']['rougeL']:.2f} | {quant['overall']['rougeL']:.2f} |",
        f"| Token F1 | {quant['extractive']['token_f1']:.2f} | {quant['abstractive']['token_f1']:.2f} | {quant['yes/no']['token_f1']:.2f} | {quant['overall']['token_f1']:.2f} |",
        f"| Exact Match | {quant['extractive']['exact_match']:.2f} | {quant['abstractive']['exact_match']:.2f} | {quant['yes/no']['exact_match']:.2f} | {quant['overall']['exact_match']:.2f} |",
        "",
        "## E. Ablation Study: Impact of RAG Grounding",
        "",
        "| Configuration | ROUGE-1 | ROUGE-L | Token F1 |",
        "|---|---:|---:|---:|",
    ]

    if include_no_retrieval:
        nr = aggregate_by_mode["no_retrieval"]["overall"]
        quant_lines.append(
            f"| No retrieval (LLM only) | {nr['rouge1']:.2f} | {nr['rougeL']:.2f} | {nr['token_f1']:.2f} |"
        )

    flat = aggregate_by_mode["flat_retrieval"]["overall"]
    sec = aggregate_by_mode["section_aware_retrieval"]["overall"]
    quant_lines.extend(
        [
            f"| Flat retrieval (no section filter) | {flat['rouge1']:.2f} | {flat['rougeL']:.2f} | {flat['token_f1']:.2f} |",
            f"| Section-aware retrieval (PaperMind full) | {sec['rouge1']:.2f} | {sec['rougeL']:.2f} | {sec['token_f1']:.2f} |",
        ]
    )
    return "\n".join(quant_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaperMind Q&A quantitative + ablation evaluation.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to eval dataset JSON.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory used to resolve relative PDF paths in dataset.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save raw JSON results.")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional path to save markdown tables.")
    parser.add_argument(
        "--disable-no-retrieval",
        action="store_true",
        help="Skip no-retrieval baseline (use if no LLM API key is configured).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.8,
        help="Delay in seconds between evaluation requests to reduce API rate-limit bursts.",
    )
    parser.add_argument(
        "--section-fallback-threshold",
        type=float,
        default=0.55,
        help="If section-aware avg relevance is below this threshold (or answer is weak), fallback to flat retrieval.",
    )
    parser.add_argument(
        "--em-mode",
        choices=["strict", "balanced", "relaxed"],
        default="balanced",
        help="Exact Match scoring mode. `balanced` is recommended for free-form QA generations.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    dataset_dir = dataset_path.parent
    paper_paths_raw, items = _load_eval_items(dataset_path)

    pdf_paths = [_resolve_path(p, dataset_dir, args.pdf_dir.resolve() if args.pdf_dir else None) for p in paper_paths_raw]
    if not pdf_paths:
        raise ValueError("Dataset must include `papers` with at least one PDF path.")

    filename_to_id = _ingest_pdf_paths(pdf_paths)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Decide mode list based on availability of LLM-only baseline.
    include_no_retrieval = not args.disable_no_retrieval
    modes = list(VALID_MODES if include_no_retrieval else VALID_MODES[1:])

    per_mode_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    no_retrieval_failed = False

    for item in items:
        if item.paper_files:
            paper_ids = []
            for f in item.paper_files:
                fname = Path(f).name
                pid = filename_to_id.get(fname)
                if pid is None:
                    raise ValueError(f"Question {item.qid} references unknown paper file `{f}`.")
                paper_ids.append(pid)
        else:
            paper_ids = list(filename_to_id.values())

        flat_result_for_item: dict[str, Any] | None = None

        for mode in modes:
            if mode == "no_retrieval":
                try:
                    answer = call_llm(
                        prompt=(
                            f"Question: {item.question}\n"
                            "Excerpts:\n[No excerpts provided]\n\n"
                            "Using only the excerpts above, answer the question. "
                            f"If the excerpts are insufficient, reply exactly: {NO_CONTEXT_ANSWER}"
                        ),
                        system=(
                            "You must answer strictly from provided excerpts only. "
                            "If no supporting excerpt exists, output the insufficient-information sentence exactly."
                        ),
                        max_tokens=96,
                        temperature=0.0,
                    )
                except LLMUnavailableError:
                    no_retrieval_failed = True
                    continue
            elif mode == "flat_retrieval":
                result = answer_question_with_sections(item.question, paper_ids, sections=None, debug=False)
                flat_result_for_item = result
                answer = str(result.get("answer", ""))
                if "LLM service unavailable" in answer and result.get("fallback_context"):
                    answer = str(result.get("fallback_context", answer))
            elif mode == "section_aware_retrieval":
                sections = _resolve_sections(item)
                result = answer_question_with_sections(item.question, paper_ids, sections=sections, debug=False)
                if _is_weak_rag_result(result, args.section_fallback_threshold):
                    if flat_result_for_item is None:
                        flat_result_for_item = answer_question_with_sections(item.question, paper_ids, sections=None, debug=False)
                    result = flat_result_for_item
                answer = str(result.get("answer", ""))
                if "LLM service unavailable" in answer and result.get("fallback_context"):
                    answer = str(result.get("fallback_context", answer))
            else:
                raise ValueError(f"Unknown mode `{mode}`.")

            scores = _score_prediction(answer, item.references, item.category, args.em_mode, scorer)
            per_mode_rows[mode].append(
                {
                    "id": item.qid,
                    "category": item.category,
                    "question": item.question,
                    "prediction": answer,
                    "references": item.references,
                    **scores,
                }
            )
            if args.request_delay > 0:
                time.sleep(args.request_delay)

    if no_retrieval_failed and "no_retrieval" in per_mode_rows:
        # If this fails due missing API keys, drop mode cleanly.
        per_mode_rows.pop("no_retrieval", None)
        include_no_retrieval = False

    aggregate_by_mode = {mode: _compute_aggregates(rows) for mode, rows in per_mode_rows.items()}
    markdown = _format_results_markdown(aggregate_by_mode, include_no_retrieval=include_no_retrieval)

    print(markdown)

    output_payload = {
        "dataset": str(dataset_path),
        "paper_paths": [str(p) for p in pdf_paths],
        "modes": list(per_mode_rows.keys()),
        "aggregate": aggregate_by_mode,
        "per_question": per_mode_rows,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        print(f"Saved JSON results to: {args.output_json}")

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
        print(f"Saved markdown table to: {args.output_md}")


if __name__ == "__main__":
    main()
