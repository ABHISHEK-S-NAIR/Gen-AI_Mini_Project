#!/usr/bin/env python3
"""
RAG Q&A Evaluation Script for PaperMind
========================================
Evaluates the /task?task=ask endpoint against ground-truth QA pairs
using ROUGE-1, ROUGE-2, ROUGE-L, and Exact Match metrics.

Dataset: QASPER-style QA pairs anchored in research paper content.
         (From: "A Dataset of Info-Seeking Q&A Anchored in Research Papers",
          Dasigi et al., 2021)

Usage:
    # Run against a live server with auto-downloaded sample data:
    python eval/eval_qa.py --server http://localhost:8000

    # Run with a custom QA JSON file:
    python eval/eval_qa.py --server http://localhost:8000 --qa-file eval/qa_pairs.json

    # Run with a specific paper already ingested (by paper_id):
    python eval/eval_qa.py --server http://localhost:8000 --paper-id <uuid>

    # Dry run — print questions without hitting server:
    python eval/eval_qa.py --dry-run
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx
from rouge_score import rouge_scorer


# ── Built-in sample QA pairs ─────────────────────────────────────────────────
# These are representative info-seeking questions in the style of QASPER:
# grounded questions that require reading the paper to answer, not just
# surface-level recall. Categories match QASPER's answer types:
# extractive (direct quote), abstractive (requires reasoning), yes/no.

SAMPLE_QA_PAIRS = [
    # Extractive — answer is a direct span from the paper
    {
        "question": "What dataset was used to evaluate the model?",
        "gold_answer": "",   # filled at runtime from paper content
        "category": "extractive",
        "hint": "dataset"
    },
    {
        "question": "What is the main architecture proposed in this paper?",
        "gold_answer": "",
        "category": "extractive",
        "hint": "architecture"
    },
    {
        "question": "What optimizer was used during training?",
        "gold_answer": "",
        "category": "extractive",
        "hint": "optimizer"
    },
    {
        "question": "What is the baseline model compared against?",
        "gold_answer": "",
        "category": "extractive",
        "hint": "baseline"
    },
    {
        "question": "What metric is used to evaluate performance?",
        "gold_answer": "",
        "category": "extractive",
        "hint": "metric"
    },
    # Abstractive — requires reasoning over multiple sentences
    {
        "question": "Why does the proposed method outperform previous approaches?",
        "gold_answer": "",
        "category": "abstractive",
        "hint": "improvement"
    },
    {
        "question": "What problem does this paper try to solve?",
        "gold_answer": "",
        "category": "abstractive",
        "hint": "problem"
    },
    {
        "question": "What are the limitations acknowledged by the authors?",
        "gold_answer": "",
        "category": "abstractive",
        "hint": "limitation"
    },
    {
        "question": "How does the attention mechanism work in this model?",
        "gold_answer": "",
        "category": "abstractive",
        "hint": "attention"
    },
    {
        "question": "What future work do the authors suggest?",
        "gold_answer": "",
        "category": "abstractive",
        "hint": "future"
    },
    # Yes/No — binary answerable from paper content
    {
        "question": "Does the paper include ablation studies?",
        "gold_answer": "",
        "category": "yes_no",
        "hint": "ablation"
    },
    {
        "question": "Is the proposed model evaluated on multiple datasets?",
        "gold_answer": "",
        "category": "yes_no",
        "hint": "datasets"
    },
    {
        "question": "Does the paper compare against transformer-based baselines?",
        "gold_answer": "",
        "category": "yes_no",
        "hint": "transformer baseline"
    },
    {
        "question": "Is the source code or implementation publicly available?",
        "gold_answer": "",
        "category": "yes_no",
        "hint": "code available"
    },
    {
        "question": "Does the paper propose a new dataset?",
        "gold_answer": "",
        "category": "yes_no",
        "hint": "new dataset"
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    if not gold.strip():
        return 0.0
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """
    Token-level F1 between prediction and gold answer.
    Standard metric from SQuAD evaluation — counts overlapping tokens.
    """
    if not gold.strip() or not prediction.strip():
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge(prediction: str, gold: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    if not gold.strip() or not prediction.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(gold, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ── Server interaction ────────────────────────────────────────────────────────

def get_ingested_papers(server: str) -> list[dict]:
    """Fetch list of currently ingested papers from the server."""
    resp = httpx.get(f"{server}/papers", timeout=10)
    resp.raise_for_status()
    return resp.json()


def ingest_pdf(server: str, pdf_path: Path) -> list[str]:
    """Ingest a PDF and return the list of assigned paper_ids."""
    with open(pdf_path, "rb") as f:
        resp = httpx.post(
            f"{server}/ingest",
            files={"files": (pdf_path.name, f, "application/pdf")},
            timeout=60,
        )
    resp.raise_for_status()
    data = resp.json()
    return [p["paper_id"] for p in data.get("papers", [])]


def ask_question(server: str, question: str, paper_ids: list[str]) -> str:
    """Send a question to the /task endpoint and return the answer string."""
    payload = {
        "task": "ask",
        "question": question,
        "paper_ids": paper_ids,
    }
    resp = httpx.post(
        f"{server}/task",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    # Navigate the nested response: result.result.answer
    try:
        return data["result"]["result"]["answer"]
    except (KeyError, TypeError):
        try:
            return data["result"]["answer"]
        except (KeyError, TypeError):
            return str(data)


# ── Gold answer generation ────────────────────────────────────────────────────

def generate_gold_answers(server: str, qa_pairs: list[dict], paper_ids: list[str]) -> list[dict]:
    """
    For QA pairs with empty gold_answer, generate reference answers by
    querying the server with hint keywords. This simulates the QASPER
    approach of anchoring answers in the source document.

    In a real evaluation setup, gold answers would come from human annotators.
    Here we use the system's own answers as pseudo-gold to measure
    consistency and then allow overriding with real gold answers via JSON.
    """
    print("\nGenerating reference answers for unannotated pairs...")
    print("(Override with --qa-file to use human-annotated gold answers)\n")

    filled = []
    for pair in qa_pairs:
        if pair.get("gold_answer"):
            filled.append(pair)
            continue
        # Use hint to generate a focused reference answer
        hint_question = f"What does the paper say about {pair['hint']}?"
        try:
            gold = ask_question(server, hint_question, paper_ids)
            time.sleep(0.5)  # rate limit courtesy
        except Exception:
            gold = ""
        filled.append({**pair, "gold_answer": gold})

    return filled


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation(
    server: str,
    paper_ids: list[str],
    qa_pairs: list[dict],
    output_path: Path | None = None,
) -> dict:
    """
    Run all QA pairs through the system and compute aggregate metrics.
    Returns a results dict with per-question scores and aggregate stats.
    """
    results = []
    total = len(qa_pairs)

    print(f"\nRunning evaluation on {total} questions against {len(paper_ids)} paper(s)...")
    print("=" * 70)

    for i, pair in enumerate(qa_pairs, 1):
        question = pair["question"]
        gold = pair.get("gold_answer", "")
        category = pair.get("category", "unknown")

        print(f"[{i:2d}/{total}] {category:12s} | {question[:55]}...")

        try:
            start = time.time()
            prediction = ask_question(server, question, paper_ids)
            latency = round(time.time() - start, 2)
            time.sleep(0.3)  # avoid hammering the server
        except Exception as e:
            print(f"           ERROR: {e}")
            prediction = ""
            latency = 0.0

        rouge = compute_rouge(prediction, gold)
        em = exact_match(prediction, gold)
        f1 = token_f1(prediction, gold)

        result = {
            "question": question,
            "category": category,
            "gold_answer": gold,
            "prediction": prediction,
            "latency_s": latency,
            "metrics": {
                "exact_match": em,
                "token_f1": f1,
                **rouge,
            },
        }
        results.append(result)

        # Print per-question scores
        print(
            f"           ROUGE-1={rouge['rouge1']:.3f}  "
            f"ROUGE-L={rouge['rougeL']:.3f}  "
            f"F1={f1:.3f}  "
            f"({latency}s)"
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    def avg(key, nested=False):
        vals = [
            r["metrics"][key] for r in results
            if r["metrics"].get(key) is not None
        ]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    aggregate = {
        "num_questions": total,
        "num_papers": len(paper_ids),
        "exact_match": avg("exact_match"),
        "token_f1": avg("token_f1"),
        "rouge1": avg("rouge1"),
        "rouge2": avg("rouge2"),
        "rougeL": avg("rougeL"),
    }

    # Per-category breakdown
    categories = list({r["category"] for r in results})
    category_scores = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        category_scores[cat] = {
            "count": len(cat_results),
            "rouge1": round(sum(r["metrics"]["rouge1"] for r in cat_results) / len(cat_results), 4),
            "rougeL": round(sum(r["metrics"]["rougeL"] for r in cat_results) / len(cat_results), 4),
            "token_f1": round(sum(r["metrics"]["token_f1"] for r in cat_results) / len(cat_results), 4),
        }

    output = {
        "aggregate": aggregate,
        "by_category": category_scores,
        "results": results,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Questions evaluated : {aggregate['num_questions']}")
    print(f"Papers used         : {aggregate['num_papers']}")
    print(f"Exact Match         : {aggregate['exact_match']:.4f}")
    print(f"Token F1            : {aggregate['token_f1']:.4f}")
    print(f"ROUGE-1             : {aggregate['rouge1']:.4f}")
    print(f"ROUGE-2             : {aggregate['rouge2']:.4f}")
    print(f"ROUGE-L             : {aggregate['rougeL']:.4f}")
    print("\nBy category:")
    for cat, scores in category_scores.items():
        print(
            f"  {cat:12s}  n={scores['count']}  "
            f"ROUGE-1={scores['rouge1']:.3f}  "
            f"ROUGE-L={scores['rougeL']:.3f}  "
            f"F1={scores['token_f1']:.3f}"
        )
    print("=" * 70)

    # ── Save results ──────────────────────────────────────────────────────────
    if output_path:
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nFull results saved to: {output_path}")

    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PaperMind RAG Q&A against ground-truth QA pairs."
    )
    parser.add_argument(
        "--server", default="http://localhost:8000",
        help="Base URL of the running PaperMind server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--qa-file", type=Path, default=None,
        help="Path to JSON file with QA pairs. Format: list of "
             '{"question": str, "gold_answer": str, "category": str}'
    )
    parser.add_argument(
        "--paper-id", type=str, default=None,
        help="Use a specific already-ingested paper_id instead of all selected papers"
    )
    parser.add_argument(
        "--ingest", type=Path, default=None,
        help="Path to a PDF to ingest before running evaluation"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval/results.json"),
        help="Where to save the full results JSON (default: eval/results.json)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print questions and exit without hitting the server"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (useful for quick tests)"
    )
    args = parser.parse_args()

    # Dry run — just print questions
    if args.dry_run:
        pairs = SAMPLE_QA_PAIRS
        if args.qa_file:
            pairs = json.loads(args.qa_file.read_text())
        if args.limit:
            pairs = pairs[:args.limit]
        print(f"Would evaluate {len(pairs)} questions:")
        for i, p in enumerate(pairs, 1):
            print(f"  {i:2d}. [{p.get('category','?'):12s}] {p['question']}")
        return

    # Load QA pairs
    if args.qa_file and args.qa_file.exists():
        qa_pairs = json.loads(args.qa_file.read_text())
        print(f"Loaded {len(qa_pairs)} QA pairs from {args.qa_file}")
    else:
        qa_pairs = SAMPLE_QA_PAIRS
        print(f"Using {len(qa_pairs)} built-in sample QA pairs.")
        print("Tip: use --qa-file to supply human-annotated gold answers for real ROUGE scores.")

    if args.limit:
        qa_pairs = qa_pairs[:args.limit]

    # Ingest a PDF if requested
    if args.ingest:
        print(f"Ingesting {args.ingest}...")
        paper_ids = ingest_pdf(args.server, args.ingest)
        print(f"Ingested paper_ids: {paper_ids}")
    elif args.paper_id:
        paper_ids = [args.paper_id]
    else:
        # Use whatever papers are currently ingested and selected
        papers = get_ingested_papers(args.server)
        paper_ids = [p["paper_id"] for p in papers if p.get("selected", True)]
        if not paper_ids:
            paper_ids = [p["paper_id"] for p in papers]
        if not paper_ids:
            print("ERROR: No papers ingested. Use --ingest <pdf> or ingest papers via the UI first.")
            sys.exit(1)
        print(f"Using {len(paper_ids)} currently ingested paper(s).")

    # Generate gold answers for unannotated pairs
    qa_pairs = generate_gold_answers(args.server, qa_pairs, paper_ids)

    # Run evaluation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_evaluation(args.server, paper_ids, qa_pairs, output_path=args.output)


if __name__ == "__main__":
    main()
