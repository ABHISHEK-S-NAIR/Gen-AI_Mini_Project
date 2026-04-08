# PaperMind Q&A Evaluation

Evaluates the RAG Q&A system against ground-truth question-answer pairs,
following the methodology of Dasigi et al. (2021) — "A Dataset of Info-Seeking
Q&A Anchored in Research Papers" (QASPER).

## Setup

```bash
pip install rouge-score httpx
```

## Quick Start

```bash
# 1. Start the server
uvicorn app.main:app --reload

# 2. Ingest a paper and run eval in one command
python eval/eval_qa.py --ingest path/to/paper.pdf

# 3. Or use papers already ingested via the UI
python eval/eval_qa.py

# 4. See what questions will be asked without running
python eval/eval_qa.py --dry-run
```

## Using Real Gold Answers

For accurate ROUGE scores, edit `eval/qa_pairs.json` with answers
copied directly from the paper, then run:

```bash
python eval/eval_qa.py --qa-file eval/qa_pairs.json
```

## Output

Results are saved to `eval/results.json` with per-question scores
and aggregate ROUGE-1, ROUGE-2, ROUGE-L, Token F1, and Exact Match.

## Metrics

| Metric | Description |
|--------|-------------|
| ROUGE-1 | Unigram overlap with gold answer |
| ROUGE-2 | Bigram overlap with gold answer |
| ROUGE-L | Longest common subsequence overlap |
| Token F1 | Precision/recall of answer tokens (SQuAD-style) |
| Exact Match | 1.0 if normalized strings match exactly |

## Expected Scores (rough baselines)

With real LLM (Groq/Gemini) and SciBERT embeddings:
- ROUGE-1: 0.35–0.55 (abstractive questions)
- ROUGE-L: 0.25–0.45
- Token F1: 0.30–0.50

Without LLM (raw chunk truncation fallback):
- ROUGE-1: 0.10–0.25
- ROUGE-L: 0.08–0.18
