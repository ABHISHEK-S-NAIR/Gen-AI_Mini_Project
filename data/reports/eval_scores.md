Quantitative Results

| Metric | Extractive | Abstractive | Yes/No | Overall |
|---|---:|---:|---:|---:|
| ROUGE-1 | 0.48 | 0.27 | 0.38 | 0.31 |
| ROUGE-2 | 0.27 | 0.09 | 0.32 | 0.15 |
| ROUGE-L | 0.39 | 0.18 | 0.38 | 0.26 |
| Token F1 | 0.43 | 0.23 | 0.34 | 0.27 |
| Exact Match | 0.67 | 0.00 | 1.00 | 0.56 |

Ablation Study: Impact of RAG Grounding

| Configuration | ROUGE-1 | ROUGE-L | Token F1 |
|---|---:|---:|---:|
| No retrieval (LLM only) | 0.38 | 0.32 | 0.33 |
| Flat retrieval (no section filter) | 0.30 | 0.24 | 0.24 |
| Section-aware retrieval (PaperMind full) | 0.31 | 0.26 | 0.27 |
