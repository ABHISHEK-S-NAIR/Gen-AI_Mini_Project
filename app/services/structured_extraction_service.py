import re

from app.core.state import state

BACKGROUND_CUES = (
    "previous work",
    "has been established",
    "rnn models",
    "earlier models",
    "prior models",
    "background",
    "related work",
)


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.?!])\s+", normalized)
    return [p.strip() for p in parts if p.strip()]


def _strip_noise(text: str) -> str:
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\([^\)]*\d{4}[^\)]*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;,")


def _phrase(text: str, max_words: int = 16) -> str:
    cleaned = _strip_noise(text)
    words = cleaned.split()
    if not words:
        return ""
    return " ".join(words[:max_words])


def _non_background_sentence(text: str) -> str:
    for sent in _sentences(text):
        lowered = sent.lower()
        if any(cue in lowered for cue in BACKGROUND_CUES):
            continue
        return sent
    return _sentences(text)[0] if _sentences(text) else ""


def _find_metrics(text: str) -> list[str]:
    """Extract all metrics from text."""
    pattern = r"\b\d+(?:\.\d+)?\s*(?:%|BLEU|ROUGE|F1|accuracy|EM|precision|recall|mAP|perplexity|loss)\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return matches if matches else []


def _find_datasets(text: str) -> list[str]:
    """Detect common datasets mentioned in the paper."""
    datasets = [
        "ImageNet", "COCO", "SQuAD", "GLUE", "SuperGLUE", "MNIST", "CIFAR-10", "CIFAR-100",
        "Pascal VOC", "ADE20K", "MS MARCO", "WikiText", "Penn Treebank", "WMT",
        "LibriSpeech", "CommonVoice", "OpenImages", "Visual Genome", "Cityscapes",
        "KITTI", "NYU Depth", "Places365", "Kinetics", "UCF101", "ActivityNet"
    ]
    found = []
    text_lower = text.lower()
    for dataset in datasets:
        if dataset.lower() in text_lower:
            found.append(dataset)
    return found


def _extract_improvements(text: str) -> list[str]:
    """Extract performance improvement statements."""
    improvements = []
    # Pattern: "improved by X%", "X% improvement", "outperforms by X%"
    patterns = [
        r"(?:improved?|improvement|gain|increase).*?by\s+(\d+\.?\d*)\s*%",
        r"(\d+\.?\d*)\s*%\s+(?:improvement|gain|increase|better)",
        r"outperforms?.*?by\s+(\d+\.?\d*)\s*%"
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        improvements.extend([f"{m}% improvement" for m in matches])
    return improvements[:3]  # Limit to top 3


def _extract_hyperparameters(text: str) -> dict[str, str]:
    """Extract training hyperparameters from text."""
    hyperparams = {}
    
    # Learning rate patterns
    lr_patterns = [
        r"learning[- ]rate[:\s=]+(\d+\.?\d*(?:e-?\d+)?)",
        r"\blr[:\s=]+(\d+\.?\d*(?:e-?\d+)?)",
        r"(?:initial|base)\s+(?:learning[- ])?rate[:\s=]+(\d+\.?\d*(?:e-?\d+)?)"
    ]
    for pattern in lr_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hyperparams["learning_rate"] = match.group(1)
            break
    
    # Batch size patterns
    batch_patterns = [
        r"batch[- ]size[:\s=]+(\d+)",
        r"mini[- ]?batch[:\s=]+(\d+)",
        r"(\d+)\s+(?:training\s+)?samples?\s+per\s+batch"
    ]
    for pattern in batch_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hyperparams["batch_size"] = match.group(1)
            break
    
    # Optimizer patterns
    optimizers = ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta", "LAMB", "AdaFactor"]
    for optimizer in optimizers:
        if re.search(rf"\b{optimizer}\b", text, re.IGNORECASE):
            hyperparams["optimizer"] = optimizer
            break
    
    # Epochs
    epoch_patterns = [
        r"(?:train|trained)\s+for\s+(\d+)\s+epochs?",
        r"(\d+)\s+epochs?\s+(?:of\s+)?training",
        r"epochs?[:\s=]+(\d+)"
    ]
    for pattern in epoch_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hyperparams["epochs"] = match.group(1)
            break
    
    # Dropout rate
    dropout_pattern = r"dropout[:\s=]+(\d+\.?\d*)"
    match = re.search(dropout_pattern, text, re.IGNORECASE)
    if match:
        hyperparams["dropout"] = match.group(1)
    
    # Weight decay
    wd_patterns = [
        r"weight[- ]decay[:\s=]+(\d+\.?\d*(?:e-?\d+)?)",
        r"L2[- ]?regularization[:\s=]+(\d+\.?\d*(?:e-?\d+)?)"
    ]
    for pattern in wd_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hyperparams["weight_decay"] = match.group(1)
            break
    
    return hyperparams


def _extract_model_dimensions(text: str) -> dict[str, str]:
    """Extract model architecture dimensions."""
    dimensions = {}
    
    # Model size (parameters)
    param_patterns = [
        r"(\d+\.?\d*)\s*(?:million|M|billion|B)\s+parameters?",
        r"parameters?[:\s]+(\d+\.?\d*)\s*(?:million|M|billion|B)",
        r"model\s+size[:\s]+(\d+\.?\d*)\s*(?:million|M|billion|B)"
    ]
    for pattern in param_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dimensions["parameters"] = match.group(1) + ("M" if "million" in match.group(0).lower() or "M" in match.group(0) else "B")
            break
    
    # Hidden dimension
    hidden_patterns = [
        r"hidden[- ](?:dimension|size|units?)[:\s=]+(\d+)",
        r"d[_-]?model[:\s=]+(\d+)",
        r"embedding[- ](?:dimension|size)[:\s=]+(\d+)"
    ]
    for pattern in hidden_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dimensions["hidden_dim"] = match.group(1)
            break
    
    # Number of layers
    layer_patterns = [
        r"(\d+)[- ]layers?",
        r"(?:number|num)[- ]of[- ]layers?[:\s=]+(\d+)",
        r"layers?[:\s=]+(\d+)",
        r"depth[:\s=]+(\d+)"
    ]
    for pattern in layer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dimensions["num_layers"] = match.group(1)
            break
    
    # Attention heads
    head_patterns = [
        r"(\d+)[- ](?:attention[- ])?heads?",
        r"heads?[:\s=]+(\d+)",
        r"multi[- ]head[^.]*?(\d+)[- ]heads?"
    ]
    for pattern in head_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dimensions["num_heads"] = match.group(1)
            break
    
    # Feed-forward dimension
    ff_patterns = [
        r"feed[- ]?forward[^.]*?(\d+)",
        r"FFN[^.]*?(\d+)",
        r"d[_-]?ff[:\s=]+(\d+)"
    ]
    for pattern in ff_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dimensions["ffn_dim"] = match.group(1)
            break
    
    return dimensions


def _extract_ablation_studies(text: str) -> list[str]:
    """Detect and extract ablation study insights."""
    ablations = []
    
    # Look for "ablation" section or mentions
    if "ablation" not in text.lower():
        return ablations
    
    # Find sentences near "ablation"
    sentences = _sentences(text)
    for i, sent in enumerate(sentences):
        if "ablation" in sent.lower():
            # Get context around ablation mention (current + next 2 sentences)
            context_sentences = sentences[i:min(i+3, len(sentences))]
            
            # Look for patterns indicating component importance
            for ctx_sent in context_sentences:
                lower = ctx_sent.lower()
                
                # Pattern: "removing X leads to/causes/results in Y"
                if any(k in lower for k in ["removing", "without", "ablating"]):
                    ablations.append(_phrase(ctx_sent, max_words=20))
                
                # Pattern: "X contributes Y"
                elif any(k in lower for k in ["contributes", "contribution", "important", "critical", "essential"]):
                    ablations.append(_phrase(ctx_sent, max_words=20))
                
                # Pattern: mentions drop/decrease/improvement
                elif any(k in lower for k in ["drop", "decrease", "degrades", "improves", "increase"]) and re.search(r"\d+", ctx_sent):
                    ablations.append(_phrase(ctx_sent, max_words=20))
            
            if len(ablations) >= 3:
                break
    
    return ablations[:3]  # Limit to top 3


def _infer_core_technique(text: str) -> str:
    lowered = text.lower()
    rules = [
        ("self-attention", "self-attention"),
        ("transformer", "transformer architecture"),
        ("masked language model", "MLM + NSP"),
        ("mlm", "MLM + NSP"),
        ("retrieval", "retrieval-augmented generation"),
        ("graph neural", "graph-guided learning"),
        ("lstm", "recurrent sequence modeling"),
    ]
    for key, value in rules:
        if key in lowered:
            return value
    return "task-specific neural modeling"


# Fields where a placeholder means extraction failed
_PLACEHOLDER_SUFFIXES = (
    "not clearly extracted",
    "not clearly extracted",
)


def _is_placeholder(value: str) -> bool:
    """Return True if the extracted value is a fallback placeholder."""
    return (
        not value
        or value.strip().endswith("not clearly extracted")
        or value.strip() == "problem not clearly extracted"
        or value.strip() == "method not clearly extracted"
        or value.strip() == "results not clearly extracted"
        or value.strip() == "novelty not clearly extracted"
        or value.strip() == "limitations not clearly extracted"
    )


def _llm_extract_structured(
    paper_id: str,
    existing: dict[str, str],
) -> dict[str, str]:
    """
    LLM fallback for structured extraction. Only fills in fields where
    existing regex extraction returned placeholders.

    Builds a focused context string from available section text, then
    calls call_llm_json asking for ONLY the fields that are still empty.
    Merges LLM results back into existing dict and returns it.

    Falls back gracefully — if LLM call fails, returns existing unchanged.
    """
    fields_needed = [
        f for f in ("problem", "proposed_method", "core_technique",
                    "results", "novelty", "limitations")
        if _is_placeholder(existing.get(f, ""))
    ]

    if not fields_needed:
        return existing

    try:
        from app.services.llm_client import call_llm_json

        sections = state.sections.get(paper_id, {})

        section_priority = {
            "problem": ["abstract", "intro"],
            "proposed_method": ["method", "intro"],
            "core_technique": ["method", "abstract"],
            "results": ["results", "conclusion"],
            "novelty": ["intro", "abstract", "conclusion"],
            "limitations": ["conclusion", "results"],
        }

        context_parts = []
        seen_sections = set()
        for field in fields_needed:
            for sec in section_priority.get(field, ["abstract"]):
                if sec not in seen_sections and sections.get(sec, "").strip():
                    words = sections[sec].strip().split()
                    context_parts.append(
                        f"[{sec.upper()}]\n" + " ".join(words[:300])
                    )
                    seen_sections.add(sec)

        if not context_parts:
            return existing

        context = "\n\n".join(context_parts)

        field_descriptions = {
            "problem": "The specific research problem or challenge this paper addresses (1-2 sentences)",
            "proposed_method": "The method or approach proposed to solve the problem (1-2 sentences)",
            "core_technique": "The single core technical technique at the heart of the method (short phrase, e.g. 'multi-head self-attention')",
            "results": "The main empirical results or outcomes reported (1-2 sentences)",
            "novelty": "What is genuinely new or novel about this contribution vs prior work (1-2 sentences)",
            "limitations": "Limitations, failure cases, or future work acknowledged by the authors (1-2 sentences)",
        }

        fields_block = "\n".join(
            f'  "{f}": {field_descriptions[f]}'
            for f in fields_needed
        )

        prompt = (
            "You are extracting structured metadata from a research paper.\n\n"
            f"Return a JSON object with ONLY these keys:\n{fields_block}\n\n"
            "Rules:\n"
            "- Base every answer strictly on the paper text below\n"
            "- If a field genuinely cannot be determined from the text, "
            'use an empty string ""\n'
            "- Return ONLY the JSON object. No markdown fences. No extra text.\n\n"
            f"Paper text:\n{context}"
        )

        system = (
            "You are a precise academic metadata extractor. "
            "Return only valid JSON with string values. No commentary."
        )

        llm_result = call_llm_json(prompt, system=system, max_tokens=600)

        if not isinstance(llm_result, dict):
            return existing

        updated = dict(existing)
        for field in fields_needed:
            llm_value = llm_result.get(field, "")
            if isinstance(llm_value, str):
                llm_value = llm_value.strip()
            else:
                llm_value = ""
            if llm_value:
                updated[field] = llm_value

        return updated

    except Exception:
        return existing


def extract_structured_data(paper_id: str) -> dict[str, str]:
    paper = state.papers.get(paper_id)
    sections = state.sections.get(paper_id, {})

    title = paper.filename if paper else "unknown.pdf"
    abstract = sections.get("abstract", "")
    intro = sections.get("intro", "")
    method = sections.get("method", "")
    results = sections.get("results", "")
    conclusion = sections.get("conclusion", "")

    if not any([abstract, intro, method, results, conclusion]):
        fallback = " ".join(c.text for c in state.chunks.get(paper_id, [])[:8])
        abstract = fallback
        intro = fallback
        method = fallback
        results = fallback
        conclusion = fallback

    method_src = _non_background_sentence(method) or _non_background_sentence(intro)
    problem_src = _non_background_sentence(" ".join([abstract, intro]))
    result_src = _non_background_sentence(" ".join([results, conclusion]))

    novelty_src = ""
    for sent in _sentences(" ".join([intro, conclusion, abstract])):
        l = sent.lower()
        if any(k in l for k in ("we propose", "we introduce", "novel", "first", "new architecture")):
            novelty_src = sent
            break
    if not novelty_src:
        novelty_src = method_src

    limitation_src = ""
    for sent in _sentences(" ".join([conclusion, results, intro])):
        l = sent.lower()
        if any(k in l for k in ("limitation", "future work", "fails", "challenging", "however")):
            limitation_src = sent
            break
    if not limitation_src:
        limitation_src = "limited evaluation details"

    full_text = " ".join([abstract, intro, method, results, conclusion])
    core_technique = _infer_core_technique(full_text)
    metrics = _find_metrics(full_text)
    datasets = _find_datasets(full_text)
    improvements = _extract_improvements(full_text)
    hyperparameters = _extract_hyperparameters(full_text)
    dimensions = _extract_model_dimensions(full_text)
    ablations = _extract_ablation_studies(full_text)

    architecture = "transformer-based" if "transformer" in full_text.lower() else "non-transformer neural"
    learning_strategy = "pretraining" if "pre-train" in full_text.lower() or "pretrain" in full_text.lower() else "task-specific training"
    contribution_type = "architectural" if "architecture" in full_text.lower() else "methodological"

    extracted = {
        "paper_id": paper_id,
        "title": title,
        "problem": _phrase(problem_src) or "problem not clearly extracted",
        "proposed_method": _phrase(method_src) or "method not clearly extracted",
        "core_technique": core_technique,
        "results": _phrase(result_src) or "results not clearly extracted",
        "novelty": _phrase(novelty_src) or "novelty not clearly extracted",
        "limitations": _phrase(limitation_src) or "limitations not clearly extracted",
        "metrics": metrics,
        "datasets": datasets,
        "improvements": improvements,
        "hyperparameters": hyperparameters,  # New field
        "dimensions": dimensions,  # New field
        "ablations": ablations,  # New field
        "architecture": architecture,
        "learning_strategy": learning_strategy,
        "contribution_type": contribution_type,
    }

    # LLM fallback: fill in any fields that regex could not extract
    extracted = _llm_extract_structured(paper_id, extracted)

    return extracted


def extract_structured_for_papers(paper_ids: list[str]) -> list[dict[str, str]]:
    return [extract_structured_data(pid) for pid in paper_ids]
