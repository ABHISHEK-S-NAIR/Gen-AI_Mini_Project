from app.core.state import state
from app.services.llm_client import call_llm
from app.services.structured_extraction_service import extract_structured_for_papers
from app.services.dimension_inference_service import (
    DimensionInference,
    create_component_breakdown,
    format_component_breakdown,
)


_EXPLAIN_SYSTEM = (
    "You are an expert at explaining research papers. "
    "Base your explanation strictly on the provided paper content. "
    "Do not add information that is not present in the text."
)

_SECTION_WORD_LIMIT = 600  # words per section fed into explanation prompts


def _build_explanation_context(paper_id: str, item: dict) -> str:
    """
    Build a labelled context block from raw paper sections.
    Falls back to structured extraction strings if sections are empty.
    This is the retrieval step — contextual embeddings from BERT power
    the section chunking upstream; here we use the retrieved text directly.
    """
    sections = state.sections.get(paper_id, {})
    parts = []

    for sec in ("abstract", "intro", "method", "results", "conclusion"):
        text = sections.get(sec, "").strip()
        if text:
            words = text.split()
            truncated = " ".join(words[:_SECTION_WORD_LIMIT])
            parts.append(f"[{sec.upper()}]\n{truncated}")

    if parts:
        return "\n\n".join(parts)

    # Fallback: use structured extraction strings if no sections available
    return (
        f"[PROBLEM]\n{item['problem']}\n\n"
        f"[METHOD]\n{item['proposed_method']} using {item['core_technique']}\n\n"
        f"[RESULTS]\n{item['results']}\n\n"
        f"[NOVELTY]\n{item['novelty']}"
    )


def _prompt_beginner(context: str) -> str:
    return f"""Explain this research paper to a curious high school student
who has never studied machine learning or computer science.

Rules you must follow:
- Use everyday language and simple analogies — no technical jargon
- If you must use a technical term, immediately explain it in plain words
- Focus on WHY this research matters in the real world
- Keep it to 3-4 short paragraphs maximum

Paper content:
{context}"""


def _prompt_intermediate(context: str, item: dict) -> str:
    metrics_str = ", ".join(item.get("metrics", [])[:4]) if item.get("metrics") else "see results section"
    datasets_str = ", ".join(item.get("datasets", [])) if item.get("datasets") else "see paper"
    return f"""Explain this research paper to a computer science undergraduate
who understands the basics of machine learning but is not an expert.

Structure your explanation with these exact sections:
PROBLEM: What challenge does this paper address and why does it matter?
METHOD: How does the proposed approach work technically? (2-3 sentences)
KEY INNOVATION: What is the single most novel idea compared to prior work?
RESULTS: What did the experiments show? Include specific metrics if available.
LIMITATIONS: What are the main weaknesses or open questions?

Important: the results section must mention metrics or benchmarks.
Known metrics from extraction: {metrics_str}
Known datasets: {datasets_str}

Paper content:
{context}"""


def _prompt_expert(context: str, item: dict) -> str:
    hyperparams_str = _format_hyperparams(item.get("hyperparameters", {}))
    dimensions_str = _format_dimensions(item.get("dimensions", {}))
    ablations_str = "; ".join(item.get("ablations", [])) if item.get("ablations") else "not reported"
    return f"""Provide a rigorous technical analysis of this research paper
for an ML researcher who will evaluate it critically.

Cover each of the following — be specific and use the paper's own numbers:
1. ARCHITECTURE & DESIGN: What is the base architecture? What are the key
   architectural choices and why were they made?
2. TRAINING STRATEGY: How was the model trained? Include optimizer, learning
   rate, batch size, and any pretraining or fine-tuning stages.
3. NOVELTY vs PRIOR WORK: What specifically distinguishes this from baselines
   mentioned in the paper?
4. EMPIRICAL EVIDENCE: Report concrete numbers — BLEU, accuracy, F1, etc.
   How large are the gains? Are they statistically significant?
5. ABLATION & ANALYSIS: What ablation studies were conducted and what did
   they reveal about which components matter most?
6. LIMITATIONS & FUTURE WORK: What does the paper itself acknowledge as
   limitations? What would a critical reviewer flag?

Pre-extracted values for reference (use paper content to verify/expand):
Architecture: {item.get('architecture', 'unknown')}
Dimensions: {dimensions_str}
Hyperparameters: {hyperparams_str}
Ablations noted: {ablations_str}

Paper content:
{context}"""


def _format_hyperparams(hyperparams: dict[str, str]) -> str:
    """Format hyperparameters into a readable string."""
    if not hyperparams:
        return "not specified"
    
    parts = []
    if "learning_rate" in hyperparams:
        parts.append(f"LR={hyperparams['learning_rate']}")
    if "batch_size" in hyperparams:
        parts.append(f"Batch={hyperparams['batch_size']}")
    if "optimizer" in hyperparams:
        parts.append(f"Optimizer={hyperparams['optimizer']}")
    if "epochs" in hyperparams:
        parts.append(f"Epochs={hyperparams['epochs']}")
    
    return ", ".join(parts) if parts else "not specified"


def _format_dimensions(dimensions: dict[str, str]) -> str:
    """Format model dimensions into a readable string."""
    if not dimensions:
        return "not specified"
    
    parts = []
    if "parameters" in dimensions:
        parts.append(f"{dimensions['parameters']} params")
    if "num_layers" in dimensions:
        parts.append(f"{dimensions['num_layers']} layers")
    if "hidden_dim" in dimensions:
        parts.append(f"d_model={dimensions['hidden_dim']}")
    if "num_heads" in dimensions:
        parts.append(f"{dimensions['num_heads']} heads")
    
    return ", ".join(parts) if parts else "not specified"


def _diagram_with_dimensions(technique: str, architecture: str, dimensions: dict[str, str]) -> str:
    """Generate ASCII diagram with dimension annotations."""
    t = technique.lower()
    a = architecture.lower()
    
    # Extract common dimensions
    hidden_dim = dimensions.get("hidden_dim", "d_model")
    num_layers = dimensions.get("num_layers", "N")
    num_heads = dimensions.get("num_heads", "h")
    ffn_dim = dimensions.get("ffn_dim", "d_ff")
    params = dimensions.get("parameters", "")
    
    # Transformer-based architectures
    if "self-attention" in t or "transformer" in t:
        param_note = f" ({params} total)" if params else ""
        return f"""[Input Sequence]
      ↓
┌─────────────────────────┐
│   Token Embedding       │
│   + Positional Encoding │
│   Output: (seq_len, {hidden_dim}) │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│ Multi-Head Attention    │
│ • {num_heads} attention heads      │
│ • Q, K, V projections   │
│ Output: (seq_len, {hidden_dim}) │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│ Feed-Forward Network    │
│ • FFN dim: {ffn_dim}           │
│ • Activation: ReLU/GELU │
│ Output: (seq_len, {hidden_dim}) │
└─────────────────────────┘
      ↓
  [× {num_layers} layers]
      ↓
[Output / Predictions]{param_note}"""
    
    # CNN architectures
    if "cnn" in t or "convolutional" in t or "resnet" in t or "conv" in a:
        param_note = f" ({params} total)" if params else ""
        return f"""[Input Image: (H, W, 3)]
      ↓
┌─────────────────────────┐
│ Conv Block 1            │
│ • 3×3 or 5×5 kernels    │
│ • BatchNorm + ReLU      │
│ • Max Pooling           │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│ Deep Conv Blocks        │
│ • Residual connections  │
│ • {num_layers} total layers        │
│ • Feature maps grow     │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│ Global Average Pooling  │
│ + Classification Head   │
└─────────────────────────┘
      ↓
[Predictions]{param_note}"""
    
    # Retrieval-augmented generation
    if "retrieval" in t:
        return f"""[Query: (seq_len, {hidden_dim})]
   ↓
┌──────────────────────┐
│  Dense Retriever     │ ←─── [Document Store]
│  • Embedding-based   │      (millions of docs)
│  • Top-k selection   │
└──────────────────────┘
   ↓
[Retrieved Contexts: k documents]
   ↓
┌──────────────────────┐
│  Generator (LM)      │
│  • Condition on ctx  │
│  • {num_layers} decoder layers  │
│  • d_model: {hidden_dim}        │
└──────────────────────┘
   ↓
[Grounded Output]"""
    
    # MLM (BERT-style)
    if "mlm" in t or "bert" in t:
        param_note = f" ({params} total)" if params else ""
        return f"""[Input Text]
      ↓
┌─────────────────────────┐
│ Tokenization            │
│ • Add [CLS], [SEP]      │
│ • Randomly mask 15%     │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│ Embedding Layer         │
│ • Token + Position + Segment │
│ Output: (seq_len, {hidden_dim})   │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│ Bidirectional Encoder   │
│ • {num_layers} transformer layers    │
│ • {num_heads} attention heads        │
│ • d_model: {hidden_dim}              │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│ MLM Prediction Head     │
│ • Predict masked tokens │
└─────────────────────────┘
      ↓
[Pre-trained Model]{param_note}"""
    
    # GAN architectures
    if "gan" in t or "generative adversarial" in t:
        return f"""[Random Noise z: ({hidden_dim},)]        [Real Data x]
      ↓                              ↓
┌──────────────────┐     ┌────────────────────┐
│   Generator      │     │   Discriminator    │
│   G(z) → x_fake  │     │   D(x) → [0,1]     │
│   • Upsampling   │     │   • Downsampling   │
│   • {num_layers} layers      │     │   • Binary classifier  │
└──────────────────┘     └────────────────────┘
      ↓                              ↑
[Generated Samples x_fake] ──────────┘
      ↓
[Adversarial Training]
  • L_G: fool discriminator
  • L_D: detect real vs fake"""
    
    # Encoder-Decoder
    if "encoder-decoder" in t or "seq2seq" in t or "sequence-to-sequence" in t:
        return f"""[Input Sequence: (src_len,)]
      ↓
┌─────────────────────────┐
│    Encoder              │
│    • {num_layers} layers            │
│    • Hidden: {hidden_dim}           │
│    • Bidirectional      │
└─────────────────────────┘
      ↓
[Context Vector: ({hidden_dim},)]
      ↓
┌─────────────────────────┐
│    Decoder              │
│    • {num_layers} layers            │
│    • Hidden: {hidden_dim}           │
│    • Autoregressive     │
└─────────────────────────┘
      ↓
[Output Sequence: (tgt_len,)]"""
    
    # GNN
    if "graph" in t or "gnn" in t:
        return f"""[Graph: G = (V, E)]
  V nodes, E edges
      ↓
┌─────────────────────────┐
│ Node Feature Init       │
│ • X_0: (V, {hidden_dim})        │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│ Message Passing Layer   │
│ • Aggregate neighbors   │
│ • Update: h_v = f(h_v, {{h_u}}) │
│ • {num_layers} iterations           │
└─────────────────────────┘
      ↓
[Node Embeddings: (V, {hidden_dim})]
      ↓
[Optional: Graph-level pooling]"""
    
    # Contrastive Learning
    if "contrastive" in t or "simclr" in t or "clip" in t:
        return f"""[Anchor x]    [Positive x+]    [Negative x-]
    ↓              ↓                ↓
┌───────────────────────────────────────────┐
│         Shared Encoder Network            │
│         • d_output: {hidden_dim}                  │
└───────────────────────────────────────────┘
    ↓              ↓                ↓
[z_anchor]    [z_pos]          [z_neg]
    ↓              ↓                ↓
┌───────────────────────────────────────────┐
│     Contrastive Loss (InfoNCE)            │
│     • Maximize: sim(z_anchor, z_pos)      │
│     • Minimize: sim(z_anchor, z_neg)      │
└───────────────────────────────────────────┘"""
    
    # Default fallback
    return f"""[Input Data]
      ↓
┌─────────────────────────┐
│ Feature Extraction      │
│ • Input dim: variable   │
│ • Hidden: {hidden_dim}              │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│ Processing Layers       │
│ • {num_layers} layers               │
└─────────────────────────┘
      ↓
[Output/Prediction]"""


def _training_vs_inference_view(technique: str, dimensions: dict[str, str], hyperparams: dict[str, str]) -> str:
    """Generate training vs inference comparison diagram."""
    lr = hyperparams.get("learning_rate", "lr")
    batch = hyperparams.get("batch_size", "B")
    optimizer = hyperparams.get("optimizer", "Optimizer")
    hidden = dimensions.get("hidden_dim", "d")
    
    return f"""
════════════════════════════════════════════════════════════════
                     TRAINING VIEW
════════════════════════════════════════════════════════════════
[Training Data Batch: ({batch}, seq_len, {hidden})]
      ↓
┌────────────────────────────────┐
│        Model Forward Pass      │
│        • Compute logits        │
│        • Generate predictions  │
└────────────────────────────────┘
      ↓
┌────────────────────────────────┐
│        Loss Computation        │
│        • Compare with labels   │
│        • Loss function         │
└────────────────────────────────┘
      ↓
┌────────────────────────────────┐
│       Backpropagation          │
│       • Compute gradients      │
│       • {optimizer} (lr={lr})        │
│       • Update weights         │
└────────────────────────────────┘
      ↓
[Updated Model Parameters]

════════════════════════════════════════════════════════════════
                    INFERENCE VIEW
════════════════════════════════════════════════════════════════
[Input Data: (seq_len, {hidden})]
      ↓
┌────────────────────────────────┐
│        Model Forward Pass      │
│        • No gradient tracking  │
│        • Use eval mode         │
│        • Faster (no backprop)  │
└────────────────────────────────┘
      ↓
[Predictions]
       ↓
[Post-processing (if needed)]
      ↓
[Final Output]
"""


def _explain_single(paper_id: str, level: str) -> dict[str, object]:
    items = extract_structured_for_papers([paper_id])
    if not items:
        return {"paper_id": paper_id, "level": level, "explanation": "No data available.", "diagram": None}

    item = items[0]

    # Format metrics list
    metrics_str = ", ".join(item['metrics'][:5]) if item['metrics'] else "no explicit metrics"
    
    # Format datasets list
    datasets_str = ", ".join(item['datasets']) if item['datasets'] else "not specified"
    
    # Format improvements list
    improvements_str = "; ".join(item['improvements']) if item['improvements'] else "not specified"
    
    # Format hyperparameters
    hyperparams_str = _format_hyperparams(item.get('hyperparameters', {}))
    
    # Format dimensions
    dimensions_str = _format_dimensions(item.get('dimensions', {}))
    
    # Format ablations
    ablations_str = "\n• ".join(item.get('ablations', [])) if item.get('ablations') else "not reported"

    if level in ("beginner", "intermediate", "expert"):
        context = _build_explanation_context(paper_id, item)

        if level == "beginner":
            prompt = _prompt_beginner(context)
        elif level == "intermediate":
            prompt = _prompt_intermediate(context, item)
        else:  # expert
            prompt = _prompt_expert(context, item)

        try:
            explanation = call_llm(
                prompt,
                system=_EXPLAIN_SYSTEM,
                max_tokens=700,
                temperature=0.3,
            )
            # Check if LLM returned stub message - treat as failure
            if "[LLM_UNAVAILABLE:" in explanation:
                raise Exception("LLM unavailable")
        except Exception as e:
            # Graceful fallback to template strings if LLM unavailable
            import logging
            logging.getLogger(__name__).warning(f"LLM explanation failed for {paper_id}: {e}")
            metrics_str = ", ".join(item.get('metrics', [])[:3]) or "no explicit metrics"
            if level == "beginner":
                explanation = (
                    f"This paper tries to solve {item['problem']}. "
                    f"It does this by using {item['core_technique']} in a simple pipeline. "
                    f"The main result says {item['results']}."
                )
            elif level == "intermediate":
                explanation = (
                    f"PROBLEM\n{item['problem']}\n\n"
                    f"METHOD\n{item['proposed_method']}\n\n"
                    f"RESULTS\n{item['results']}\nMetrics: {metrics_str}\n\n"
                    f"LIMITATIONS\n{item['limitations']}"
                )
            else:
                explanation = (
                    f"ARCHITECTURE & DESIGN\n"
                    f"• Architecture: {item['architecture']}\n"
                    f"• Core Technique: {item['core_technique']}\n\n"
                    f"TRAINING STRATEGY\n"
                    f"• Learning Strategy: {item['learning_strategy']}\n\n"
                    f"EMPIRICAL EVIDENCE\n"
                    f"• Results: {item['results']}\n"
                    f"• Metrics: {metrics_str}"
                )

        return {
            "paper_id": paper_id,
            "paper_name": item["title"],
            "level": level,
            "explanation": explanation,
            "diagram": None,
        }
    
    # Visual level with enhanced diagrams
    if level == "visual":
        dimensions = item.get('dimensions', {})
        diagram = _diagram_with_dimensions(item["core_technique"], item["architecture"], dimensions)
        
        explanation = (
            f"Visual Architecture Flow\n\n"
            f"Technique: {item['core_technique']}\n"
            f"Architecture: {item['architecture']}\n"
            f"Model Size: {dimensions_str}"
        )
        
        return {
            "paper_id": paper_id,
            "paper_name": item["title"],
            "level": "visual",
            "explanation": explanation,
            "diagram": diagram,
        }
    
    # Training vs Inference view
    if level == "training":
        hyperparams = item.get('hyperparameters', {})
        dimensions = item.get('dimensions', {})
        diagram = _training_vs_inference_view(item["core_technique"], dimensions, hyperparams)
        
        explanation = (
            f"Training vs Inference Comparison\n\n"
            f"Training Configuration:\n"
            f"• {hyperparams_str}\n\n"
            f"Model Configuration:\n"
            f"• {dimensions_str}"
        )
        
        return {
            "paper_id": paper_id,
            "paper_name": item["title"],
            "level": "training",
            "explanation": explanation,
            "diagram": diagram,
        }
    
    # Pipeline inference with detailed dimensions
    if level == "pipeline":
        dimensions = item.get('dimensions', {})
        technique = item.get('core_technique', 'transformer')
        
        try:
            dim_inference = DimensionInference(dimensions, technique)
            pipeline = dim_inference.infer_pipeline()
            diagram = dim_inference.format_pipeline_diagram(pipeline)
            
            explanation = (
                f"Pipeline Dimension Inference\n\n"
                f"This shows the exact tensor shapes and parameter counts at each layer.\n"
                f"Technique: {technique}\n"
                f"Architecture: {item['architecture']}\n"
                f"Total Layers: {len(pipeline)}"
            )
            
            return {
                "paper_id": paper_id,
                "paper_name": item["title"],
                "level": "pipeline",
                "explanation": explanation,
                "diagram": diagram,
            }
        except Exception as e:
            return {
                "paper_id": paper_id,
                "paper_name": item["title"],
                "level": "pipeline",
                "explanation": f"Pipeline inference failed: {str(e)}",
                "diagram": None,
            }
    
    # Component breakdown with actual data
    if level == "components":
        components = create_component_breakdown(item)
        diagram = format_component_breakdown(components)
        
        explanation = (
            f"Component Breakdown\n\n"
            f"This shows all extracted components with actual paper data.\n"
            f"Architecture: {item['architecture']}\n"
            f"Technique: {item['core_technique']}"
        )
        
        return {
            "paper_id": paper_id,
            "paper_name": item["title"],
            "level": "components",
            "explanation": explanation,
            "diagram": diagram,
        }
    
    # Default: return visual
    dimensions = item.get('dimensions', {})
    return {
        "paper_id": paper_id,
        "paper_name": item["title"],
        "level": "visual",
        "explanation": f"Visual Architecture Flow\n\nTechnique: {item['core_technique']}\nArchitecture: {item['architecture']}\nModel Size: {dimensions_str}",
        "diagram": _diagram_with_dimensions(item["core_technique"], item["architecture"], dimensions),
    }


def explain(paper_ids: list[str] | str, level: str) -> dict[str, object]:
    """
    Explain multiple papers at the specified level.
    
    Args:
        paper_ids: List of paper IDs to explain, or a single paper ID string (for backward compatibility)
        level: Explanation level (beginner, intermediate, expert, visual, training, pipeline, components)
    
    Returns:
        Dictionary containing explanations for all papers, or single explanation dict if input was a string
    """
    # Backward compatibility: accept single paper ID as string
    if isinstance(paper_ids, str):
        return _explain_single(paper_ids, level)
    
    if not paper_ids:
        return {"explanations": [], "level": level}
    
    explanations = []
    for paper_id in paper_ids:
        explanation = _explain_single(paper_id, level)
        explanations.append(explanation)
    
    return {
        "explanations": explanations,
        "level": level,
        "count": len(explanations),
    }
