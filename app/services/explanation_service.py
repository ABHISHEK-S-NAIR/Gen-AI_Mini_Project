from app.services.structured_extraction_service import extract_structured_for_papers


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
    """Generate ASCII diagram based on technique and architecture."""
    t = technique.lower()
    a = architecture.lower()
    
    # Transformer-based architectures
    if "self-attention" in t or "transformer" in t:
        return """[Input Sequence]
      ↓
┌─────────────────┐
│ Token Embedding │
│ + Positional    │
└─────────────────┘
      ↓
┌─────────────────┐
│ Multi-Head      │
│ Self-Attention  │
└─────────────────┘
      ↓
┌─────────────────┐
│ Feed-Forward    │
│ Network         │
└─────────────────┘
      ↓
[Output / Predictions]"""
    
    # Retrieval-augmented generation
    if "retrieval" in t:
        return """[Query]
   ↓
┌──────────────┐
│  Retriever   │ ←─── [Document Store]
└──────────────┘
   ↓
[Retrieved Context]
   ↓
┌──────────────┐
│  Generator   │
│  (LM/Model)  │
└──────────────┘
   ↓
[Grounded Output]"""
    
    # Masked Language Model (BERT-style)
    if "mlm" in t or "bert" in t:
        return """[Input Text]
      ↓
┌─────────────────┐
│ Tokenization    │
│ + [MASK] tokens │
└─────────────────┘
      ↓
┌─────────────────┐
│ Bidirectional   │
│ Encoder Layers  │
└─────────────────┘
      ↓
┌─────────────────┐
│ MLM Head        │
│ Predict [MASK]  │
└─────────────────┘
      ↓
[Fine-tuned Model]"""
    
    # CNN architectures
    if "cnn" in t or "convolutional" in t or "resnet" in t or "conv" in a:
        return """[Input Image]
      ↓
┌─────────────────┐
│ Convolutional   │
│ Layers + Pool   │
└─────────────────┘
      ↓
┌─────────────────┐
│ Feature Maps    │
│ (Deep Layers)   │
└─────────────────┘
      ↓
┌─────────────────┐
│ Classification  │
│ / Output Head   │
└─────────────────┘
      ↓
[Predictions]"""
    
    # GAN architectures
    if "gan" in t or "generative adversarial" in t:
        return """[Random Noise z]        [Real Data]
      ↓                    ↓
┌─────────────┐     ┌─────────────┐
│  Generator  │     │ Discriminator│
│   G(z)      │     │   D(x)      │
└─────────────┘     └─────────────┘
      ↓                    ↑
[Fake Data] ───────────────┘
      ↓
[Adversarial Training Loop]
  G tries to fool D
  D tries to detect fakes"""
    
    # Encoder-Decoder (Seq2Seq)
    if "encoder-decoder" in t or "seq2seq" in t or "sequence-to-sequence" in t:
        return """[Input Sequence]
      ↓
┌─────────────────┐
│    Encoder      │
│  (e.g., LSTM)   │
└─────────────────┘
      ↓
[Context Vector]
      ↓
┌─────────────────┐
│    Decoder      │
│  (e.g., LSTM)   │
└─────────────────┘
      ↓
[Output Sequence]"""
    
    # Graph Neural Networks
    if "graph" in t or "gnn" in t:
        return """[Graph Structure]
  (Nodes + Edges)
      ↓
┌─────────────────┐
│ Node Feature    │
│ Aggregation     │
└─────────────────┘
      ↓
┌─────────────────┐
│ Message Passing │
│ (k iterations)  │
└─────────────────┘
      ↓
[Node/Graph Embedding]"""
    
    # Contrastive Learning
    if "contrastive" in t or "simclr" in t or "clip" in t:
        return """[Anchor Sample]    [Positive]    [Negative]
      ↓              ↓             ↓
┌──────────────────────────────────────┐
│         Encoder Network              │
└──────────────────────────────────────┘
      ↓              ↓             ↓
[Embedding Space - Pull similar close,
                   Push different apart]"""
    
    # Default fallback
    return """[Input Data]
      ↓
┌─────────────────┐
│ Feature         │
│ Extraction      │
└─────────────────┘
      ↓
┌─────────────────┐
│ Task-Specific   │
│ Processing      │
└─────────────────┘
      ↓
[Output/Prediction]"""


def explain(paper_id: str, level: str) -> dict[str, object]:
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

    beginner = (
        f"This paper tries to solve {item['problem']}. "
        f"It does this by using {item['core_technique']} in a simple pipeline. "
        f"The main result says {item['results']}."
    )

    intermediate = (
        f"📋 PROBLEM\n"
        f"{item['problem']}\n\n"
        f"🔧 PROPOSED SOLUTION\n"
        f"{item['proposed_method']}\n\n"
        f"⚙️ HOW IT WORKS\n"
        f"Technique: {item['core_technique']}\n"
        f"Training: {item['learning_strategy']}\n"
        f"Architecture: {item['architecture']}\n\n"
        f"📊 RESULTS\n"
        f"{item['results']}\n"
        f"Metrics: {metrics_str}\n"
        f"Improvements: {improvements_str}\n\n"
        f"🗂️ DATASETS USED\n"
        f"{datasets_str}\n\n"
        f"💡 KEY INNOVATION\n"
        f"{item['novelty']}\n\n"
        f"⚠️ LIMITATIONS\n"
        f"{item['limitations']}"
    )

    expert = (
        f"ARCHITECTURE & DESIGN\n"
        f"• Base Architecture: {item['architecture']}\n"
        f"• Core Technique: {item['core_technique']}\n"
        f"• Contribution Type: {item['contribution_type']}\n"
        f"• Model Dimensions: {dimensions_str}\n\n"
        f"TRAINING STRATEGY\n"
        f"• Learning Strategy: {item['learning_strategy']}\n"
        f"• Hyperparameters: {hyperparams_str}\n"
        f"• Datasets: {datasets_str}\n\n"
        f"KEY INNOVATION\n"
        f"• {item['novelty']}\n\n"
        f"EMPIRICAL RESULTS\n"
        f"• Results: {item['results']}\n"
        f"• Metrics: {metrics_str}\n"
        f"• Performance Gains: {improvements_str}\n\n"
        f"ABLATION STUDIES\n"
        f"• {ablations_str}\n\n"
        f"LIMITATIONS & CONSTRAINTS\n"
        f"• {item['limitations']}"
    )

    if level == "beginner":
        return {"paper_id": paper_id, "paper_name": item["title"], "level": "beginner", "explanation": beginner, "diagram": None}
    
    if level == "intermediate":
        return {
            "paper_id": paper_id,
            "paper_name": item["title"],
            "level": "intermediate",
            "explanation": intermediate,
            "diagram": None,
        }
    
    if level == "expert":
        return {"paper_id": paper_id, "paper_name": item["title"], "level": "expert", "explanation": expert, "diagram": None}
    
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
    
    # Training vs Inference view (new level)
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
    
    # Default: return visual
    dimensions = item.get('dimensions', {})
    return {
        "paper_id": paper_id,
        "paper_name": item["title"],
        "level": "visual",
        "explanation": f"Visual Architecture Flow\n\nTechnique: {item['core_technique']}\nArchitecture: {item['architecture']}\nModel Size: {dimensions_str}",
        "diagram": _diagram_with_dimensions(item["core_technique"], item["architecture"], dimensions),
    }
