from app.services.structured_extraction_service import extract_structured_for_papers


def _diagram_for(technique: str, architecture: str = "") -> str:
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
        f"• Contribution Type: {item['contribution_type']}\n\n"
        f"TRAINING STRATEGY\n"
        f"• Learning Strategy: {item['learning_strategy']}\n"
        f"• Datasets: {datasets_str}\n\n"
        f"KEY INNOVATION\n"
        f"• {item['novelty']}\n\n"
        f"EMPIRICAL RESULTS\n"
        f"• Results: {item['results']}\n"
        f"• Metrics: {metrics_str}\n"
        f"• Performance Gains: {improvements_str}\n\n"
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

    return {
        "paper_id": paper_id,
        "paper_name": item["title"],
        "level": "visual",
        "explanation": f"Visual Architecture Flow\n\nTechnique: {item['core_technique']}\nArchitecture: {item['architecture']}",
        "diagram": _diagram_for(item["core_technique"], item["architecture"]),
    }
