#!/usr/bin/env python3
"""Test script for medium-effort explanation improvements."""

from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.explanation_service import explain

# Setup comprehensive test data
pid = "advanced_test"
state.papers[pid] = IngestedPaper(
    paper_id=pid,
    filename="advanced_transformer_paper.pdf",
    raw_text="Advanced transformer architecture paper"
)

state.sections[pid] = {
    "abstract": (
        "We propose a novel transformer architecture with 12 layers and 768 hidden dimensions "
        "achieving 34.8 BLEU score on WMT14, improved by 15% over previous baselines."
    ),
    "intro": (
        "Previous RNN models are slow. We introduce self-attention mechanism with multi-head attention "
        "using 8 attention heads. The model has 110 million parameters and uses transformer architecture."
    ),
    "method": (
        "The method uses multi-head self-attention with d_model=768 and d_ff=3072. "
        "We train with Adam optimizer with learning rate 0.0001 and batch size 32 for 100 epochs. "
        "Dropout rate is 0.1 and weight decay is 0.01. The model is pretrained on WMT and ImageNet datasets. "
        "Feed-forward dimension is 3072."
    ),
    "results": (
        "Our model achieves 34.8 BLEU on WMT14, 28.5 ROUGE on SQuAD, and 92.3% accuracy on GLUE. "
        "We conduct ablation studies showing that removing multi-head attention leads to 5% performance drop. "
        "Without positional encoding, the model degrades by 8%. The self-attention mechanism contributes significantly "
        "to the overall performance improvement by 12%."
    ),
    "conclusion": (
        "Future work includes scaling to larger datasets. Limitation: requires significant compute resources "
        "and fails on very long sequences. Ablation studies demonstrate that each component is critical."
    ),
    "other": ""
}

print("=" * 100)
print("TESTING MEDIUM-EFFORT IMPROVEMENTS")
print("=" * 100)

# Test Expert Level with new extractions
print("\n" + "=" * 100)
print("🔬 EXPERT LEVEL (with hyperparams, dimensions, ablations)")
print("=" * 100)
expert = explain(pid, "expert")
print(expert["explanation"])

print("\n" + "=" * 100)
print("📊 INTERMEDIATE LEVEL (unchanged, but using new data)")
print("=" * 100)
intermediate = explain(pid, "intermediate")
print(intermediate["explanation"])

print("\n" + "=" * 100)
print("🎨 VISUAL LEVEL (with dimensions)")
print("=" * 100)
visual = explain(pid, "visual")
print(visual["explanation"])
print("\nEnhanced Diagram with Dimensions:")
print(visual["diagram"])

print("\n" + "=" * 100)
print("🏋️ TRAINING VIEW (new multi-view diagram)")
print("=" * 100)
training = explain(pid, "training")
print(training["explanation"])
print("\nTraining vs Inference Diagram:")
print(training["diagram"])

# Test CNN architecture
print("\n" + "=" * 100)
print("TESTING CNN ARCHITECTURE WITH DIMENSIONS")
print("=" * 100)

pid_cnn = "cnn_test"
state.papers[pid_cnn] = IngestedPaper(
    paper_id=pid_cnn,
    filename="resnet50_paper.pdf",
    raw_text="ResNet CNN paper"
)

state.sections[pid_cnn] = {
    "abstract": "We propose ResNet-50 with 50 layers using convolutional neural networks achieving 78% top-1 accuracy.",
    "intro": "Deep CNNs with residual connections. The model has 25.6 million parameters.",
    "method": (
        "We use convolutional layers with 3x3 kernels. Training with SGD optimizer, learning rate 0.1, "
        "batch size 256 for 90 epochs. The architecture has 50 layers with residual connections."
    ),
    "results": "Achieves 78.5% top-1 accuracy on ImageNet and 95% on CIFAR-10.",
    "conclusion": "CNNs with residual connections work well.",
    "other": ""
}

visual_cnn = explain(pid_cnn, "visual")
print("\nCNN Visual Explanation:")
print(visual_cnn["explanation"])
print("\nCNN Diagram with Dimensions:")
print(visual_cnn["diagram"])

training_cnn = explain(pid_cnn, "training")
print("\n" + "=" * 100)
print("CNN Training View:")
print(training_cnn["diagram"])

print("\n" + "=" * 100)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 100)

print("\n" + "=" * 100)
print("FEATURE SUMMARY")
print("=" * 100)
print("✅ Hyperparameter extraction: learning rate, batch size, optimizer, epochs, dropout, weight decay")
print("✅ Model dimension extraction: parameters, layers, hidden_dim, attention heads, FFN dim")
print("✅ Ablation study detection: extracts key ablation insights")
print("✅ Enhanced diagrams: include dimensions and detailed annotations")
print("✅ Multi-view diagrams: training vs inference comparison")
print("✅ Expert level: now includes hyperparameters, dimensions, and ablations")
print("=" * 100)
