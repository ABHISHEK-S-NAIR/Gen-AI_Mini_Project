#!/usr/bin/env python3
"""Quick test script to verify explanation improvements."""

from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.explanation_service import explain

# Setup test data
pid = "test_paper_1"
state.papers[pid] = IngestedPaper(
    paper_id=pid,
    filename="sample_transformer_paper.pdf",
    raw_text="Sample paper about transformers"
)

state.sections[pid] = {
    "abstract": "We propose a novel transformer architecture for neural machine translation achieving 34.8 BLEU score.",
    "intro": "Previous RNN models are slow. We introduce attention mechanism with 12 layers.",
    "method": "The method uses multi-head self-attention with 8 attention heads. We pretrain on WMT dataset with 40M parameters.",
    "results": "Our model achieves 34.8 BLEU on WMT14 and 28.5 BLEU on IWSLT, improved by 15% over baseline. The accuracy is 92.3%.",
    "conclusion": "Future work includes scaling to larger datasets. Limitation: requires significant compute resources.",
    "other": ""
}

print("=" * 80)
print("TESTING EXPLANATION IMPROVEMENTS")
print("=" * 80)

# Test Intermediate Level
print("\n📝 INTERMEDIATE LEVEL:")
print("-" * 80)
intermediate = explain(pid, "intermediate")
print(intermediate["explanation"])

print("\n" + "=" * 80)

# Test Expert Level
print("\n🔬 EXPERT LEVEL:")
print("-" * 80)
expert = explain(pid, "expert")
print(expert["explanation"])

print("\n" + "=" * 80)

# Test Visual Level
print("\n🎨 VISUAL LEVEL:")
print("-" * 80)
visual = explain(pid, "visual")
print(visual["explanation"])
print("\nDiagram:")
print(visual["diagram"])

print("\n" + "=" * 80)
print("✅ All explanation levels generated successfully!")
print("=" * 80)

# Test with CNN paper
print("\n\n" + "=" * 80)
print("TESTING CNN ARCHITECTURE DIAGRAM")
print("=" * 80)

pid2 = "test_paper_2"
state.papers[pid2] = IngestedPaper(
    paper_id=pid2,
    filename="resnet_paper.pdf",
    raw_text="ResNet paper"
)

state.sections[pid2] = {
    "abstract": "We propose ResNet using convolutional neural networks.",
    "intro": "Deep CNNs are powerful. We use residual connections.",
    "method": "The method uses convolutional layers with skip connections tested on ImageNet and CIFAR-10.",
    "results": "Achieves 78.5% top-1 accuracy on ImageNet.",
    "conclusion": "CNNs with residual connections work well.",
    "other": ""
}

visual_cnn = explain(pid2, "visual")
print("\nCNN Diagram:")
print(visual_cnn["diagram"])

print("\n" + "=" * 80)
print("✅ CNN diagram generated successfully!")
print("=" * 80)
