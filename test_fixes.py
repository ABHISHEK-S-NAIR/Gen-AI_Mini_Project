#!/usr/bin/env python3
"""Quick test to verify training, pipeline, and components levels work."""

from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.explanation_service import explain

# Setup test data
pid = "fix_test"
state.papers[pid] = IngestedPaper(
    paper_id=pid,
    filename="test_paper.pdf",
    raw_text="Test transformer paper"
)

state.sections[pid] = {
    "abstract": (
        "We propose a transformer with 12 layers and 768 hidden dimensions "
        "achieving 34.8 BLEU score on WMT14. The model uses 8 attention heads."
    ),
    "intro": "We introduce self-attention with 110 million parameters.",
    "method": (
        "We use multi-head attention with d_model=768 and d_ff=3072. "
        "Training with Adam optimizer, learning rate 0.0001, batch size 32, 100 epochs. "
        "Dropout 0.1, weight decay 0.01."
    ),
    "results": "Achieves 34.8 BLEU on WMT14, 28.5 ROUGE on SQuAD.",
    "conclusion": "Future work includes scaling.",
    "other": ""
}

print("=" * 80)
print("TESTING FIXED EXPLANATION LEVELS")
print("=" * 80)

# Test Training Level
print("\n" + "=" * 80)
print("1. TRAINING LEVEL")
print("=" * 80)
try:
    training = explain(pid, "training")
    print("✓ Training level works!")
    print(f"  Explanation length: {len(training.get('explanation', ''))} chars")
    print(f"  Has diagram: {bool(training.get('diagram'))}")
    if training.get('diagram'):
        print("\nDiagram preview (first 200 chars):")
        print(training['diagram'][:200] + "...")
except Exception as e:
    print(f"✗ Training level FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test Pipeline Level
print("\n" + "=" * 80)
print("2. PIPELINE LEVEL")
print("=" * 80)
try:
    pipeline = explain(pid, "pipeline")
    print("✓ Pipeline level works!")
    print(f"  Explanation length: {len(pipeline.get('explanation', ''))} chars")
    print(f"  Has diagram: {bool(pipeline.get('diagram'))}")
    if pipeline.get('explanation'):
        print("\nExplanation:")
        print(pipeline['explanation'])
    if pipeline.get('diagram'):
        print("\nDiagram preview (first 300 chars):")
        print(pipeline['diagram'][:300] + "...")
except Exception as e:
    print(f"✗ Pipeline level FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test Components Level
print("\n" + "=" * 80)
print("3. COMPONENTS LEVEL")
print("=" * 80)
try:
    components = explain(pid, "components")
    print("✓ Components level works!")
    print(f"  Explanation length: {len(components.get('explanation', ''))} chars")
    print(f"  Has diagram: {bool(components.get('diagram'))}")
    if components.get('explanation'):
        print("\nExplanation:")
        print(components['explanation'])
    if components.get('diagram'):
        print("\nDiagram preview (first 300 chars):")
        print(components['diagram'][:300] + "...")
except Exception as e:
    print(f"✗ Components level FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("If all three levels show '✓', then the fixes are working correctly!")
print("=" * 80)
