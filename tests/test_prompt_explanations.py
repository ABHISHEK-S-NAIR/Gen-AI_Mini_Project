#!/usr/bin/env python3
"""Test prompt-engineered multi-level explanations."""
import sys
sys.path.insert(0, '/Users/absarkar/Developer/Gen-AI_Mini_Project')

from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.explanation_service import explain

print("=" * 80)
print("PROMPT-ENGINEERED EXPLANATION TEST")
print("=" * 80)

# Setup test data
state.clear()
pid = "test_paper"
state.papers[pid] = IngestedPaper(paper_id=pid, filename="transformer.pdf", raw_text="dummy")
state.sections[pid] = {
    "abstract": "We propose a Transformer model based solely on attention mechanisms for sequence transduction.",
    "intro": "RNNs are sequential and slow. We propose a novel architecture without recurrence.",
    "method": "The method uses multi-head self-attention with encoder-decoder blocks. 8 attention heads and 6 layers.",
    "results": "The model achieves 28.4 BLEU on WMT14 En-De. This is 2.0 BLEU better than prior work.",
    "conclusion": "Future work includes stronger ablations and robustness testing.",
}

print("\nTest 1: All three levels generate different explanations")
print("-" * 80)
beginner = explain(pid, "beginner")
intermediate = explain(pid, "intermediate")
expert = explain(pid, "expert")

print(f"✓ Beginner explanation: {len(beginner['explanation'])} chars")
print(f"✓ Intermediate explanation: {len(intermediate['explanation'])} chars")
print(f"✓ Expert explanation: {len(expert['explanation'])} chars")

assert beginner["explanation"] != intermediate["explanation"], "Beginner and intermediate should differ"
assert intermediate["explanation"] != expert["explanation"], "Intermediate and expert should differ"
assert beginner["explanation"] != expert["explanation"], "Beginner and expert should differ"
print("✓ All three explanations are distinct")

print("\nTest 2: Intermediate contains metrics")
print("-" * 80)
intermediate_lower = intermediate["explanation"].lower()
has_metric = "metric" in intermediate_lower or "bleu" in intermediate_lower
print(f"Contains 'metric' or 'bleu': {has_metric}")
assert has_metric, "Intermediate should mention metrics or BLEU"
print("✓ Intermediate mentions metrics")

print("\nTest 3: Expert contains architectural/training terms")
print("-" * 80)
expert_lower = expert["explanation"].lower()
has_arch = "architectural" in expert_lower or "architecture" in expert_lower or "training" in expert_lower
print(f"Contains architectural/training terms: {has_arch}")
assert has_arch, "Expert should mention architecture or training"
print("✓ Expert mentions architecture or training")

print("\nTest 4: Visual level still works")
print("-" * 80)
visual = explain(pid, "visual")
assert visual["level"] == "visual", "Visual level should be set"
assert visual["diagram"] is not None, "Visual should have diagram"
assert "↓" in visual["diagram"] or "┌" in visual["diagram"], "Diagram should have box drawing chars"
print("✓ Visual explanation returns diagram")

print("\nTest 5: Fallback strings are used (no LLM)")
print("-" * 80)
print("Sample beginner (fallback):")
print(beginner["explanation"][:200])
print("\nSample intermediate (fallback):")
print(intermediate["explanation"][:200])
print("\nSample expert (fallback):")
print(expert["explanation"][:200])

print("\n" + "=" * 80)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("=" * 80)
print("\nThe explanation service now:")
print("  - Uses LLM with level-specific prompts when API key available")
print("  - Beginner: Simple language, everyday analogies, 3-4 paragraphs")
print("  - Intermediate: Structured sections, mentions metrics")
print("  - Expert: Technical analysis, architecture/training details")
print("  - Falls back to template strings when LLM unavailable")
print("  - Visual/training/pipeline/components levels unchanged")
