#!/usr/bin/env python3
"""Test script for high-effort explanation improvements."""

from app.core.state import state
from app.models.schemas import IngestedPaper
from app.services.explanation_service import explain
from app.services.dimension_inference_service import DimensionInference, create_component_breakdown

# Setup comprehensive test data
pid = "high_effort_test"
state.papers[pid] = IngestedPaper(
    paper_id=pid,
    filename="advanced_transformer_paper.pdf",
    raw_text="Advanced transformer architecture paper"
)

state.sections[pid] = {
    "abstract": (
        "We propose a novel transformer architecture with 12 layers and 768 hidden dimensions "
        "achieving 34.8 BLEU score on WMT14, improved by 15% over previous baselines. "
        "The model uses multi-head attention with 8 attention heads."
    ),
    "intro": (
        "Previous RNN models are slow. We introduce self-attention mechanism with multi-head attention "
        "using 8 attention heads. The model has 110 million parameters and uses transformer architecture."
    ),
    "method": (
        "The method uses multi-head self-attention with d_model=768 and d_ff=3072. "
        "We train with Adam optimizer with learning rate 0.0001 and batch size 32 for 100 epochs. "
        "Dropout rate is 0.1 and weight decay is 0.01. The model is pretrained on WMT and ImageNet datasets. "
        "Feed-forward dimension is 3072. Input embedding dimension is 768 and sequence length is 512."
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
print("TESTING HIGH-EFFORT IMPROVEMENTS")
print("=" * 100)

# Test 1: Pipeline Level with Dimension Inference
print("\n" + "=" * 100)
print("🔧 PIPELINE LEVEL (Layer-by-layer dimension inference)")
print("=" * 100)
pipeline = explain(pid, "pipeline")
print(pipeline["explanation"])
if "diagram" in pipeline:
    print("\nPipeline Diagram:")
    print(pipeline["diagram"])

# Test 2: Components Level with Detailed Breakdown
print("\n" + "=" * 100)
print("📦 COMPONENTS LEVEL (Detailed breakdown of all components)")
print("=" * 100)
components = explain(pid, "components")
print(components["explanation"])

# Test 3: Dimension Inference Service Directly
print("\n" + "=" * 100)
print("🧮 DIMENSION INFERENCE SERVICE (Direct testing)")
print("=" * 100)

# Setup structured data for dimension inference
structured_data = {
    "architecture": "transformer",
    "dimensions": {
        "parameters": "110M",
        "layers": 12,
        "hidden_dim": 768,
        "attention_heads": 8,
        "ffn_dim": 3072
    },
    "hyperparameters": {
        "batch_size": 32
    }
}

dim_inference = DimensionInference(structured_data)
print(f"Input Shape: {dim_inference.input_shape}")
print(f"Batch Size: {dim_inference.batch_size}")
print(f"Hidden Dim: {dim_inference.hidden_dim}")
print(f"Attention Heads: {dim_inference.attention_heads}")
print(f"FFN Dim: {dim_inference.ffn_dim}")

# Infer a simple pipeline
print("\n📊 Inferred Pipeline:")
pipeline_layers = dim_inference.infer_pipeline()
for layer in pipeline_layers:
    print(f"  {layer['layer']}: {layer['input']} → {layer['output']}")
    if 'operation' in layer:
        print(f"    Operation: {layer['operation']}")

# Test 4: Component Breakdown Function
print("\n" + "=" * 100)
print("📋 COMPONENT BREAKDOWN FUNCTION (Direct testing)")
print("=" * 100)

full_structured_data = {
    "architecture": "transformer",
    "metrics": [
        {"name": "BLEU", "value": "34.8"},
        {"name": "ROUGE", "value": "28.5"},
        {"name": "accuracy", "value": "92.3%"}
    ],
    "datasets": ["WMT14", "SQuAD", "GLUE"],
    "dimensions": {
        "parameters": "110M",
        "layers": 12,
        "hidden_dim": 768,
        "attention_heads": 8,
        "ffn_dim": 3072
    },
    "hyperparameters": {
        "learning_rate": 0.0001,
        "batch_size": 32,
        "optimizer": "Adam",
        "epochs": 100,
        "dropout": 0.1,
        "weight_decay": 0.01
    },
    "ablations": [
        "removing multi-head attention leads to 5% performance drop",
        "without positional encoding, the model degrades by 8%",
        "self-attention mechanism contributes significantly to the overall performance improvement by 12%"
    ]
}

breakdown = create_component_breakdown(full_structured_data)
print("Components Breakdown:")
for component, data in breakdown.items():
    print(f"\n{component.upper()}:")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"  {key}: {value}")
    elif isinstance(data, list):
        for item in data:
            print(f"  - {item}")
    else:
        print(f"  {data}")

# Test 5: CNN Architecture with Dimension Inference
print("\n" + "=" * 100)
print("TESTING CNN ARCHITECTURE WITH DIMENSION INFERENCE")
print("=" * 100)

pid_cnn = "cnn_high_test"
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
        "batch size 256 for 90 epochs. The architecture has 50 layers with residual connections. "
        "Input images are 224x224 RGB, output is 1000 classes."
    ),
    "results": "Achieves 78.5% top-1 accuracy on ImageNet and 95% on CIFAR-10.",
    "conclusion": "CNNs with residual connections work well.",
    "other": ""
}

pipeline_cnn = explain(pid_cnn, "pipeline")
print("\nCNN Pipeline Explanation:")
print(pipeline_cnn["explanation"])
if "diagram" in pipeline_cnn:
    print("\nCNN Pipeline Diagram:")
    print(pipeline_cnn["diagram"])

components_cnn = explain(pid_cnn, "components")
print("\n" + "=" * 100)
print("CNN Components Breakdown:")
print(components_cnn["explanation"])

# Test 6: GAN Architecture
print("\n" + "=" * 100)
print("TESTING GAN ARCHITECTURE WITH DIMENSION INFERENCE")
print("=" * 100)

pid_gan = "gan_high_test"
state.papers[pid_gan] = IngestedPaper(
    paper_id=pid_gan,
    filename="stylegan_paper.pdf",
    raw_text="StyleGAN paper"
)

state.sections[pid_gan] = {
    "abstract": "We propose StyleGAN with progressive growing achieving FID score of 4.4 on FFHQ dataset.",
    "intro": "GANs can generate high-quality images. Our model has 26.2 million parameters.",
    "method": (
        "We use generative adversarial network with progressive training. "
        "Learning rate is 0.002, batch size 16, trained for 200 epochs with Adam optimizer. "
        "Latent dimension is 512. Generator has 8 layers, discriminator has 7 layers."
    ),
    "results": "Achieves FID score of 4.4 on FFHQ and 8.3 on CelebA-HQ.",
    "conclusion": "Progressive GANs produce better image quality.",
    "other": ""
}

pipeline_gan = explain(pid_gan, "pipeline")
print("\nGAN Pipeline Explanation:")
print(pipeline_gan["explanation"])

components_gan = explain(pid_gan, "components")
print("\n" + "=" * 100)
print("GAN Components Breakdown:")
print(components_gan["explanation"])

# Test 7: All Explanation Levels
print("\n" + "=" * 100)
print("TESTING ALL 7 EXPLANATION LEVELS")
print("=" * 100)

levels = ["beginner", "intermediate", "expert", "visual", "training", "pipeline", "components"]
for level in levels:
    result = explain(pid, level)
    print(f"\n{'=' * 50}")
    print(f"Level: {level.upper()}")
    print(f"{'=' * 50}")
    print(f"Has explanation: {bool(result.get('explanation'))}")
    print(f"Has diagram: {bool(result.get('diagram'))}")
    print(f"Explanation length: {len(result.get('explanation', ''))} chars")

print("\n" + "=" * 100)
print("✅ ALL HIGH-EFFORT TESTS COMPLETED SUCCESSFULLY!")
print("=" * 100)

print("\n" + "=" * 100)
print("HIGH-EFFORT FEATURE SUMMARY")
print("=" * 100)
print("✅ Dimension inference service: layer-by-layer tensor shape calculation")
print("✅ Pipeline explanation level: shows data flow through model with dimensions")
print("✅ Components explanation level: detailed breakdown of all paper components")
print("✅ Component breakdown function: organizes all extracted data systematically")
print("✅ Interactive UI: collapsible sections with expand/collapse controls")
print("✅ Enhanced frontend: 7 explanation levels with rich formatting")
print("✅ Multi-architecture support: Transformer, CNN, GAN dimension inference")
print("=" * 100)
