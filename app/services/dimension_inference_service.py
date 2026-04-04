"""
Dimension Inference Service

This module provides advanced dimension inference through neural network pipelines,
calculating tensor shapes at each layer based on extracted architecture specifications.
"""

import re
from typing import Any


class DimensionInference:
    """Infers tensor dimensions through a neural network pipeline."""
    
    def __init__(self, dimensions: dict[str, str], technique: str):
        """Initialize with extracted dimensions and technique type."""
        self.dimensions = dimensions
        self.technique = technique.lower()
        
        # Parse dimensions safely
        def parse_int(value: str | None, default: int) -> int:
            """Safely parse string to int with fallback."""
            if not value:
                return default
            try:
                # Remove any non-numeric characters except digits
                clean = re.sub(r'[^\d]', '', str(value))
                return int(clean) if clean else default
            except (ValueError, AttributeError):
                return default
        
        self.hidden_dim = parse_int(dimensions.get("hidden_dim"), 768)
        self.num_layers = parse_int(dimensions.get("num_layers") or dimensions.get("layers"), 12)
        self.num_heads = parse_int(dimensions.get("num_heads") or dimensions.get("attention_heads"), 8)
        self.ffn_dim = parse_int(dimensions.get("ffn_dim"), self.hidden_dim * 4)
        self.params = dimensions.get("parameters", "unknown")
    
    def infer_transformer_pipeline(self, seq_len: int = 512, batch_size: int = 32) -> list[dict[str, Any]]:
        """Infer dimensions through a transformer pipeline."""
        pipeline = []
        
        # Input embedding
        pipeline.append({
            "layer": "Input",
            "operation": "Raw input tokens",
            "input_shape": f"({batch_size}, {seq_len})",
            "output_shape": f"({batch_size}, {seq_len})",
            "params": 0,
            "description": "Integer token IDs"
        })
        
        # Token + Positional Embedding
        vocab_size = 50000  # Typical vocab size
        embed_params = vocab_size * self.hidden_dim + seq_len * self.hidden_dim
        pipeline.append({
            "layer": "Embedding",
            "operation": "Token + Positional Embedding",
            "input_shape": f"({batch_size}, {seq_len})",
            "output_shape": f"({batch_size}, {seq_len}, {self.hidden_dim})",
            "params": embed_params,
            "description": f"Vocab: {vocab_size}, d_model: {self.hidden_dim}"
        })
        
        current_shape = f"({batch_size}, {seq_len}, {self.hidden_dim})"
        
        # Transformer layers
        for layer_idx in range(self.num_layers):
            # Multi-Head Attention
            if self.num_heads:
                head_dim = self.hidden_dim // self.num_heads
                # Q, K, V projections + output projection
                attn_params = 4 * (self.hidden_dim * self.hidden_dim)
                pipeline.append({
                    "layer": f"Layer {layer_idx + 1}: Multi-Head Attention",
                    "operation": f"Q, K, V computation ({self.num_heads} heads)",
                    "input_shape": current_shape,
                    "output_shape": current_shape,
                    "params": attn_params,
                    "description": f"Head dim: {head_dim}, Attention scores: ({batch_size}, {self.num_heads}, {seq_len}, {seq_len})"
                })
            
            # Feed-Forward Network
            ffn_params = (self.hidden_dim * self.ffn_dim) + (self.ffn_dim * self.hidden_dim)
            pipeline.append({
                "layer": f"Layer {layer_idx + 1}: Feed-Forward",
                "operation": "FFN(x) = W2(ReLU(W1(x)))",
                "input_shape": current_shape,
                "output_shape": current_shape,
                "params": ffn_params,
                "description": f"Intermediate: ({batch_size}, {seq_len}, {self.ffn_dim})"
            })
        
        # Output head (classification or LM head)
        output_params = self.hidden_dim * vocab_size
        pipeline.append({
            "layer": "Output Head",
            "operation": "Linear projection to vocabulary",
            "input_shape": current_shape,
            "output_shape": f"({batch_size}, {seq_len}, {vocab_size})",
            "params": output_params,
            "description": "Logits over vocabulary"
        })
        
        return pipeline
    
    def infer_cnn_pipeline(self, img_height: int = 224, img_width: int = 224, num_classes: int = 1000) -> list[dict[str, Any]]:
        """Infer dimensions through a CNN pipeline."""
        pipeline = []
        batch_size = 32
        
        # Input
        pipeline.append({
            "layer": "Input",
            "operation": "RGB Image",
            "input_shape": f"({batch_size}, 3, {img_height}, {img_width})",
            "output_shape": f"({batch_size}, 3, {img_height}, {img_width})",
            "params": 0,
            "description": "Standard ImageNet size"
        })
        
        current_h, current_w = img_height, img_width
        current_channels = 64
        
        # Initial conv
        params_conv1 = (3 * 7 * 7 * current_channels) + current_channels
        pipeline.append({
            "layer": "Conv1",
            "operation": "Conv2d(3→64, kernel=7, stride=2, padding=3)",
            "input_shape": f"({batch_size}, 3, {current_h}, {current_w})",
            "output_shape": f"({batch_size}, {current_channels}, {current_h//2}, {current_w//2})",
            "params": params_conv1,
            "description": "Initial feature extraction"
        })
        current_h, current_w = current_h // 2, current_w // 2
        
        # MaxPool
        pipeline.append({
            "layer": "MaxPool",
            "operation": "MaxPool2d(kernel=3, stride=2, padding=1)",
            "input_shape": f"({batch_size}, {current_channels}, {current_h}, {current_w})",
            "output_shape": f"({batch_size}, {current_channels}, {current_h//2}, {current_w//2})",
            "params": 0,
            "description": "Spatial downsampling"
        })
        current_h, current_w = current_h // 2, current_w // 2
        
        # Residual blocks (simplified)
        stages = [(64, 256), (128, 512), (256, 1024), (512, 2048)]
        for stage_idx, (in_ch, out_ch) in enumerate(stages):
            blocks_in_stage = [3, 4, 6, 3][stage_idx]  # ResNet-50 config
            
            for block_idx in range(blocks_in_stage):
                # Residual block params (simplified)
                block_params = (in_ch * 64) + (64 * 64) + (64 * out_ch)
                
                if block_idx == 0 and stage_idx > 0:
                    # Downsample
                    current_h, current_w = current_h // 2, current_w // 2
                
                pipeline.append({
                    "layer": f"Stage{stage_idx + 1}_Block{block_idx + 1}",
                    "operation": f"ResidualBlock({in_ch}→{out_ch})",
                    "input_shape": f"({batch_size}, {in_ch if block_idx == 0 else out_ch}, {current_h}, {current_w})",
                    "output_shape": f"({batch_size}, {out_ch}, {current_h}, {current_w})",
                    "params": block_params,
                    "description": "Conv→BN→ReLU→Conv→BN→Add"
                })
                in_ch = out_ch
        
        # Global Average Pooling
        pipeline.append({
            "layer": "Global Avg Pool",
            "operation": "AdaptiveAvgPool2d((1, 1))",
            "input_shape": f"({batch_size}, 2048, {current_h}, {current_w})",
            "output_shape": f"({batch_size}, 2048, 1, 1)",
            "params": 0,
            "description": "Spatial pooling to single vector"
        })
        
        # Flatten
        pipeline.append({
            "layer": "Flatten",
            "operation": "Reshape",
            "input_shape": f"({batch_size}, 2048, 1, 1)",
            "output_shape": f"({batch_size}, 2048)",
            "params": 0,
            "description": "Prepare for classification"
        })
        
        # Classification head
        fc_params = 2048 * num_classes + num_classes
        pipeline.append({
            "layer": "FC / Classifier",
            "operation": f"Linear(2048→{num_classes})",
            "input_shape": f"({batch_size}, 2048)",
            "output_shape": f"({batch_size}, {num_classes})",
            "params": fc_params,
            "description": "Final classification"
        })
        
        return pipeline
    
    def infer_pipeline(self) -> list[dict[str, Any]]:
        """Infer pipeline based on technique type."""
        if "transformer" in self.technique or "self-attention" in self.technique or "bert" in self.technique:
            return self.infer_transformer_pipeline()
        elif "cnn" in self.technique or "convolutional" in self.technique or "resnet" in self.technique:
            return self.infer_cnn_pipeline()
        else:
            # Generic pipeline
            return self.infer_transformer_pipeline()  # Default to transformer
    
    def format_pipeline_diagram(self, pipeline: list[dict[str, Any]]) -> str:
        """Format pipeline as detailed ASCII diagram with dimensions."""
        lines = []
        lines.append("=" * 80)
        lines.append("DETAILED PIPELINE WITH DIMENSION INFERENCE")
        lines.append("=" * 80)
        
        total_params = sum(layer["params"] for layer in pipeline)
        lines.append(f"\nTotal Parameters: {total_params:,}")
        lines.append(f"Number of Operations: {len(pipeline)}\n")
        
        for idx, layer in enumerate(pipeline, 1):
            lines.append(f"\n{'─' * 80}")
            lines.append(f"[{idx}] {layer['layer']}")
            lines.append(f"{'─' * 80}")
            lines.append(f"Operation: {layer['operation']}")
            lines.append(f"Input:     {layer['input_shape']}")
            lines.append(f"Output:    {layer['output_shape']}")
            lines.append(f"Params:    {layer['params']:,}")
            lines.append(f"Note:      {layer['description']}")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


def create_component_breakdown(item: dict[str, Any]) -> dict[str, Any]:
    """Create detailed component breakdown with actual extracted data."""
    
    # Calculate head_dim safely
    def calc_head_dim():
        try:
            hidden = item.get("dimensions", {}).get("hidden_dim")
            heads = item.get("dimensions", {}).get("num_heads")
            if hidden and heads:
                # Remove any non-numeric characters
                hidden_clean = re.sub(r'[^\d]', '', str(hidden))
                heads_clean = re.sub(r'[^\d]', '', str(heads))
                if hidden_clean and heads_clean:
                    return str(int(hidden_clean) // int(heads_clean))
        except (ValueError, ZeroDivisionError):
            pass
        return "N/A"
    
    components = {
        "architecture": {
            "type": item.get("architecture", "unknown"),
            "technique": item.get("core_technique", "unknown"),
            "layers": item.get("dimensions", {}).get("num_layers", "not specified"),
            "parameters": item.get("dimensions", {}).get("parameters", "not specified"),
        },
        "input_processing": {
            "embedding_dim": item.get("dimensions", {}).get("hidden_dim", "not specified"),
            "vocab_size": "extracted from paper if available",
            "sequence_length": "variable (paper dependent)",
        },
        "attention_mechanism": {
            "num_heads": item.get("dimensions", {}).get("num_heads", "not specified"),
            "head_dim": calc_head_dim(),
            "attention_type": "Multi-head" if item.get("dimensions", {}).get("num_heads") else "Single-head",
        },
        "feed_forward": {
            "hidden_dim": item.get("dimensions", {}).get("hidden_dim", "not specified"),
            "ffn_dim": item.get("dimensions", {}).get("ffn_dim", "not specified"),
            "activation": "typically ReLU or GELU",
        },
        "training_config": {
            "optimizer": item.get("hyperparameters", {}).get("optimizer", "not specified"),
            "learning_rate": item.get("hyperparameters", {}).get("learning_rate", "not specified"),
            "batch_size": item.get("hyperparameters", {}).get("batch_size", "not specified"),
            "epochs": item.get("hyperparameters", {}).get("epochs", "not specified"),
            "regularization": {
                "dropout": item.get("hyperparameters", {}).get("dropout", "not specified"),
                "weight_decay": item.get("hyperparameters", {}).get("weight_decay", "not specified"),
            },
        },
        "evaluation": {
            "datasets": item.get("datasets", []),
            "metrics": item.get("metrics", []),
            "improvements": item.get("improvements", []),
        },
        "ablations": {
            "studies_conducted": len(item.get("ablations", [])) > 0,
            "findings": item.get("ablations", []),
        },
    }
    
    return components


def format_component_breakdown(components: dict[str, Any]) -> str:
    """Format component breakdown as readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMPONENT BREAKDOWN WITH EXTRACTED DATA")
    lines.append("=" * 80)
    
    # Architecture
    lines.append("\n┌─ ARCHITECTURE OVERVIEW")
    lines.append(f"│  Type:       {components['architecture']['type']}")
    lines.append(f"│  Technique:  {components['architecture']['technique']}")
    lines.append(f"│  Layers:     {components['architecture']['layers']}")
    lines.append(f"│  Parameters: {components['architecture']['parameters']}")
    
    # Input Processing
    lines.append("\n┌─ INPUT PROCESSING")
    lines.append(f"│  Embedding Dimension: {components['input_processing']['embedding_dim']}")
    lines.append(f"│  Vocab Size:          {components['input_processing']['vocab_size']}")
    lines.append(f"│  Sequence Length:     {components['input_processing']['sequence_length']}")
    
    # Attention Mechanism
    lines.append("\n┌─ ATTENTION MECHANISM")
    lines.append(f"│  Number of Heads: {components['attention_mechanism']['num_heads']}")
    lines.append(f"│  Head Dimension:  {components['attention_mechanism']['head_dim']}")
    lines.append(f"│  Type:            {components['attention_mechanism']['attention_type']}")
    
    # Feed-Forward
    lines.append("\n┌─ FEED-FORWARD NETWORK")
    lines.append(f"│  Hidden Dim:  {components['feed_forward']['hidden_dim']}")
    lines.append(f"│  FFN Dim:     {components['feed_forward']['ffn_dim']}")
    lines.append(f"│  Activation:  {components['feed_forward']['activation']}")
    
    # Training Config
    lines.append("\n┌─ TRAINING CONFIGURATION")
    lines.append(f"│  Optimizer:     {components['training_config']['optimizer']}")
    lines.append(f"│  Learning Rate: {components['training_config']['learning_rate']}")
    lines.append(f"│  Batch Size:    {components['training_config']['batch_size']}")
    lines.append(f"│  Epochs:        {components['training_config']['epochs']}")
    lines.append(f"│  Dropout:       {components['training_config']['regularization']['dropout']}")
    lines.append(f"│  Weight Decay:  {components['training_config']['regularization']['weight_decay']}")
    
    # Evaluation
    lines.append("\n┌─ EVALUATION")
    datasets_str = ", ".join(components['evaluation']['datasets']) if components['evaluation']['datasets'] else "not specified"
    metrics_str = ", ".join(components['evaluation']['metrics'][:5]) if components['evaluation']['metrics'] else "not specified"
    lines.append(f"│  Datasets:     {datasets_str}")
    lines.append(f"│  Metrics:      {metrics_str}")
    if components['evaluation']['improvements']:
        lines.append(f"│  Improvements:")
        for imp in components['evaluation']['improvements']:
            lines.append(f"│    • {imp}")
    
    # Ablations
    lines.append("\n┌─ ABLATION STUDIES")
    if components['ablations']['studies_conducted']:
        lines.append(f"│  Status: Conducted")
        lines.append(f"│  Findings:")
        for finding in components['ablations']['findings']:
            lines.append(f"│    • {finding}")
    else:
        lines.append(f"│  Status: Not reported in paper")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
