"""
Improved GPU layer calculation with KV cache awareness.
More accurate than Ollama's simple estimation.
"""

from typing import Optional, Tuple
from dataclasses import dataclass

from oprel.runtime.kv_cache import estimate_kv_cache_size
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LayerCalculation:
    """Result of GPU layer calculation"""
    gpu_layers: int
    total_layers: int
    model_vram_gb: float
    kv_cache_vram_gb: float
    total_vram_gb: float
    gpu_percentage: float
    recommendation: str


def calculate_optimal_gpu_layers(
    model_size_gb: float,
    total_layers: int,
    available_vram_gb: float,
    context_length: int = 2048,
    batch_size: int = 1,
    hidden_dim: int = 4096,
    num_kv_heads: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    kv_cache_type: str = "f16",
    safety_margin: float = 0.15
) -> LayerCalculation:
    """
    Calculate how many layers fit on GPU, accounting for KV cache.
    More accurate than Ollama's simple layer_size * N calculation.
    
    Args:
        model_size_gb: Total model size
        total_layers: Total transformer layers
        available_vram_gb: Available VRAM
        context_length: Requested context
        batch_size: Batch size
        hidden_dim: Model embedding dimension
        num_kv_heads: KV heads for GQA
        num_attention_heads: Total attention heads
        safety_margin: Reserve % VRAM for overhead (activations, etc.)
        
    Returns:
        LayerCalculation with optimal GPU layer count
    """
    # Reserve safety margin
    usable_vram = available_vram_gb * (1 - safety_margin)
    
    # Estimate KV cache for full model
    kv_cache = estimate_kv_cache_size(
        batch_size=batch_size,
        context_length=context_length,
        hidden_dim=hidden_dim,
        num_layers=total_layers,
        num_kv_heads=num_kv_heads,
        num_attention_heads=num_attention_heads,
        kv_cache_type=kv_cache_type
    )
    
    # Check if full model + KV cache fits
    total_needed = model_size_gb + kv_cache.size_gb
    
    if total_needed <= usable_vram:
        # Full model fits!
        logger.info(
            f"Full model fits: {model_size_gb:.1f}GB + "
            f"{kv_cache.size_gb:.1f}GB KV = {total_needed:.1f}GB "
            f"(available: {usable_vram:.1f}GB)"
        )
        return LayerCalculation(
            gpu_layers=total_layers,
            total_layers=total_layers,
            model_vram_gb=model_size_gb,
            kv_cache_vram_gb=kv_cache.size_gb,
            total_vram_gb=total_needed,
            gpu_percentage=100.0,
            recommendation="Full GPU - all layers fit"
        )
    
    # Model doesn't fit - calculate partial offload
    # Weight size per layer (assume uniform distribution)
    weight_per_layer = model_size_gb / total_layers
    
    # KV cache size per layer
    kv_per_layer = kv_cache.size_gb / total_layers
    
    # Memory per layer (weights + KV)
    mem_per_layer = weight_per_layer + kv_per_layer
    
    # Calculate max layers that fit
    max_layers = int(usable_vram / mem_per_layer)
    max_layers = min(max_layers, total_layers)
    max_layers = max(max_layers, 0)
    
    # Actual VRAM usage
    actual_vram = max_layers * mem_per_layer
    gpu_pct = (max_layers / total_layers) * 100
    
    if max_layers == 0:
        recommendation = "CPU only - model too large for GPU"
    elif max_layers < total_layers * 0.3:
        recommendation = f"Hybrid (mostly CPU) - only {max_layers}/{total_layers} layers on GPU"
    elif max_layers < total_layers:
        recommendation = f"Hybrid GPU/CPU - {max_layers}/{total_layers} layers on GPU"
    else:
        recommendation = "Full GPU"
    
    logger.info(
        f"Partial offload: {max_layers}/{total_layers} layers "
        f"({gpu_pct:.0f}%) fit in {usable_vram:.1f}GB VRAM"
    )
    
    return LayerCalculation(
        gpu_layers=max_layers,
        total_layers=total_layers,
        model_vram_gb=max_layers * weight_per_layer,
        kv_cache_vram_gb=max_layers * kv_per_layer,
        total_vram_gb=actual_vram,
        gpu_percentage=gpu_pct,
        recommendation=recommendation
    )


def calculate_from_metadata(
    gguf_metadata,
    available_vram_gb: float,
    context_length: Optional[int] = None,
    batch_size: int = 1,
    kv_cache_type: str = "auto",
    is_vision: bool = False
) -> LayerCalculation:
    """
    Convenience: Calculate GPU layers from GGUF metadata.
    
    Args:
        gguf_metadata: GGUFMetadata from gguf_parser.py
        available_vram_gb: Available VR AM
        context_length: Override context (uses metadata value if None)
        batch_size: Batch size
    """
    ctx_len = context_length or gguf_metadata.context_length
    
    # Auto-resolve KV cache type if requested
    if kv_cache_type == "auto":
        if is_vision:
            # Vision models need F16 for stability
            resolved_type = "f16"
        else:
            model_quant = (gguf_metadata.quantization or "").upper()
            if any(q in model_quant for q in ["Q2", "Q3", "Q4"]):
                resolved_type = "q4_0"
            elif any(q in model_quant for q in ["Q5", "Q6", "Q8"]):
                resolved_type = "q8_0"
            else:
                resolved_type = "f16"
    else:
        resolved_type = kv_cache_type

    return calculate_optimal_gpu_layers(
        model_size_gb=gguf_metadata.file_size_bytes / (1024 ** 3),
        total_layers=gguf_metadata.num_layers,
        available_vram_gb=available_vram_gb,
        context_length=ctx_len,
        batch_size=batch_size,
        hidden_dim=gguf_metadata.embedding_dim,
        num_kv_heads=gguf_metadata.num_kv_heads,
        num_attention_heads=gguf_metadata.num_attention_heads,
        kv_cache_type=resolved_type
    )
