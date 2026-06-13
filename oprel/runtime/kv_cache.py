"""
KV cache size estimation and management.
Accurate memory calculations to prevent OOM crashes.
"""

from typing import Optional
from dataclasses import dataclass

from oprel.utils.logging import get_logger

logger = get_logger(__name__)

# Map KV cache types to bytes per element
# FP16 = 2, Q8_0 = 1, Q4_0 = 0.5 (approximate)
KV_CACHE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "q8_0": 1.0,
    "q5_1": 0.65,
    "q5_0": 0.625,
    "q4_1": 0.55,
    "q4_0": 0.5,
}

@dataclass
class KVCacheEstimate:
    """KV cache memory estimate"""
    size_bytes: int
    size_gb: float
    context_length: int
    batch_size: int
    formula_used: str


def estimate_kv_cache_size(
    batch_size: int,
    context_length: int,
    hidden_dim: int,
    num_layers: int,
    num_kv_heads: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    kv_cache_type: str = "f16"
) -> KVCacheEstimate:
    """
    Estimate KV cache VRAM usage for transformer model.
    
    Formula for standard multi-head attention:
        KV_cache = batch × seq_len × hidden_dim × layers × 2 (K+V) × dtype
        
    Formula for GQA (Grouped Query Attention):
        KV_cache = batch × seq_len × (hidden_dim / num_heads) × num_kv_heads × layers × 2 × dtype
    
    Args:
        batch_size: Batch size (usually 1 for inference)
        context_length: Max sequence length (2048, 4096, 8192, etc.)
        hidden_dim: Model embedding dimension (4096 for 7B, 5120 for 13B)
        num_layers: Number of transformer layers
        num_kv_heads: KV heads for GQA (if None, assumes standard MHA)
        num_attention_heads: Total attention heads (for GQA calculation)
        kv_cache_type: "f16", "q8_0", "q4_0", etc.
        
    Returns:
        KVCacheEstimate with size in bytes and GB
    """
    dtype_bytes = KV_CACHE_BYTES.get(kv_cache_type.lower(), 2)
    
    # Use GQA formula if num_kv_heads provided
    if num_kv_heads is not None and num_attention_heads is not None:
        # GQA: fewer KV heads than attention heads
        head_dim = hidden_dim // num_attention_heads
        kv_cache_bytes = (
            batch_size * context_length * 
            head_dim * num_kv_heads *
            num_layers * 2 * dtype_bytes
        )
        formula = f"GQA: batch({batch_size}) × ctx({context_length}) × head_dim({head_dim}) × kv_heads({num_kv_heads}) × layers({num_layers}) × 2 × {dtype_bytes}"
    else:
        # Standard MHA
        kv_cache_bytes = (
            batch_size * context_length *
            hidden_dim * num_layers * 2 * dtype_bytes
        )
        formula = f"MHA: batch({batch_size}) × ctx({context_length}) × hidden({hidden_dim}) × layers({num_layers}) × 2 × {dtype_bytes}"
    
    kv_cache_gb = kv_cache_bytes / (1024 ** 3)
    
    logger.debug(
        f"KV cache estimate: {kv_cache_gb:.2f}GB for "
        f"context={context_length}, batch={batch_size}"
    )
    
    return KVCacheEstimate(
        size_bytes=kv_cache_bytes,
        size_gb=kv_cache_gb,
        context_length=context_length,
        batch_size=batch_size,
        formula_used=formula
    )


def estimate_kv_cache_from_metadata(
    gguf_metadata,
    batch_size: int = 1,
    context_length: Optional[int] = None,
    kv_cache_type: str = "f16"
) -> KVCacheEstimate:
    """
    Convenience function: Estimate KV cache from GGUF metadata.
    
    Args:
        gguf_metadata: GGUFMetadata object from gguf_parser.py
        batch_size: Batch size
        context_length: Override context (uses metadata.context_length if None)
    """
    ctx_len = context_length or gguf_metadata.context_length
    
    return estimate_kv_cache_size(
        batch_size=batch_size,
        context_length=ctx_len,
        hidden_dim=gguf_metadata.embedding_dim,
        num_layers=gguf_metadata.num_layers,
        num_kv_heads=gguf_metadata.num_kv_heads,
        num_attention_heads=gguf_metadata.num_attention_heads,
        kv_cache_type=kv_cache_type
    )


def recommend_max_context(
    available_vram_gb: float,
    model_size_gb: float,
    hidden_dim: int,
    num_layers: int,
    num_kv_heads: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    kv_cache_type: str = "f16",
    safety_margin: float = 0.15
) -> int:
    """
    Recommend maximum context length that fits in VRAM.
    
    Args:
        available_vram_gb: Total available VRAM
        model_size_gb: Model weight size
        hidden_dim, num_layers, num_kv_heads, num_attention_heads: Model dims
        safety_margin: Reserve % of VRAM for overhead
        
    Returns:
        Recommended max context length
    """
    # Available VRAM for KV cache
    usable_vram = available_vram_gb * (1 - safety_margin)
    vram_for_kv = usable_vram - model_size_gb
    
    if vram_for_kv <= 0:
        logger.warning("Model size exceeds VRAM - hybrid offloading needed")
        return 512  # Minimum context
    
    # Binary search for max context length
    context_options = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    
    for ctx in context_options:
        est = estimate_kv_cache_size(
            batch_size=1,
            context_length=ctx,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            num_attention_heads=num_attention_heads,
            kv_cache_type=kv_cache_type
        )
        
        if est.size_gb > vram_for_kv:
            # Too large, return previous
            idx = context_options.index(ctx)
            if idx > 0:
                return context_options[idx - 1]
            else:
                return 512
    
    # All fit, return max
    return context_options[-1]
