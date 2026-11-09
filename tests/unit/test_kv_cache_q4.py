"""Unit tests verifying strict q4 KV cache enforcement and 32k context sizing."""

import pytest
from oprel.core.config import Config
from oprel.runtime.backends.llama_cpp import _resolve_kv_cache_type
from oprel.runtime.kv_cache import recommend_max_context, estimate_kv_cache_size


def test_resolve_kv_cache_type_strictly_q4():
    # 1. Default should be q4_0
    cfg = Config()
    assert _resolve_kv_cache_type(cfg, "Q4_K_M", is_vision=False) == "q4_0"

    # 2. Vision models MUST resolve to q4_0 (no f16 fallback!)
    assert _resolve_kv_cache_type(cfg, "Q4_K_M", is_vision=True) == "q4_0"

    # 3. Requesting f16 or q8_0 should be clamped to q4_0
    cfg_f16 = Config(kv_cache_type="f16")
    assert _resolve_kv_cache_type(cfg_f16, "Q8_0", is_vision=False) == "q4_0"

    cfg_q8 = Config(kv_cache_type="q8_0")
    assert _resolve_kv_cache_type(cfg_q8, "Q8_0", is_vision=False) == "q4_0"

    # 4. Valid 4-bit quantizations (iq4_nl, q4_1, turboquant) allowed
    cfg_iq4 = Config(kv_cache_type="iq4_nl")
    assert _resolve_kv_cache_type(cfg_iq4, "Q4_K_M", is_vision=False) == "iq4_nl"

    cfg_turbo = Config(kv_cache_type="turboquant")
    assert _resolve_kv_cache_type(cfg_turbo, "Q4_K_M", is_vision=False) == "q4_0"


def test_recommend_max_context_scales_to_32k():
    # 4GB GPU (3.3GB available), Gemma 3 4B (4.2GB model size)
    # Previously, vram_for_kv <= 0 clamped to 512!
    # Now, under hybrid offload with q4_0, it scales safely up to 32k!
    ctx = recommend_max_context(
        available_vram_gb=3.3,
        model_size_gb=4.2,
        hidden_dim=2560,
        num_layers=34,
        num_kv_heads=4,
        num_attention_heads=8,
        kv_cache_type="q4_0",
        max_supported_ctx=32768,
    )
    assert ctx == 32768


def test_q4_kv_cache_size_is_tiny():
    # Dense formula at 32k context with q4_0
    est_q4 = estimate_kv_cache_size(
        batch_size=1,
        context_length=32768,
        hidden_dim=2048,
        num_layers=34,
        num_kv_heads=4,
        num_attention_heads=8,
        kv_cache_type="q4_0",
    )
    est_f16 = estimate_kv_cache_size(
        batch_size=1,
        context_length=32768,
        hidden_dim=2048,
        num_layers=34,
        num_kv_heads=4,
        num_attention_heads=8,
        kv_cache_type="f16",
    )
    # q4_0 must be 1/4th the memory of f16!
    assert est_q4.size_gb == est_f16.size_gb / 4.0
    assert est_q4.size_gb <= 1.10
    assert est_f16.size_gb >= 4.20
