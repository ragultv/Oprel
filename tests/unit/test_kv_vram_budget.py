from __future__ import annotations

import pytest
from oprel.runtime.kv_cache import calculate_context_for_vram_budget, recommend_max_context
from oprel.runtime.layer_calculator import calculate_optimal_gpu_layers


def test_calculate_context_for_600mb_vram_budget_gemma():
    """Verify Gemma 3 4B with GQA (4 KV heads, dim 128) achieves large context in 600MB."""
    # Gemma 3 4B: 34 layers, 16 attention heads, 4 kv heads, embedding_dim 2048 (head_dim 128)
    ctx = calculate_context_for_vram_budget(
        hidden_dim=2048,
        num_layers=34,
        num_kv_heads=4,
        num_attention_heads=16,
        kv_cache_type="q4_0",
        vram_budget_mb=600.0,
        max_supported_ctx=32768,
    )
    # 600MB / (2 * 34 * 4 * 128 * 0.5) bytes = 629,145,600 / 17,408 = 36,141 tokens -> capped at 32,768
    assert ctx == 32768


def test_calculate_context_for_600mb_vram_budget_llama3():
    """Verify Llama 3 8B with GQA (8 KV heads, dim 128) calculates exact fit in 600MB."""
    # Llama 3 8B: 32 layers, 32 attention heads, 8 kv heads, embedding_dim 4096 (head_dim 128)
    ctx = calculate_context_for_vram_budget(
        hidden_dim=4096,
        num_layers=32,
        num_kv_heads=8,
        num_attention_heads=32,
        kv_cache_type="q4_0",
        vram_budget_mb=600.0,
        max_supported_ctx=32768,
    )
    # 600MB / (2 * 32 * 8 * 128 * 0.5) bytes = 629,145,600 / 32,768 = 19,200 tokens -> 18,944 (divisible by 512)
    assert ctx == 18944


def test_calculate_context_for_600mb_vram_budget_mha():
    """Verify older non-GQA model scales down gracefully."""
    # 32 layers, 32 KV heads, head_dim 128
    ctx = calculate_context_for_vram_budget(
        hidden_dim=4096,
        num_layers=32,
        num_kv_heads=32,
        num_attention_heads=32,
        kv_cache_type="q4_0",
        vram_budget_mb=600.0,
        max_supported_ctx=32768,
    )
    # 600MB / (2 * 32 * 32 * 128 * 0.5) bytes = 629,145,600 / 131,072 = 4,800 tokens -> rounded down to 4,608
    assert ctx == 4608


def test_layer_calculator_preserves_600mb_kv_cache():
    """Verify GPU layer calculation reserves 600MB for KV cache and doesn't over-allocate weights."""
    calc = calculate_optimal_gpu_layers(
        model_size_gb=2.64,
        total_layers=34,
        available_vram_gb=2.9,
        context_length=32768,
        kv_cache_type="q4_0",
        safety_margin=0.15,
    )
    # Available VRAM = 2.9GB
    # Headroom = max(0.40, 2.9 * 0.15 = 0.435) = 0.435GB
    # KV cache reserved budget = 0.60GB
    # Usable for weights = 2.9 - 0.435 - 0.60 = 1.865GB
    # Weight per layer = 2.64 / 34 = 0.0776GB
    # Expected GPU layers = int(1.865 / 0.0776) = 24 layers
    assert calc.gpu_layers == 24
    # Actual VRAM allocated must leave at least 0.40GB headroom
    assert calc.total_vram_gb + 0.40 <= 2.9
