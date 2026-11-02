"""
Pre-load fit check: will this model run on this machine, and with what settings?

Pure functions only (no I/O) so they are cheap to test.  Estimates are
conservative: weights + KV cache + a fixed compute-buffer overhead (+ vision
projector).  They are estimates, not guarantees; validate with the benchmark
protocol in the Oprel guide.
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from oprel.runtime.kv_cache import estimate_kv_cache_size
from oprel.runtime.layer_calculator import calculate_optimal_gpu_layers

# llama.cpp compute buffers and the CUDA context sit on top of weights + KV cache.
COMPUTE_OVERHEAD_GB = 0.5
CUDA_CONTEXT_GB = 0.3
CONTEXT_CANDIDATES = (16384, 8192, 4096, 2048, 1024)
RAM_RESERVE_FRACTION = 0.15  # leave room for the OS and the UI

_RANK = {"gpu": 3, "hybrid": 2, "cpu": 1, "insufficient": 0}


@dataclass
class PreflightResult:
    verdict: str  # "gpu" | "hybrid" | "cpu" | "insufficient"
    gpu_layers: int
    total_layers: int
    context_length: int
    suggested_context: int
    vram_needed_gb: float
    ram_needed_gb: float
    kv_cache_gb: float
    messages: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 2)
        return d


def _kv_gb(meta, ctx: int, kv_type: str) -> float:
    return estimate_kv_cache_size(
        batch_size=1,
        context_length=ctx,
        hidden_dim=meta.embedding_dim,
        num_layers=meta.num_layers,
        num_kv_heads=meta.num_kv_heads,
        num_attention_heads=meta.num_attention_heads,
        kv_cache_type=kv_type,
    ).size_gb


def _plan(meta, ctx, free_vram_gb, free_ram_gb, kv_type, mmproj_gb):
    model_gb = meta.file_size_bytes / 1024**3
    kv_gb = _kv_gb(meta, ctx, kv_type)
    total_layers = meta.num_layers
    fixed_overhead = COMPUTE_OVERHEAD_GB + mmproj_gb

    gpu_layers = 0
    if free_vram_gb and free_vram_gb > 0:
        budget = free_vram_gb - CUDA_CONTEXT_GB - fixed_overhead
        if budget > 0:
            gpu_layers = calculate_optimal_gpu_layers(
                model_size_gb=model_gb,
                total_layers=total_layers,
                available_vram_gb=budget,
                context_length=ctx,
                hidden_dim=meta.embedding_dim,
                num_kv_heads=meta.num_kv_heads,
                num_attention_heads=meta.num_attention_heads,
                kv_cache_type=kv_type,
            ).gpu_layers

    gpu_frac = gpu_layers / total_layers if total_layers else 0.0
    vram_needed = (model_gb + kv_gb) * gpu_frac + (CUDA_CONTEXT_GB + fixed_overhead if gpu_layers else 0.0)
    ram_needed = (model_gb + kv_gb) * (1 - gpu_frac) + (0.0 if gpu_layers else fixed_overhead)

    if ram_needed > free_ram_gb * (1 - RAM_RESERVE_FRACTION):
        verdict = "insufficient"
    elif gpu_layers >= total_layers:
        verdict = "gpu"
    elif gpu_layers > 0:
        verdict = "hybrid"
    else:
        verdict = "cpu"
    return verdict, gpu_layers, vram_needed, ram_needed, kv_gb


def check_fit(
    meta,
    free_vram_gb: Optional[float],
    free_ram_gb: float,
    context_length: int = 4096,
    kv_cache_type: str = "f16",
    mmproj_size_gb: float = 0.0,
) -> PreflightResult:
    """Decide whether `meta` (a GGUFMetadata) fits and which settings to use."""
    ctx = min(context_length, meta.context_length or context_length)
    verdict, layers, vram, ram, kv = _plan(meta, ctx, free_vram_gb, free_ram_gb, kv_cache_type, mmproj_size_gb)

    # Pick the largest context (<= requested) that gives a better verdict class.
    suggested = ctx
    if verdict != "gpu":
        best_ctx, best_rank = ctx, _RANK[verdict]
        for cand in CONTEXT_CANDIDATES:
            if cand >= ctx or cand > (meta.context_length or cand):
                continue
            v, *_ = _plan(meta, cand, free_vram_gb, free_ram_gb, kv_cache_type, mmproj_size_gb)
            if _RANK[v] > best_rank:
                best_ctx, best_rank = cand, _RANK[v]
        suggested = best_ctx

    messages: List[str] = []
    if verdict == "insufficient":
        messages.append(
            f"Needs about {ram:.1f} GB of system RAM but only {free_ram_gb:.1f} GB is free. "
            "Choose a smaller model or a lower-bit quantization."
        )
    elif verdict == "cpu" and free_vram_gb:
        messages.append("Not enough free GPU memory; this model will run on the CPU and be slow.")
    elif verdict == "hybrid":
        messages.append(f"{layers} of {meta.num_layers} layers fit on the GPU; the rest run on the CPU.")
    if suggested != ctx:
        messages.append(f"A context of {suggested} tokens would fit better than {ctx}.")

    return PreflightResult(
        verdict=verdict,
        gpu_layers=layers,
        total_layers=meta.num_layers,
        context_length=ctx,
        suggested_context=suggested,
        vram_needed_gb=vram,
        ram_needed_gb=ram,
        kv_cache_gb=kv,
        messages=messages,
    )
