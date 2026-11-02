"""
Hardware tiers and the default settings that go with them.

Defaults only apply when the user has not set a value explicitly.
Tier boundaries follow the Oprel guide (section 4); they are starting points
to be tuned with the benchmark protocol, not measured optima.
"""

from typing import Optional

TIER_A = "A"  # <= 5 GB VRAM, or CPU only
TIER_B = "B"  # 5-7.5 GB VRAM
TIER_C = "C"  # > 7.5 GB VRAM

_CTX = {TIER_A: 4096, TIER_B: 6144, TIER_C: 8192}


def detect_tier(vram_gb: Optional[float]) -> str:
    if not vram_gb or vram_gb <= 5.0:
        return TIER_A
    if vram_gb <= 7.5:
        return TIER_B
    return TIER_C


def default_context(vram_gb: Optional[float]) -> int:
    """Context length to use when the user did not choose one."""
    return _CTX[detect_tier(vram_gb)]


# (alias, quantization, approximate download size in GB)
_STARTERS = {
    "cpu_low": ("qwen3-1.7b", "Q4_K_M", 1.1),
    TIER_A: ("qwen3-4b", "Q4_K_M", 2.5),
    TIER_B: ("qwen3-8b", "Q4_K_M", 5.0),
    TIER_C: ("qwen3-8b", "Q5_K_M", 5.9),
}


def recommend_starter(vram_gb: Optional[float], ram_gb: float) -> dict:
    """One safe first model for the detected hardware, with a plain-language reason."""
    tier = detect_tier(vram_gb)
    if not vram_gb and ram_gb < 12:
        key, reason = "cpu_low", "No dedicated GPU was found and memory is limited, so a very small model keeps replies responsive."
    elif tier == TIER_A and ram_gb < 8:
        key, reason = "cpu_low", "Your memory is limited, so this small model avoids slowdowns."
    elif tier == TIER_A:
        key, reason = TIER_A, "Fits comfortably in a small graphics card or in regular memory."
    elif tier == TIER_B:
        key, reason = TIER_B, "A strong general model that mostly fits on your graphics card."
    else:
        key, reason = TIER_C, "A strong general model that fits fully on your graphics card."
    alias, quant, size = _STARTERS[key]
    return {"alias": alias, "quantization": quant, "approx_size_gb": size, "reason": reason}
