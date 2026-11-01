"""PyTorch implementation of 4-bit KV quantization, packing, and dequantization."""
from __future__ import annotations

import torch
from oprel.runtime.kv_store.config import LEVELS


def pack4_torch(q: torch.Tensor) -> torch.Tensor:
    """uint8 tensor [..., D] with even last axis -> packed uint8 tensor [..., D/2]."""
    assert q.dtype == torch.uint8, f"Expected torch.uint8, got {q.dtype}"
    # Even elements occupy lower 4 bits, odd elements occupy upper 4 bits
    even = q[..., 0::2]
    odd = q[..., 1::2]
    return (even | (odd << 4)).to(torch.uint8)


def unpack4_torch(p: torch.Tensor) -> torch.Tensor:
    """Packed uint8 tensor [..., D/2] -> unpacked uint8 tensor [..., D]."""
    assert p.dtype == torch.uint8, f"Expected torch.uint8, got {p.dtype}"
    out_shape = p.shape[:-1] + (p.shape[-1] * 2,)
    out = torch.empty(out_shape, dtype=torch.uint8, device=p.device)
    out[..., 0::2] = p & 0x0F
    out[..., 1::2] = p >> 4
    return out


def _quant_torch(x: torch.Tensor, dim: int):
    """Asymmetric 4-bit min/max quantization along `dim`."""
    mn = x.amin(dim=dim, keepdim=True)
    mx = x.amax(dim=dim, keepdim=True)
    scale = (mx - mn) / float(LEVELS)

    scale16 = scale.to(torch.float16)
    scale16 = torch.where(
        scale16 == 0,
        torch.tensor(1.0, dtype=torch.float16, device=x.device),
        scale16,
    )
    mn16 = mn.to(torch.float16)

    # Quantize against fp16-rounded params so dequantization matches exactly
    s = scale16.to(torch.float32)
    m = mn16.to(torch.float32)

    # Round-half-to-even to match NumPy's np.rint
    q = torch.clamp(torch.round((x - m) / s), 0, LEVELS).to(torch.uint8)
    return q, scale16, mn16


def quantize_keys_torch(K: torch.Tensor, group: int):
    """K [T, D] -> per-channel groups over tokens. Returns q[T,D], scale[T/G,D], min[T/G,D]."""
    assert K.ndim == 2, f"Expected 2D tensor [T, D], got {K.shape}"
    T, D = K.shape
    assert T % group == 0, f"T ({T}) must be divisible by group ({group})"
    x = K.view(T // group, group, D).to(torch.float32)
    q, s, m = _quant_torch(x, dim=1)
    return q.view(T, D), s.select(1, 0), m.select(1, 0)


def dequantize_keys_torch(
    q: torch.Tensor,
    s: torch.Tensor,
    m: torch.Tensor,
    group: int,
) -> torch.Tensor:
    """Dequantize keys from 4-bit integer, fp16 scale and min."""
    T, D = q.shape
    x = q.view(T // group, group, D).to(torch.float32)
    s_f32 = s.unsqueeze(1).to(torch.float32)
    m_f32 = m.unsqueeze(1).to(torch.float32)
    x = x * s_f32 + m_f32
    return x.view(T, D)


def quantize_values_torch(V: torch.Tensor, group: int):
    """V [T, D] -> per-token groups over channels. Returns q[T,D], scale[T,D/G], min[T,D/G]."""
    assert V.ndim == 2, f"Expected 2D tensor [T, D], got {V.shape}"
    T, D = V.shape
    assert D % group == 0, f"D ({D}) must be divisible by group ({group})"
    x = V.view(T, D // group, group).to(torch.float32)
    q, s, m = _quant_torch(x, dim=2)
    return q.view(T, D), s.select(2, 0), m.select(2, 0)


def dequantize_values_torch(
    q: torch.Tensor,
    s: torch.Tensor,
    m: torch.Tensor,
    group: int,
) -> torch.Tensor:
    """Dequantize values from 4-bit integer, fp16 scale and min."""
    T, D = q.shape
    x = q.view(T, D // group, group).to(torch.float32)
    s_f32 = s.unsqueeze(2).to(torch.float32)
    m_f32 = m.unsqueeze(2).to(torch.float32)
    x = x * s_f32 + m_f32
    return x.view(T, D)
