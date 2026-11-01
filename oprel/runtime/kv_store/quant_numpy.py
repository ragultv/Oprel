"""Reference NumPy implementation (test oracle) for 4-bit KV quantization."""
from __future__ import annotations

import numpy as np
from oprel.runtime.kv_store.config import LEVELS


def pack4_numpy(q: np.ndarray) -> np.ndarray:
    """uint8 values 0..15 with even last axis -> two values per byte."""
    assert q.max() <= LEVELS
    return (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)


def unpack4_numpy(p: np.ndarray) -> np.ndarray:
    """Two 4-bit values per byte -> uint8 array."""
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), np.uint8)
    out[..., 0::2] = p & 0x0F
    out[..., 1::2] = p >> 4
    return out


def _quant_numpy(x: np.ndarray, axis: int):
    """Asymmetric 4-bit min/max quantization along `axis`."""
    mn = x.min(axis=axis, keepdims=True)
    mx = x.max(axis=axis, keepdims=True)
    scale = (mx - mn) / LEVELS
    scale16 = scale.astype(np.float16)
    scale16 = np.where(scale16 == 0, 1.0, scale16).astype(np.float16)  # avoid div by 0
    mn16 = mn.astype(np.float16)
    # quantize against fp16-rounded params so dequantization matches exactly
    s = scale16.astype(np.float32)
    m = mn16.astype(np.float32)
    q = np.clip(np.rint((x - m) / s), 0, LEVELS).astype(np.uint8)
    return q, scale16, mn16


def quantize_keys_numpy(K: np.ndarray, group: int):
    """K [T, D] -> per-channel groups over tokens. Returns q[T,D], scale[T/G,D], min[T/G,D]."""
    T, D = K.shape
    q, s, m = _quant_numpy(K.reshape(T // group, group, D).astype(np.float32), axis=1)
    return q.reshape(T, D), s[:, 0, :], m[:, 0, :]


def dequantize_keys_numpy(q: np.ndarray, s: np.ndarray, m: np.ndarray, group: int) -> np.ndarray:
    """Dequantize keys from 4-bit integer, fp16 scale and min."""
    T, D = q.shape
    x = q.reshape(T // group, group, D).astype(np.float32)
    x = x * s[:, None, :].astype(np.float32) + m[:, None, :].astype(np.float32)
    return x.reshape(T, D)


def quantize_values_numpy(V: np.ndarray, group: int):
    """V [T, D] -> per-token groups over channels. Returns q[T,D], scale[T,D/G], min[T,D/G]."""
    T, D = V.shape
    q, s, m = _quant_numpy(V.reshape(T, D // group, group).astype(np.float32), axis=2)
    return q.reshape(T, D), s[:, :, 0], m[:, :, 0]


def dequantize_values_numpy(q: np.ndarray, s: np.ndarray, m: np.ndarray, group: int) -> np.ndarray:
    """Dequantize values from 4-bit integer, fp16 scale and min."""
    T, D = q.shape
    x = q.reshape(T, D // group, group).astype(np.float32)
    x = x * s[:, :, None].astype(np.float32) + m[:, :, None].astype(np.float32)
    return x.reshape(T, D)
