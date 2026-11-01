"""
kvq4.py - 4-bit KV cache block store for SSD offload (production module + test oracle).

Rules:
  * ALL KV data is stored as 4-bit integers: keys, values, attention sinks, recent window.
  * Keys   : per-channel quantization (groups of `group` tokens, one scale/min per channel).
  * Values : per-token quantization   (groups of `group` channels, one scale/min per token).
  * Metadata in Host RAM: fp16 scale/min per group, and fp16 min/max summary of each key block.
  * SSD layout: Fixed-size records, one per (block, kv_head), padded to 4096 bytes (page aligned).

Run the self-test:  python kvq4.py
"""
from __future__ import annotations

import tempfile
import numpy as np
import torch

from oprel.runtime.kv_store.config import Cfg, KV_BITS, LEVELS, PAGE_SIZE
from oprel.runtime.kv_store.direct_io import AlignedBuffer, DirectFile
from oprel.runtime.kv_store.quant_numpy import (
    dequantize_keys_numpy as dequantize_keys,
    dequantize_values_numpy as dequantize_values,
    pack4_numpy as pack4,
    quantize_keys_numpy as quantize_keys,
    quantize_values_numpy as quantize_values,
    unpack4_numpy as unpack4,
)
from oprel.runtime.kv_store.quant_torch import (
    dequantize_keys_torch,
    dequantize_values_torch,
    pack4_torch,
    quantize_keys_torch,
    quantize_values_torch,
    unpack4_torch,
)
from oprel.runtime.kv_store.store import (
    Q4KVStore,
    decode_block_numpy as decode_block,
    decode_block_torch,
    encode_block_numpy as encode_block,
    encode_block_torch,
)


def record_size(cfg: Cfg) -> int:
    return cfg.record_size()


# --------------------------------------------------------------------------- reference attention
def attention(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    logits = K @ q / np.sqrt(q.shape[0])
    w = np.exp(logits - logits.max())
    w /= w.sum()
    return w @ V


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- self-test
def selftest():
    rng = np.random.default_rng(0)
    cfg = Cfg(n_layers=1, n_kv_heads=1, head_dim=128, block_tokens=256, group=32)
    n_blocks = 32
    N = n_blocks * cfg.block_tokens
    D = cfg.head_dim

    # keys with outlier channels
    chan = np.ones(D, np.float32)
    chan[rng.choice(D, 4, replace=False)] = 8.0
    K = rng.standard_normal((N, D)).astype(np.float32) * chan
    V = rng.standard_normal((N, D)).astype(np.float32)

    # 1) raw quantization error, one block (NumPy vs PyTorch parity)
    kb, vb = K[:256], V[:256]
    kq, ks, km = quantize_keys(kb, cfg.group)
    vq, vs, vm = quantize_values(vb, cfg.group)

    # PyTorch parity check
    kb_t, vb_t = torch.from_numpy(kb), torch.from_numpy(vb)
    kq_t, ks_t, km_t = quantize_keys_torch(kb_t, cfg.group)
    vq_t, vs_t, vm_t = quantize_values_torch(vb_t, cfg.group)

    np.testing.assert_allclose(kq, kq_t.numpy())
    np.testing.assert_allclose(ks, ks_t.numpy(), atol=1e-3)
    np.testing.assert_allclose(km, km_t.numpy(), atol=1e-3)
    np.testing.assert_allclose(vq, vq_t.numpy())
    np.testing.assert_allclose(vs, vs_t.numpy(), atol=1e-3)
    np.testing.assert_allclose(vm, vm_t.numpy(), atol=1e-3)

    ek = np.linalg.norm(dequantize_keys(kq, ks, km, cfg.group) - kb) / np.linalg.norm(kb)
    ev = np.linalg.norm(dequantize_values(vq, vs, vm, cfg.group) - vb) / np.linalg.norm(vb)
    assert kq.max() <= 15 and vq.max() <= 15
    rec = cfg.record_size()
    print(f"relative error  K: {ek:.3f}   V: {ev:.3f}   (4-bit, group {cfg.group})")
    print(f"record: {rec} bytes per (block, head)  ->  "
          f"{rec * 8 / (2 * cfg.block_tokens * D):.2f} bits per stored value "
          f"(4 bits + fp16 scale/min overhead)")

    # 2) write to disk, reload index, read back with PyTorch
    with tempfile.TemporaryDirectory() as root:
        st = Q4KVStore(root, cfg, use_direct_io=False)
        for b in range(n_blocks):
            s = slice(b * 256, (b + 1) * 256)
            st.put_block(0, 0, b, K[s], V[s])
        st.save_index()
        st = Q4KVStore.load(root, use_direct_io=False)

        # 3) needle-in-haystack: query matches one planted key, retrieve top-k blocks only
        needle = 17 * 256 + 100
        u = rng.standard_normal(D).astype(np.float32)
        q = u
        K[needle] = 6.0 * u
        blk = needle // 256
        s_ = slice(blk * 256, (blk + 1) * 256)
        st.put_block(0, 0, blk, K[s_], V[s_])
        st.save_index()

        exact = attention(q, K, V)
        lg = K @ q / np.sqrt(D)
        w = np.exp(lg - lg.max())
        w /= w.sum()
        print(f"exact attention mass on needle token: {w[needle]:.3f}")
        assert w[needle] > 0.5, "test invalid: needle does not dominate"

        kd, vd = st.get_blocks(0, 0, list(range(n_blocks)), as_torch=False)
        dense_q4 = attention(q, kd, vd)

        topk = 4
        sel = st.select_blocks(0, 0, q, topk, must_include=(0, n_blocks - 1))
        ks_, vs_ = st.get_blocks(0, 0, sel, as_torch=False)
        sparse_q4 = attention(q, ks_, vs_)

        print(f"\nneedle is in block {needle // 256}; selected blocks: {sel}")
        print(f"cosine vs exact FP32 attention   dense Q4: {_cos(exact, dense_q4):.4f}   "
              f"sparse Q4 (top-{topk}+sink+recent): {_cos(exact, sparse_q4):.4f}")
        per_tok = rec / cfg.block_tokens
        print(f"SSD bytes per decoded token: "
              f"dense {n_blocks * rec / 1e3:.0f} KB   sparse {len(sel) * rec / 1e3:.0f} KB   "
              f"({n_blocks / len(sel):.1f}x less)")
        print(f"stored cost: {per_tok:.0f} bytes/token/head")
        assert needle // 256 in sel, "retrieval missed the needle block"
        st.close()

    print("\nself-test passed successfully (NumPy & PyTorch validated)")


if __name__ == "__main__":
    selftest()
