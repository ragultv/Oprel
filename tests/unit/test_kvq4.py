"""Unit tests for 4-bit KV Cache SSD Store (PyTorch backend & NumPy oracle)."""
from __future__ import annotations

import tempfile
import numpy as np
import pytest
import torch

from oprel.runtime.kv_store.config import Cfg, PAGE_SIZE
from oprel.runtime.kv_store.direct_io import AlignedBuffer, DirectFile
from oprel.runtime.kv_store.quant_numpy import (
    dequantize_keys_numpy,
    dequantize_values_numpy,
    pack4_numpy,
    quantize_keys_numpy,
    quantize_values_numpy,
    unpack4_numpy,
)
from oprel.runtime.kv_store.quant_torch import (
    dequantize_keys_torch,
    dequantize_values_torch,
    pack4_torch,
    quantize_keys_torch,
    quantize_values_torch,
    unpack4_torch,
)
from oprel.runtime.kv_store.store import Q4KVStore


@pytest.fixture
def sample_cfg():
    return Cfg(
        n_layers=2,
        n_kv_heads=4,
        head_dim=128,
        block_tokens=256,
        group=32,
        summary_tokens=16,
    )


def test_config_validation():
    # Valid config
    cfg = Cfg(n_layers=1, n_kv_heads=1, head_dim=128, block_tokens=256, group=32, summary_tokens=16)
    rec = cfg.record_size()
    assert rec % PAGE_SIZE == 0
    assert rec > 0

    # Invalid head_dim (not divisible by group)
    with pytest.raises(AssertionError):
        Cfg(n_layers=1, n_kv_heads=1, head_dim=130, block_tokens=256, group=32)

    # Invalid block_tokens (not divisible by summary_tokens)
    with pytest.raises(AssertionError):
        Cfg(n_layers=1, n_kv_heads=1, head_dim=128, block_tokens=250, group=32, summary_tokens=16)


def test_bit_packing_roundtrip():
    rng = np.random.default_rng(42)
    # 256 tokens x 128 dim uint8 in range [0, 15]
    arr = rng.integers(0, 16, size=(256, 128), dtype=np.uint8)

    # NumPy roundtrip
    packed_np = pack4_numpy(arr)
    assert packed_np.shape == (256, 64)
    unpacked_np = unpack4_numpy(packed_np)
    np.testing.assert_array_equal(arr, unpacked_np)

    # PyTorch roundtrip
    t_arr = torch.from_numpy(arr)
    packed_t = pack4_torch(t_arr)
    assert packed_t.shape == (256, 64)
    unpacked_t = unpack4_torch(packed_t)
    assert torch.equal(t_arr, unpacked_t)

    # Parity between NumPy and PyTorch packed output
    np.testing.assert_array_equal(packed_np, packed_t.numpy())


def test_quantization_parity_numpy_vs_torch(sample_cfg):
    rng = np.random.default_rng(123)
    T, D = sample_cfg.block_tokens, sample_cfg.head_dim

    # Keys with outlier channels
    chan = np.ones(D, dtype=np.float32)
    chan[rng.choice(D, 4, replace=False)] = 8.0
    K_np = rng.standard_normal((T, D)).astype(np.float32) * chan
    V_np = rng.standard_normal((T, D)).astype(np.float32)

    K_t = torch.from_numpy(K_np)
    V_t = torch.from_numpy(V_np)

    # Quantize Keys
    kq_np, ks_np, km_np = quantize_keys_numpy(K_np, sample_cfg.group)
    kq_t, ks_t, km_t = quantize_keys_torch(K_t, sample_cfg.group)

    np.testing.assert_array_equal(kq_np, kq_t.numpy())
    np.testing.assert_allclose(ks_np, ks_t.numpy(), atol=1e-3)
    np.testing.assert_allclose(km_np, km_t.numpy(), atol=1e-3)

    # Quantize Values
    vq_np, vs_np, vm_np = quantize_values_numpy(V_np, sample_cfg.group)
    vq_t, vs_t, vm_t = quantize_values_torch(V_t, sample_cfg.group)

    np.testing.assert_array_equal(vq_np, vq_t.numpy())
    np.testing.assert_allclose(vs_np, vs_t.numpy(), atol=1e-3)
    np.testing.assert_allclose(vm_np, vm_t.numpy(), atol=1e-3)

    # Dequantize Keys parity
    K_rec_np = dequantize_keys_numpy(kq_np, ks_np, km_np, sample_cfg.group)
    K_rec_t = dequantize_keys_torch(kq_t, ks_t, km_t, sample_cfg.group)
    np.testing.assert_allclose(K_rec_np, K_rec_t.numpy(), atol=1e-3)

    # Dequantize Values parity
    V_rec_np = dequantize_values_numpy(vq_np, vs_np, vm_np, sample_cfg.group)
    V_rec_t = dequantize_values_torch(vq_t, vs_t, vm_t, sample_cfg.group)
    np.testing.assert_allclose(V_rec_np, V_rec_t.numpy(), atol=1e-3)

    # Check relative error bounded around ~7-8%
    rel_err_k = np.linalg.norm(K_rec_np - K_np) / np.linalg.norm(K_np)
    rel_err_v = np.linalg.norm(V_rec_np - V_np) / np.linalg.norm(V_np)
    assert rel_err_k < 0.12, f"Key relative error too high: {rel_err_k:.3f}"
    assert rel_err_v < 0.12, f"Value relative error too high: {rel_err_v:.3f}"


def test_direct_io_aligned_buffer_and_file():
    buf_size = PAGE_SIZE * 2
    aligned = AlignedBuffer(buf_size)
    assert aligned.size == buf_size
    test_data = b"X" * buf_size
    aligned.write_bytes(test_data)
    assert aligned.to_bytes() == test_data
    aligned.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test_direct.bin"
        df_w = DirectFile(path, mode="w", direct=True)
        df_w.seek_and_write(0, test_data)
        df_w.close()

        df_r = DirectFile(path, mode="r", direct=True)
        read_back = df_r.seek_and_read(0, buf_size)
        df_r.close()
        assert read_back == test_data


def test_store_write_retrieve_and_quest_selection(sample_cfg):
    rng = np.random.default_rng(999)
    T, D = sample_cfg.block_tokens, sample_cfg.head_dim
    n_blocks = 8

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Q4KVStore(tmpdir, sample_cfg, use_direct_io=False)

        # Write 8 blocks
        all_K = []
        all_V = []
        for b in range(n_blocks):
            K_b = torch.randn(T, D, dtype=torch.float32)
            V_b = torch.randn(T, D, dtype=torch.float32)
            all_K.append(K_b)
            all_V.append(V_b)
            store.put_block(layer=0, head=0, block=b, K=K_b, V=V_b)

        try:
            # Plant needle query matching block 5 with distinct signature
            needle_block = 5
            all_K[needle_block] += 5.0
            store.put_block(layer=0, head=0, block=needle_block, K=all_K[needle_block], V=all_V[needle_block])
            q = all_K[needle_block][100].clone()

            # Quest select top-2 blocks + sinks
            selected = store.select_blocks(layer=0, head=0, q=q, k=2, must_include=(0, n_blocks - 1))
            assert needle_block in selected
            assert 0 in selected
            assert (n_blocks - 1) in selected

            # Read back as PyTorch tensors
            k_ret, v_ret = store.get_blocks(layer=0, head=0, blocks=[needle_block], as_torch=True)
            assert k_ret.shape == (T, D)
            assert v_ret.shape == (T, D)

            # Read back as NumPy arrays
            k_np, v_np = store.get_blocks(layer=0, head=0, blocks=[needle_block], as_torch=False)
            np.testing.assert_allclose(k_ret.numpy(), k_np, atol=1e-4)

            # Save and reload index
            store.save_index()
            loaded_store = Q4KVStore.load(tmpdir, use_direct_io=False)
            assert (0, 0) in loaded_store.summ
            assert len(loaded_store.summ[(0, 0)]) == n_blocks

            # Verify Quest selection from reloaded store
            reloaded_selected = loaded_store.select_blocks(layer=0, head=0, q=q, k=2, must_include=(0, n_blocks - 1))
            assert reloaded_selected == selected
        finally:
            store.close()
            if "loaded_store" in locals():
                loaded_store.close()
