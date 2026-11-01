"""4-bit KV cache SSD store with Host-RAM Quest summary index."""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from oprel.runtime.kv_store.config import Cfg, KV_BITS, PAGE_SIZE
from oprel.runtime.kv_store.direct_io import DirectFile
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


def encode_block_numpy(cfg: Cfg, K: np.ndarray, V: np.ndarray) -> bytes:
    """Encode K, V [T, D] into padded record bytes using NumPy."""
    kq, ks, km = quantize_keys_numpy(K, cfg.group)
    vq, vs, vm = quantize_values_numpy(V, cfg.group)
    buf = b"".join(
        np.ascontiguousarray(p).tobytes()
        for p in (pack4_numpy(kq), ks, km, pack4_numpy(vq), vs, vm)
    )
    rec = cfg.record_size()
    return buf + b"\0" * (rec - len(buf))


def encode_block_torch(cfg: Cfg, K: torch.Tensor, V: torch.Tensor) -> bytes:
    """Encode K, V [T, D] into padded record bytes using PyTorch."""
    kq, ks, km = quantize_keys_torch(K, cfg.group)
    vq, vs, vm = quantize_values_torch(V, cfg.group)
    buf = b"".join(
        p.contiguous().cpu().numpy().tobytes()
        for p in (pack4_torch(kq), ks, km, pack4_torch(vq), vs, vm)
    )
    rec = cfg.record_size()
    return buf + b"\0" * (rec - len(buf))


def decode_block_numpy(cfg: Cfg, buf: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Decode raw record bytes to (K, V) [T, D] float32 NumPy arrays."""
    T, D, G = cfg.block_tokens, cfg.head_dim, cfg.group
    off = 0

    def take(dtype, shape):
        nonlocal off
        count = int(np.prod(shape))
        a = np.frombuffer(buf, dtype=dtype, count=count, offset=off).reshape(shape)
        off += a.nbytes
        return a

    kp = take(np.uint8, (T, D // 2))
    ks = take(np.float16, (T // G, D))
    km = take(np.float16, (T // G, D))
    vp = take(np.uint8, (T, D // 2))
    vs = take(np.float16, (T, D // G))
    vm = take(np.float16, (T, D // G))

    K = dequantize_keys_numpy(unpack4_numpy(kp), ks, km, G)
    V = dequantize_values_numpy(unpack4_numpy(vp), vs, vm, G)
    return K, V


def decode_block_torch(
    cfg: Cfg,
    buf: bytes,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode raw record bytes directly into PyTorch tensors."""
    T, D, G = cfg.block_tokens, cfg.head_dim, cfg.group
    off = 0

    def take(dtype, shape):
        nonlocal off
        count = int(np.prod(shape))
        arr = np.frombuffer(buf, dtype=dtype, count=count, offset=off).reshape(shape)
        off += arr.nbytes
        return torch.from_numpy(arr.copy()).to(device)

    kp = take(np.uint8, (T, D // 2))
    ks = take(np.float16, (T // G, D))
    km = take(np.float16, (T // G, D))
    vp = take(np.uint8, (T, D // 2))
    vs = take(np.float16, (T, D // G))
    vm = take(np.float16, (T, D // G))

    K = dequantize_keys_torch(unpack4_torch(kp), ks, km, G)
    V = dequantize_values_torch(unpack4_torch(vp), vs, vm, G)
    return K, V


class Q4KVStore:
    """
    SSD-resident 4-bit KV cache block store with Host-RAM Quest summaries.
    Designed for memory-constrained local inference (e.g. GTX 1650 4GB VRAM).
    """

    def __init__(
        self,
        root: str,
        cfg: Cfg,
        io_threads: int = 8,
        use_direct_io: bool = True,
    ):
        os.makedirs(root, exist_ok=True)
        self.root = os.path.abspath(root)
        self.cfg = cfg
        self.rec = cfg.record_size()
        self.use_direct_io = use_direct_io
        self.pool = ThreadPoolExecutor(max_workers=io_threads)

        # Thread-local storage for read file handles (parallel reads per layer)
        self._tl = threading.local()

        # Shared write handles & lock
        self._wlock = threading.Lock()
        self._wfiles: Dict[int, DirectFile] = {}

        # Host-RAM summary index: (layer, head) -> dict of block_id -> (kmin, kmax) [P, D] fp16
        self.summ: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

        # Cached stacked tensors for vectorized Quest retrieval: (layer, head) -> (ids, kmin_stack, kmax_stack)
        self._stack_cache: Dict[Tuple[int, int], Tuple[List[int], torch.Tensor, torch.Tensor]] = {}

    def _path(self, layer: int) -> str:
        return os.path.join(self.root, f"layer_{layer}.kv")

    def _offset(self, head: int, block: int) -> int:
        return (block * self.cfg.n_kv_heads + head) * self.rec

    def _get_read_file(self, layer: int) -> DirectFile:
        h = getattr(self._tl, "handles", None)
        if h is None:
            h = self._tl.handles = {}
        if layer not in h:
            h[layer] = DirectFile(self._path(layer), mode="r", direct=self.use_direct_io)
        return h[layer]

    def _get_write_file(self, layer: int) -> DirectFile:
        with self._wlock:
            if layer not in self._wfiles:
                self._wfiles[layer] = DirectFile(self._path(layer), mode="w", direct=self.use_direct_io)
            return self._wfiles[layer]

    def put_block(
        self,
        layer: int,
        head: int,
        block: int,
        K: Union[np.ndarray, torch.Tensor],
        V: Union[np.ndarray, torch.Tensor],
    ):
        """Quantize and write a single (block, kv_head) to SSD."""
        c = self.cfg
        if isinstance(K, torch.Tensor):
            assert K.shape == (c.block_tokens, c.head_dim) and V.shape == K.shape
            data = encode_block_torch(c, K, V)
            P = c.block_tokens // c.summary_tokens
            # Summary min/max over 16-token pages stored in Host-RAM
            Kp = K.detach().cpu().to(torch.float32).view(P, c.summary_tokens, c.head_dim)
            kmin = Kp.amin(dim=1).to(torch.float16).numpy()
            kmax = Kp.amax(dim=1).to(torch.float16).numpy()
        else:
            assert K.shape == (c.block_tokens, c.head_dim) and V.shape == K.shape
            data = encode_block_numpy(c, K, V)
            P = c.block_tokens // c.summary_tokens
            Kp = K.astype(np.float32).reshape(P, c.summary_tokens, c.head_dim)
            kmin = Kp.min(axis=1).astype(np.float16)
            kmax = Kp.max(axis=1).astype(np.float16)

        # Write to disk
        wf = self._get_write_file(layer)
        with self._wlock:
            wf.seek_and_write(self._offset(head, block), data)

        # Update Host-RAM summary index
        self.summ.setdefault((layer, head), {})[block] = (kmin, kmax)
        self._stack_cache.pop((layer, head), None)

    def _read_block_raw(self, layer: int, head: int, block: int) -> bytes:
        rf = self._get_read_file(layer)
        return rf.seek_and_read(self._offset(head, block), self.rec)

    def get_blocks(
        self,
        layer: int,
        head: int,
        blocks: List[int],
        as_torch: bool = True,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Read requested blocks in parallel -> (K, V) concatenated [len(blocks)*T, D]."""
        if not blocks:
            if as_torch:
                empty = torch.empty((0, self.cfg.head_dim), dtype=torch.float32, device=device)
                return empty, empty
            return np.empty((0, self.cfg.head_dim), dtype=np.float32), np.empty((0, self.cfg.head_dim), dtype=np.float32)

        futures = [self.pool.submit(self._read_block_raw, layer, head, b) for b in blocks]
        raw_bufs = [f.result() for f in futures]

        if as_torch:
            decoded = [decode_block_torch(self.cfg, buf, device=device) for buf in raw_bufs]
            K_out = torch.cat([d[0] for d in decoded], dim=0)
            V_out = torch.cat([d[1] for d in decoded], dim=0)
            return K_out, V_out
        else:
            decoded = [decode_block_numpy(self.cfg, buf) for buf in raw_bufs]
            K_out = np.concatenate([d[0] for d in decoded], axis=0)
            V_out = np.concatenate([d[1] for d in decoded], axis=0)
            return K_out, V_out

    def select_blocks(
        self,
        layer: int,
        head: int,
        q: Union[np.ndarray, torch.Tensor],
        k: int,
        must_include: Tuple[int, ...] = (),
    ) -> List[int]:
        """
        Quest-style upper-bound score selection:
        Finds the top-k blocks whose Key bounding boxes maximize q . k.
        """
        key = (layer, head)
        if key not in self.summ or not self.summ[key]:
            return list(must_include)

        if key not in self._stack_cache:
            ids = sorted(self.summ[key].keys())
            kmin_stack = torch.from_numpy(np.stack([self.summ[key][i][0] for i in ids])).to(torch.float32)
            kmax_stack = torch.from_numpy(np.stack([self.summ[key][i][1] for i in ids])).to(torch.float32)
            self._stack_cache[key] = (ids, kmin_stack, kmax_stack)

        ids, kmin, kmax = self._stack_cache[key]

        if isinstance(q, np.ndarray):
            q_t = torch.from_numpy(q).to(torch.float32)
        else:
            q_t = q.detach().cpu().to(torch.float32)

        # Upper bound of q . k per summary page, then max page represents the block
        # Shape: [num_blocks, P, D] * [D] -> sum(-1) -> [num_blocks, P] -> max(-1) -> [num_blocks]
        score = torch.maximum(q_t * kmin, q_t * kmax).sum(dim=-1).amax(dim=-1)

        top_indices = torch.argsort(score, descending=True)[:k].tolist()
        chosen = {ids[i] for i in top_indices}
        for b in must_include:
            if b in self.summ[key]:
                chosen.add(b)
        return sorted(chosen)

    def save_index(self):
        """Save RAM summary index to index.npz and metadata to meta.json."""
        arrs = {}
        for (l, h), d in self.summ.items():
            ids = sorted(d.keys())
            arrs[f"l{l}_h{h}_ids"] = np.array(ids, dtype=np.int32)
            arrs[f"l{l}_h{h}_kmin"] = np.stack([d[i][0] for i in ids])
            arrs[f"l{l}_h{h}_kmax"] = np.stack([d[i][1] for i in ids])
        np.savez(os.path.join(self.root, "index.npz"), **arrs)

        meta = {**asdict(self.cfg), "kv_bits": KV_BITS}
        with open(os.path.join(self.root, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(
        cls,
        root: str,
        io_threads: int = 8,
        use_direct_io: bool = True,
    ) -> "Q4KVStore":
        """Load an existing store and reconstruct its Host-RAM summary index."""
        with open(os.path.join(root, "meta.json")) as f:
            meta = json.load(f)
        assert meta.pop("kv_bits") == KV_BITS, "Mismatched kv_bits in meta.json"
        cfg = Cfg(**meta)
        store = cls(root, cfg, io_threads=io_threads, use_direct_io=use_direct_io)

        index_path = os.path.join(root, "index.npz")
        if os.path.exists(index_path):
            z = np.load(index_path)
            for name in z.files:
                if name.endswith("_ids"):
                    prefix = name[:-4]  # e.g. "l0_h0"
                    l, h = map(int, prefix.replace("l", "").replace("h", "").split("_"))
                    ids = z[name]
                    kmin, kmax = z[f"{prefix}_kmin"], z[f"{prefix}_kmax"]
                    store.summ[(l, h)] = {
                        int(b): (kmin[i], kmax[i]) for i, b in enumerate(ids)
                    }
        return store

    def close(self):
        """Close write handles and shutdown thread pool."""
        with self._wlock:
            for wf in self._wfiles.values():
                wf.close()
            self._wfiles.clear()
        self.pool.shutdown(wait=True)
