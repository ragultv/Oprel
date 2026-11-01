"""Configuration for 4-bit KV cache SSD store."""
from __future__ import annotations

from dataclasses import dataclass

KV_BITS = 4
LEVELS = (1 << KV_BITS) - 1  # 15
PAGE_SIZE = 4096


@dataclass
class Cfg:
    n_layers: int
    n_kv_heads: int
    head_dim: int = 128
    block_tokens: int = 256
    group: int = 32
    summary_tokens: int = 16  # RAM index granularity (finer than the SSD block)

    def __post_init__(self):
        assert self.block_tokens % self.summary_tokens == 0, (
            f"block_tokens ({self.block_tokens}) must be divisible by summary_tokens ({self.summary_tokens})"
        )
        assert self.block_tokens % self.group == 0, (
            f"block_tokens ({self.block_tokens}) must be divisible by group ({self.group})"
        )
        assert self.head_dim % self.group == 0, (
            f"head_dim ({self.head_dim}) must be divisible by group ({self.group})"
        )
        assert self.head_dim % 2 == 0, (
            f"head_dim ({self.head_dim}) must be even for 4-bit pair packing"
        )

    def record_size(self) -> int:
        """Fixed record size per (block, kv_head) padded to PAGE_SIZE (4096 bytes)."""
        T, D, G = self.block_tokens, self.head_dim, self.group
        # [K packed uint4: T*D/2] + [K scale fp16: (T/G)*D*2] + [K min fp16: (T/G)*D*2]
        # + [V packed uint4: T*D/2] + [V scale fp16: T*(D/G)*2] + [V min fp16: T*(D/G)*2]
        raw = 2 * (T * D // 2) + 2 * 2 * (T // G) * D + 2 * 2 * T * (D // G)
        return (raw + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
