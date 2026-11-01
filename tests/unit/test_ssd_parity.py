"""Cached-vs-uncached parity and position coverage for the native SSD KV archive.

Reference: a context that prefills the whole prompt once and never evicts.
Cached:    the same prompt, with oldest tokens archived to SSD (``archive_oldest``) and restored.

All prompts stay inside Gemma-3's sliding window (n_swa = 1024), so every archived block holds the
KV of *all* layers and a restore can be compared exactly. Eviction compacts positions, so the
restored layout (sinks | restored blocks | tail) is only comparable to the reference when every
archived block is restored: that puts each token back at its original position, and it is the
only case compared here. Parity after eviction *without* restore (compacted positions) is
deliberately not compared; it is measured only as the "block absent" control.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend

MODEL_PATH = r"C:\Users\tragu\.cache\oprel\models\models--unsloth--gemma-3-4b-it-GGUF\snapshots\5a3566e716d80f709ed7b79817eaf7733d2a1fce\gemma-3-4b-it-Q5_K_M.gguf"
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="GGUF model not available")

RESULTS_PATH = os.environ.get("OPREL_PARITY_RESULTS")  # optional JSON dump for the write-up
N_GREEDY = 16

# Tolerances, set from measurements on this machine (docs/dev-guide/ssd_kv_validation.md) with
# roughly 2-7x headroom. Logits are compared at the last prompt position, whose next token is the
# first digit of a code stored in the archived block.
#   f16:  restored KL 2e-7..1.5e-6 (noise floor 1.6e-7), top-20 |dlogit| 0.19..0.26 (floor 0.31)
#   q4_0: restored KL 4e-5..1.5e-4 (noise floor 4.5e-6), top-20 |dlogit| 1.4..2.4  (floor 0.30)
#         the extra error comes from re-quantising K when the lazy K-shift rotates Q4_0 cells.
# The "block absent" control (last token re-decoded without the archived block) measured KL 3e-3..8
# and must stay at least 10x above the restored error.
TOLERANCE = {
    "f16": {"kl": 1e-5, "top20_max_abs": 0.75},
    "q4_0": {"kl": 1e-3, "top20_max_abs": 4.0},
}


def prompt(n_records: int = 40) -> str:
    body = " ".join(f"Record {i}: the vault code for room {i} is {(i * 37) % 1000}." for i in range(n_records))
    # Ends so that the next token is the first digit of the code: it depends on the archived records.
    return (f"<start_of_turn>user\n{body}\nWhat is the vault code for room 7?<end_of_turn>\n"
            f"<start_of_turn>model\nThe vault code for room 7 is ")


def metrics(ref: np.ndarray, got: np.ndarray) -> dict:
    d = np.abs(ref - got)
    p = np.exp(ref - ref.max()); p /= p.sum()
    q = np.exp(got - got.max()); q /= q.sum()
    kl = float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
    top_ref = set(np.argsort(ref)[-10:].tolist())
    top_got = set(np.argsort(got)[-10:].tolist())
    top20 = np.argsort(ref)[-20:]
    return {
        "top20_max_abs": float(np.abs(ref[top20] - got[top20]).max()),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "cos": float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got))),
        "kl": kl,
        "argmax_equal": bool(ref.argmax() == got.argmax()),
        "top10_overlap": len(top_ref & top_got),
    }


_RECORDED = {}


def record(key, value):
    _RECORDED[key] = value
    if RESULTS_PATH:
        Path(RESULTS_PATH).write_text(json.dumps(_RECORDED, indent=2), encoding="utf-8")
    print(f"\n[PARITY] {key}: {json.dumps(value)}")


@pytest.fixture(scope="module")
def backend():
    b = LlamaSSDNativeBackend()
    b.load_model(MODEL_PATH, n_gpu_layers=33)
    yield b
    b.close()


@pytest.fixture
def make_ctx(backend, tmp_path, monkeypatch):
    counter = {"n": 0}

    def make(kv_type: str):
        monkeypatch.setenv("OPREL_SSD_KV_TYPE", kv_type)
        if backend.context_handle:
            backend.dll.oprel_free_context(backend.context_handle)
            backend.context_handle = None
        counter["n"] += 1
        d = tmp_path / f"ctx{counter['n']}"
        d.mkdir()
        backend.create_context(ssd_cache_dir=str(d), active_ctx_size=4096, max_ssd_context=32768)
        backend.set_autonomous_restore(False)
        assert backend.set_session(f"parity{counter['n']}")
        return d

    yield make
    if backend.context_handle:
        backend.dll.oprel_free_context(backend.context_handle)
        backend.context_handle = None


def logits(backend) -> np.ndarray:
    return np.array(backend.get_logits(), dtype=np.float32)


def greedy(backend, n=N_GREEDY):
    toks = []
    for _ in range(n):
        t = backend.generate_next_token(temperature=0.0)
        assert t >= 0
        toks.append(t)
    return toks


def reference(backend, make_ctx, kv):
    make_ctx(kv)
    n = backend.ingest(prompt(), reset=True)
    assert 600 < n < 1000, f"prompt must stay inside the SWA window, got {n} tokens"
    ref = logits(backend)
    assert backend.refresh_logits()
    floor = logits(backend)  # same cache, last token re-decoded alone: the batch-vs-single noise floor
    record(f"{kv}/noise_floor", metrics(ref, floor))
    return n, ref, greedy(backend)


def check(kv, label, ref, got, control=None):
    m = metrics(ref, got)
    entry = {"kv_type": kv, **m}
    if control is not None:
        entry["control_block_absent"] = metrics(ref, control)
    record(f"{kv}/{label}", entry)
    tol = TOLERANCE[kv]
    assert m["argmax_equal"], f"{label}: next-token argmax differs"
    assert m["kl"] <= tol["kl"], f"{label}: KL {m['kl']:.3e} > {tol['kl']}"
    assert m["top20_max_abs"] <= tol["top20_max_abs"], f"{label}: top-20 |dlogit| {m['top20_max_abs']:.4f} > {tol['top20_max_abs']}"
    if control is not None:
        c = entry["control_block_absent"]
        assert c["kl"] > 10 * m["kl"], "control is not sensitive: block-absent KL is not clearly larger"


def noblock_logits(backend):
    """Logits at the last position with the archived block absent (applies the pending shift)."""
    backend.reserialize_range(4, 8)  # serialising forces the K-shift refresh decode
    return logits(backend)


@pytest.mark.parametrize("kv", ["f16", "q4_0"])
def test_single_block_restore_matches_uncached_logits_and_greedy_tokens(backend, make_ctx, kv):
    n, ref, ref_tokens = reference(backend, make_ctx, kv)

    make_ctx(kv)
    assert backend.ingest(prompt(), reset=True) == n
    blk = backend.archive_oldest(252)
    assert blk == 0
    absent = noblock_logits(backend)

    before = backend.get_stats()
    assert backend.restore_kv_state(blk)
    after = backend.get_stats()
    cached = logits(backend)

    # Positions: the block is back at its original place and the live window is whole again.
    assert (after["last_restore_pos_start"], after["last_restore_pos_end"]) == (4, 256)
    assert after["active_tokens"] == after["total_tokens"] == n
    # The block's 252 tokens were not decoded: one refresh decode of the last live token only.
    assert after["decode_calls"] == before["decode_calls"] + 1
    assert after["restore_decode_calls"] == 1 and after["restore_prefill_tokens"] == 0
    check(kv, "single_block", ref, cached, control=absent)

    assert greedy(backend) == ref_tokens, "greedy continuation differs from the uncached run"


@pytest.mark.parametrize("kv", ["f16", "q4_0"])
@pytest.mark.parametrize("order", ["in_order", "reversed"])
def test_later_block_after_lazy_k_shift_restores_exact_positions(backend, make_ctx, kv, order):
    """Block 1 is archived after block 0's tail shift and restored 200 positions to the right of
    where it was stored: its K vectors must be re-rotated by exactly that delta."""
    n, ref, ref_tokens = reference(backend, make_ctx, kv)

    make_ctx(kv)
    backend.ingest(prompt(), reset=True)
    assert backend.archive_oldest(200) == 0   # original tokens [4, 204)
    assert backend.archive_oldest(200) == 1   # original tokens [204, 404), stored at live [4, 204)
    absent = noblock_logits(backend)
    assert backend.get_stats()["active_tokens"] == n - 400

    for b in ([0, 1] if order == "in_order" else [1, 0]):
        assert backend.restore_kv_state(b)
    st = backend.get_stats()
    assert st["active_tokens"] == st["total_tokens"] == n
    assert sorted(backend.get_active_restored_blocks()) == [0, 1]
    check(kv, f"later_block_{order}", ref, logits(backend), control=absent)
    assert greedy(backend) == ref_tokens


@pytest.mark.parametrize("kv", ["f16", "q4_0"])
def test_rearchiving_a_restored_range_keeps_global_positions(backend, make_ctx, kv):
    """Restored cells must leave the window before the next eviction; otherwise they would be
    archived again under the wrong global positions."""
    n, ref, ref_tokens = reference(backend, make_ctx, kv)

    make_ctx(kv)
    backend.ingest(prompt(), reset=True)
    assert backend.archive_oldest(200) == 0
    assert backend.restore_kv_state(0)
    assert backend.get_active_restored_blocks() == [0]

    # Evict again while block 0's cells are live at the front of the window.
    assert backend.archive_oldest(300) == 1
    st = backend.get_stats()
    assert backend.get_active_restored_blocks() == [], "restored cells were not dropped before evicting"
    assert st["active_tokens"] == n - 500, "restored cells were archived a second time"
    spans = {b["block_index"]: (b["p0"], b["n_tokens"]) for b in backend.score_all_blocks("record", max_blocks=8)}
    assert spans[0] == (4, 200)
    assert spans[1] == (204, 300), f"block 1 must start at original token 204, got {spans[1]}"

    for b in (0, 1):
        assert backend.restore_kv_state(b)
    st = backend.get_stats()
    assert st["active_tokens"] == st["total_tokens"] == n
    check(kv, "rearchive", ref, logits(backend))
    assert greedy(backend) == ref_tokens
