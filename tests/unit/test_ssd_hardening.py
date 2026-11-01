"""Hardening tests for the native SSD KV archive (eviction, restore, sessions, caps, concurrency).

These run the real native wrapper against a real GGUF model. They are skipped only when the
model or the built wrapper DLL is not present on the machine.
"""
import os
import sys
import threading
from pathlib import Path

import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend

MODEL_PATH = r"C:\Users\tragu\.cache\oprel\models\models--unsloth--gemma-3-4b-it-GGUF\snapshots\5a3566e716d80f709ed7b79817eaf7733d2a1fce\gemma-3-4b-it-Q5_K_M.gguf"

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="GGUF model not available")

ACTIVE_CTX = 1024
BLOCK_HEADER_BYTES = 120  # sizeof(OprelSSDBlockHeader)
SECTOR = 4096


def long_prompt(n_records: int = 200) -> str:
    body = " ".join(f"Record {i}: the vault code for room {i} is {(i * 37) % 1000}." for i in range(n_records))
    return f"<start_of_turn>user\n{body}\nWhat is the vault code for room 7?<end_of_turn>\n<start_of_turn>model\n"


SHORT = "<start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n"


@pytest.fixture(scope="module")
def backend():
    b = LlamaSSDNativeBackend()
    b.load_model(MODEL_PATH, n_gpu_layers=33)
    yield b
    b.close()


@pytest.fixture
def fresh_ctx(backend, tmp_path, monkeypatch):
    """Returns a function creating a brand-new context in tmp_path (freeing the previous one)."""

    def make(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        if backend.context_handle:
            backend.dll.oprel_free_context(backend.context_handle)
            backend.context_handle = None
        backend.create_context(ssd_cache_dir=str(tmp_path), active_ctx_size=ACTIVE_CTX, max_ssd_context=32768)
        return tmp_path

    yield make
    if backend.context_handle:
        backend.dll.oprel_free_context(backend.context_handle)
        backend.context_handle = None


def block_file(tmp: Path, sid: str) -> Path:
    return tmp / f"{sid}.kv"


def test_eviction_archives_blocks_and_payload_is_readable(backend, fresh_ctx):
    fresh_ctx()
    assert backend.set_session("evict")
    n = backend.ingest(long_prompt(), reset=True)
    st = backend.get_stats()
    assert n > ACTIVE_CTX, f"prompt too short to force eviction ({n} tokens)"
    assert st["ssd_blocks"] >= 1
    assert st["total_ssd_bytes_written"] > 0
    payload = backend.get_block_payload(0)
    assert len(payload) == st["last_block_bytes"] or len(payload) > 0


def test_restore_reads_ssd_and_does_not_decode_prefix(backend, fresh_ctx):
    """Evidence that a restore is a KV injection, not a re-decode of the cached prefix.

    Inject-only (refresh_logits=False): llama_decode is not called at all. Default restore: exactly
    one single-token decode of the last live token (applies the lazy K-shift and refreshes the
    logits); the block's own tokens are never decoded in either case.
    """
    fresh_ctx()
    backend.set_autonomous_restore(False)
    assert backend.set_session("restore")
    backend.ingest(long_prompt(), reset=True)
    before = backend.get_stats()
    assert before["ssd_blocks"] >= 2

    assert backend.restore_kv_state(0, refresh_logits=False), "restore of block 0 failed"
    after = backend.get_stats()

    assert after["decode_calls"] == before["decode_calls"], "inject-only restore called llama_decode"
    assert after["restore_decode_calls"] == 0
    assert after["restore_prefill_tokens"] == 0
    assert after["total_ssd_bytes_read"] > before["total_ssd_bytes_read"], "no bytes were read from SSD"
    assert after["archive_validation_passes"] == before["archive_validation_passes"] + 1
    assert after["last_restore_pos_start"] == 4
    assert after["last_restore_pos_end"] > after["last_restore_pos_start"]
    assert 0 in backend.get_active_restored_blocks()

    # Default restore of the next block: one refresh decode, still zero prefill tokens.
    assert backend.restore_kv_state(1)
    refreshed = backend.get_stats()
    assert refreshed["decode_calls"] == after["decode_calls"] + 1
    assert refreshed["restore_decode_calls"] == 1 and refreshed["restore_prefill_tokens"] == 0
    assert refreshed["last_restore_pos_start"] == after["last_restore_pos_end"], "block 1 must follow block 0"

    # 764 restored tokens overflow this 1024-token window, so the next generated token first evicts:
    # the restored blocks leave the window (never re-archived), a refresh decode applies the pending
    # K-shift, then the token itself is decoded = exactly two decode calls, and one new block.
    blocks_before = refreshed["ssd_blocks"]
    tok = backend.generate_next_token(temperature=0.0)
    assert tok >= 0
    final = backend.get_stats()
    assert final["decode_calls"] == refreshed["decode_calls"] + 2
    assert final["ssd_blocks"] == blocks_before + 1
    assert backend.get_active_restored_blocks() == []


def test_restored_cells_are_bit_identical_to_the_archived_payload(backend, fresh_ctx):
    """Re-serialise the restored live range and compare it with the bytes stored on SSD.

    Block 0 is archived at its live positions ([4, 4+n)), so no K-shift is involved and the
    restored cells must round-trip exactly: same positions, same quantised K/V bytes.
    """
    fresh_ctx()
    backend.set_autonomous_restore(False)
    assert backend.set_session("bitexact")
    backend.ingest(long_prompt(), reset=True)
    stored = backend.get_block_payload(0)
    assert len(stored) > 0
    assert backend.restore_kv_state(0)
    st = backend.get_stats()
    assert (st["last_restore_pos_start"], st["last_restore_pos_end"]) == (4, 4 + (st["last_restore_pos_end"] - 4))
    live = backend.reserialize_range(st["last_restore_pos_start"], st["last_restore_pos_end"])
    assert len(live) == len(stored)
    assert live == stored, "restored KV differs from the archived payload"


def test_corrupt_and_truncated_block_file_is_rejected_and_inference_continues(backend, fresh_ctx):
    tmp = fresh_ctx()
    backend.set_autonomous_restore(False)
    assert backend.set_session("blk")
    backend.ingest(long_prompt(), reset=True)
    n_blocks = backend.get_stats()["ssd_blocks"]
    assert n_blocks >= 1
    assert backend.save_session()

    # Fresh context so the block file is closed and can be damaged on disk.
    fresh_ctx_dir = fresh_ctx()
    assert fresh_ctx_dir == tmp
    kv = block_file(tmp, "blk")
    raw = bytearray(kv.read_bytes())
    assert len(raw) >= SECTOR

    # 1) Flip one payload byte of block 0 -> CRC must reject it.
    raw[BLOCK_HEADER_BYTES + 100] ^= 0xFF
    kv.write_bytes(bytes(raw))
    assert backend.set_session("blk"), "set_session must load the session"
    assert backend.get_stats()["ssd_blocks"] == n_blocks
    before = backend.get_stats()
    assert backend.restore_kv_state(0) is False
    after = backend.get_stats()
    assert after["archive_validation_failures"] == before["archive_validation_failures"] + 1
    assert after["decode_calls"] == before["decode_calls"]

    # 2) Truncate the block file mid-way: blocks that no longer fit become unrestorable on load.
    assert backend.save_session()
    fresh_ctx()
    kv.write_bytes(bytes(raw[: len(raw) // 2]))
    assert backend.set_session("blk")
    st = backend.get_stats()
    last = st["ssd_blocks"] - 1
    assert backend.restore_kv_state(last) is False, "block beyond EOF must not restore"

    # Fallback: uncached inference still works on this context.
    backend.ingest(SHORT, reset=True)
    assert backend.generate_next_token(temperature=0.0) >= 0


def test_corrupt_session_files_are_rejected_and_removed(backend, fresh_ctx):
    tmp = fresh_ctx()
    assert backend.set_session("good")
    backend.ingest(SHORT, reset=True)
    assert backend.save_session()
    sess = tmp / "sessions" / "session_good.bin"
    original = sess.read_bytes()
    assert not list((tmp / "sessions").glob("*.tmp")), "atomic save left a temp file"

    for label, mutated in {
        "truncated": original[: len(original) // 2],
        "bitflip": original[:200] + bytes([original[200] ^ 0x01]) + original[201:],
        "tiny": original[:6],
    }.items():
        assert backend.set_session("other")
        sess.write_bytes(mutated)
        assert backend.set_session("good"), f"{label}: must fall back to a fresh session"
        assert not sess.exists(), f"{label}: corrupt session file was not removed"
        assert backend.get_stats()["total_tokens"] == 0, f"{label}: state leaked from a rejected session"
        backend.ingest(SHORT, reset=True)
        assert backend.save_session()
        original = sess.read_bytes()


def test_session_id_cannot_escape_cache_dir(backend, fresh_ctx):
    tmp = fresh_ctx()
    outside_before = set(os.listdir(tmp.parent))
    for evil in ["../../escape", "..\\..\\escape", "a/b/c", ".hidden", "con:x"]:
        assert backend.set_session(evil)
        backend.ingest(SHORT, reset=True)
        assert backend.save_session()
    assert set(os.listdir(tmp.parent)) == outside_before, "files were created outside the cache dir"
    for p in (tmp / "sessions").iterdir():
        assert ".." not in p.name and "/" not in p.name and "\\" not in p.name
    # Distinct unsafe IDs must not collide after sanitising.
    assert len(list((tmp / "sessions").glob("session_*.bin"))) == 5


def test_block_file_cap_drops_instead_of_writing_and_decoding_continues(backend, fresh_ctx):
    tmp = fresh_ctx(OPREL_SSD_MAX_BYTES=1)
    backend.set_autonomous_restore(False)
    assert backend.set_session("cap")
    n = backend.ingest(long_prompt(), reset=True)
    assert n > ACTIVE_CTX
    st = backend.get_stats()
    assert st["ssd_blocks"] >= 1, "dropped ranges must still be accounted for"
    assert st["total_ssd_bytes_written"] == 0
    assert block_file(tmp, "cap").stat().st_size == 0
    assert backend.restore_kv_state(0) is False
    assert backend.generate_next_token(temperature=0.0) >= 0


def test_reset_reclaims_block_file(backend, fresh_ctx):
    tmp = fresh_ctx()
    assert backend.set_session("reset")
    backend.ingest(long_prompt(), reset=True)
    assert block_file(tmp, "reset").stat().st_size > 0
    backend.ingest(SHORT, reset=True)
    st = backend.get_stats()
    assert st["ssd_blocks"] == 0
    assert block_file(tmp, "reset").stat().st_size == 0, "old archive bytes were not reclaimed"


def test_session_count_is_bounded(backend, fresh_ctx):
    tmp = fresh_ctx(OPREL_SSD_MAX_SESSIONS=2)
    for sid in ["s1", "s2", "s3", "s4"]:
        assert backend.set_session(sid)
        backend.ingest(SHORT, reset=True)
        assert backend.save_session()
    files = sorted(p.name for p in (tmp / "sessions").glob("session_*.bin"))
    assert len(files) <= 2, files
    assert "session_s4.bin" in files, "the active session must never be pruned"
    # A pruned session's block file goes with it.
    for sid in ["s1", "s2", "s3"]:
        if f"session_{sid}.bin" not in files:
            assert not block_file(tmp, sid).exists()


def test_restored_block_tracking_survives_session_reload(backend, fresh_ctx):
    fresh_ctx()
    backend.set_autonomous_restore(False)
    assert backend.set_session("track")
    backend.ingest(long_prompt(), reset=True)
    assert backend.restore_kv_state(0)
    assert backend.get_active_restored_blocks() == [0]
    assert backend.save_session()
    assert backend.set_session("other")
    assert backend.get_active_restored_blocks() == []
    assert backend.set_session("track")
    assert backend.get_active_restored_blocks() == [0], "restored block was forgotten on reload"
    # Restoring it again must be a no-op rather than inserting the cells twice.
    d0 = backend.get_stats()["decode_calls"]
    assert backend.restore_kv_state(0)
    assert backend.get_stats()["decode_calls"] == d0


def test_concurrent_calls_on_one_context_do_not_corrupt_state(backend, fresh_ctx):
    fresh_ctx()
    assert backend.set_session("conc")
    errors = []

    def worker(i):
        try:
            for _ in range(3):
                backend.ingest(SHORT, reset=True)
                backend.get_stats()
                tok = backend.generate_next_token(temperature=0.0)
                assert tok >= 0
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join(timeout=240) for t in threads]
    assert not errors, errors
    assert all(not t.is_alive() for t in threads)
    backend.ingest(SHORT, reset=True)
    assert backend.generate_next_token(temperature=0.0) >= 0
