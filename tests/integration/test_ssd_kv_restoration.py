import os
from pathlib import Path

import pytest

from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend


MODEL_PATH = os.environ.get("OPREL_TEST_MODEL")
LLAMA_DLL_PATH = os.environ.get("OPREL_TEST_LLAMA_DLL")
WRAPPER_DLL_PATH = os.environ.get("OPREL_TEST_WRAPPER_DLL")


pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not LLAMA_DLL_PATH or not WRAPPER_DLL_PATH or not Path(MODEL_PATH).exists(),
    reason="Set OPREL_TEST_MODEL, OPREL_TEST_LLAMA_DLL, and OPREL_TEST_WRAPPER_DLL for native integration tests",
)


def _new_backend(tmp_path, active_ctx_size=1024):
    backend = LlamaSSDNativeBackend(
        llama_dll_path=LLAMA_DLL_PATH,
        wrapper_dll_path=WRAPPER_DLL_PATH,
    )
    backend.load_model(MODEL_PATH, n_gpu_layers=0)
    backend.create_context(
        ssd_cache_dir=str(tmp_path),
        active_ctx_size=active_ctx_size,
        max_ssd_context=8192,
        retrieval_top_k=1,
    )
    backend.set_autonomous_restore(False)
    return backend


def _archive_prompt():
    return (
        "The archival marker is cobalt-orbit-731. "
        "This sentence is deliberately repeated to force a deterministic archive. "
        * 120
    )


def _archive_file(tmp_path):
    files = list(Path(tmp_path).glob("*.kv"))
    assert len(files) == 1
    return files[0]


def test_restore_reads_archive_without_decode_and_preserves_positions(tmp_path):
    backend = _new_backend(tmp_path)
    reference = _new_backend(tmp_path / "reference", active_ctx_size=4096)
    try:
        assert backend.ingest(_archive_prompt(), reset=True) > 0
        query = " Continue with the archival marker and answer in one short phrase."
        assert reference.ingest(_archive_prompt(), reset=True) > 0
        assert reference.ingest(query, reset=False) > 0
        reference_logits = reference.get_logits()

        before = backend.get_stats()
        assert before["ssd_blocks"] > 0
        block_payload = backend.get_block_payload(0)
        assert block_payload

        decode_before = before["decode_calls"]
        assert backend.restore_kv_state(0)
        after = backend.get_stats()

        assert after["total_ssd_bytes_read"] > before["total_ssd_bytes_read"]
        assert after["archive_validation_passes"] == before["archive_validation_passes"] + 1
        assert after["decode_calls"] == decode_before
        assert after["restore_decode_calls"] == after["decode_calls"] - decode_before
        assert after["restore_prefill_tokens"] == 0
        assert after["last_restore_kv_match"] == -1
        assert after["last_restore_pos_start"] == 4
        assert after["last_restore_pos_end"] > after["last_restore_pos_start"]

        assert backend.ingest(query, reset=False) > 0
        restored_logits = backend.get_logits()
        assert len(restored_logits) == len(reference_logits)
        assert max(
            abs(a - b) for a, b in zip(restored_logits, reference_logits)
        ) <= 1e-4
    finally:
        backend.close()
        reference.close()


def test_restore_rejects_corrupted_archive_before_touching_live_kv(tmp_path):
    backend = _new_backend(tmp_path)
    try:
        assert backend.ingest(_archive_prompt(), reset=True) > 0
        before = backend.get_stats()
        archive = _archive_file(tmp_path)
        with archive.open("r+b") as fp:
            fp.seek(128)
            original = fp.read(1)
            fp.seek(128)
            fp.write(bytes([original[0] ^ 0x01]))

        assert not backend.restore_kv_state(0)
        after = backend.get_stats()
        assert after["archive_validation_failures"] == before["archive_validation_failures"] + 1
        assert after["restore_decode_calls"] == 0
        assert after["restore_prefill_tokens"] == 0
    finally:
        backend.close()


def test_restore_rejects_mismatched_model_identity(tmp_path):
    backend = _new_backend(tmp_path)
    try:
        assert backend.ingest(_archive_prompt(), reset=True) > 0
        before = backend.get_stats()
        archive = _archive_file(tmp_path)
        with archive.open("r+b") as fp:
            fp.seek(16)
            original = fp.read(1)
            fp.seek(16)
            fp.write(bytes([original[0] ^ 0x01]))

        assert not backend.restore_kv_state(0)
        after = backend.get_stats()
        assert after["archive_validation_failures"] == before["archive_validation_failures"] + 1
        assert after["restore_decode_calls"] == 0
    finally:
        backend.close()
