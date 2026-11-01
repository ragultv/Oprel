import os
import sys
import time
import tempfile
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend

MODEL_PATH = r"C:\Users\tragu\.cache\oprel\models\models--unsloth--gemma-3-4b-it-GGUF\snapshots\5a3566e716d80f709ed7b79817eaf7733d2a1fce\gemma-3-4b-it-Q5_K_M.gguf"

def test_session_persistence_and_invalidation():
    assert os.path.exists(MODEL_PATH), f"Model path not found: {MODEL_PATH}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"[TEST] Using temporary SSD cache dir: {tmp_dir}", flush=True)

        backend = LlamaSSDNativeBackend()
        backend.load_model(MODEL_PATH, n_gpu_layers=33)
        backend.create_context(ssd_cache_dir=tmp_dir, active_ctx_size=1024, max_ssd_context=32768)

        # 1. Set session 'test_sess_1'
        ok = backend.set_session("test_sess_1")
        assert ok, "Failed to set session test_sess_1"
        print("[TEST] Swapped to test_sess_1", flush=True)

        # 2. Ingest prompt
        prompt = "<start_of_turn>user\nHello world from test session!<end_of_turn>\n<start_of_turn>model\n"
        n_tokens = backend.ingest(prompt, reset=True)
        print(f"[TEST] Ingested {n_tokens} prompt tokens", flush=True)
        assert n_tokens > 0

        # 3. Save session explicitly using new oprel_save_session API
        save_ok = backend.save_session()
        assert save_ok, "oprel_save_session failed"
        print("[TEST] Explicit save_session succeeded!", flush=True)

        # Check that session file exists on disk
        sess_bin = os.path.join(tmp_dir, "sessions", "session_test_sess_1.bin")
        assert os.path.exists(sess_bin), f"Session file not found: {sess_bin}"
        file_sz = os.path.getsize(sess_bin)
        print(f"[TEST] session_test_sess_1.bin size on disk: {file_sz} bytes", flush=True)
        assert file_sz > 0

        # 4. Switch to a new session 'test_sess_2'
        ok2 = backend.set_session("test_sess_2")
        assert ok2, "Failed to switch to test_sess_2"
        stats2 = backend.get_stats()
        print(f"[TEST] test_sess_2 initialized with total_tokens={stats2.get('total_tokens')}", flush=True)
        assert stats2.get("total_tokens") == 0

        # 5. Switch BACK to 'test_sess_1' - must instantly recall tokens without re-prefill!
        t0 = time.perf_counter()
        ok_recall = backend.set_session("test_sess_1")
        t_recall = time.perf_counter() - t0
        assert ok_recall, "Failed to restore test_sess_1"
        stats1 = backend.get_stats()
        print(f"[TEST] Restored test_sess_1 in {t_recall*1000:.2f} ms: total_tokens={stats1.get('total_tokens')}", flush=True)
        assert stats1.get("total_tokens") == n_tokens, f"Expected {n_tokens} tokens, got {stats1.get('total_tokens')}"

        # 6. Test Incompatible Session Invalidation:
        # Create a fake corrupted/old version 1 session file
        bad_sess_bin = os.path.join(tmp_dir, "sessions", "session_bad_sess.bin")
        with open(bad_sess_bin, "wb") as f:
            f.write(b"\x4F\x50\x53\x4E") # 'OPSN'
            f.write((1).to_bytes(4, "little")) # Version 1 (incompatible)
            f.write(b"\x00" * 64)

        # Attempt to load bad session
        print("[TEST] Testing rejection of incompatible version 1 session...", flush=True)
        bad_ok = backend.set_session("bad_sess")
        assert bad_ok, "set_session should fall back to fresh session"
        # The bad session bin should have been deleted by the rejection logic
        assert not os.path.exists(bad_sess_bin), "Stale session file was not invalidated/removed!"
        print("[TEST] Stale session file was cleanly invalidated and removed from disk!", flush=True)

        backend.close()
        print("\n[TEST] ALL PERSISTENCE AND INVALIDATION TESTS PASSED!", flush=True)

if __name__ == "__main__":
    test_session_persistence_and_invalidation()
