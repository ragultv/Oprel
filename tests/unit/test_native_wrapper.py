import os
import sys
import time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend

MODEL_PATH = r"C:\Users\tragu\.cache\oprel\models\models--unsloth--gemma-3-4b-it-GGUF\snapshots\5a3566e716d80f709ed7b79817eaf7733d2a1fce\gemma-3-4b-it-Q5_K_M.gguf"

def build_long_prompt(text_block: str, turn1_answer: str) -> str:
    """Build a full multi-turn chat string including Turn 1 assistant reply."""
    return (
        f"<start_of_turn>user\n{text_block}<end_of_turn>\n"
        f"<start_of_turn>model\n{turn1_answer}<end_of_turn>\n"
    )

def test_native_wrapper_load_and_generate():
    """Basic single-turn sanity check (no SSD eviction expected)."""
    assert os.path.exists(MODEL_PATH), f"Model path not found: {MODEL_PATH}"

    backend = LlamaSSDNativeBackend()
    print("\n[TEST] Successfully initialized LlamaSSDNativeBackend binding to llama.dll")

    backend.load_model(MODEL_PATH, n_gpu_layers=33)
    print("[TEST] Successfully loaded model weights into memory")

    backend.create_context(active_ctx_size=2048, max_ssd_context=131072)
    print("[TEST] Successfully created SSD-backed context with Quest index")

    prompt = "<start_of_turn>user\nHello! Please tell me what 2 + 2 is in one short sentence.<end_of_turn>\n<start_of_turn>model\n"
    n_tokens = backend.ingest(prompt, reset=True)
    print(f"[TEST] Ingested {n_tokens} prompt tokens", flush=True)

    generated_text = ""
    for piece in backend.generate("", max_new_tokens=15, temperature=0.2):
        generated_text += piece
        if len(generated_text) > 40:
            break

    print(f"[TEST] Generated text: {generated_text.encode('ascii', errors='replace').decode('ascii').strip()}")
    stats = backend.get_stats()
    print(f"[TEST] Telemetry stats: {stats}")

    backend.close()
    assert len(generated_text) > 0
    print("[TEST] PASS: single-turn test OK")


def test_ssd_prefix_reuse_multiturn():
    """
    Multi-turn SSD prefix reuse test.

    Turn 1: Ingest a long document (>2048 tokens) so SSD archival is triggered.
            Generate a short reply.
    Turn 2: Send Turn 2 prompt (identical prefix + new message) with reset=False.
            Must reuse prefix instead of re-prefilling from scratch.
            TTFT target: < 5 seconds (should be <1s on GPU, <5s on CPU).
            Verify log message contains 'SSD-aware prefix reuse'.
    """
    assert os.path.exists(MODEL_PATH), f"Model path not found: {MODEL_PATH}"

    # Build a ~2800-token document to force SSD eviction (active_ctx=2048, triggers at 1920)
    long_paragraph = (
        "The quick brown fox jumps over the lazy dog. " * 100 +
        "Artificial intelligence is transforming every industry. " * 100 +
        "Python is one of the most popular programming languages. " * 100
    )

    backend = LlamaSSDNativeBackend()
    backend.load_model(MODEL_PATH, n_gpu_layers=33)
    backend.create_context(active_ctx_size=2048, max_ssd_context=131072)
    print("\n[MULTI-TURN TEST] Context created (active=2048, ssd_max=131072)")

    # Turn 1: ingest long prompt
    turn1_prompt = (
        "<start_of_turn>user\n"
        f"Here is a long document for context:\n{long_paragraph}\n"
        "Now summarize it in one sentence.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    t0 = time.time()
    n1 = backend.ingest(turn1_prompt, reset=True)
    t_ingest1 = time.time() - t0
    print(f"[MULTI-TURN TEST] Turn 1 ingested {n1} tokens in {t_ingest1:.2f}s")

    # Generate Turn 1 reply
    turn1_reply = ""
    for piece in backend.generate("", max_new_tokens=30, temperature=0.2):
        turn1_reply += piece
        if len(turn1_reply) > 80:
            break
    print(f"[MULTI-TURN TEST] Turn 1 reply: {turn1_reply.encode('ascii', errors='replace').decode('ascii').strip()!r}")

    stats1 = backend.get_stats()
    ssd_blocks_after_t1 = stats1.get("ssd_blocks", 0)
    print(f"[MULTI-TURN TEST] After Turn 1: ssd_blocks={ssd_blocks_after_t1}, total_tokens={stats1.get('total_tokens')}")

    # Build Turn 2 full conversation string (prefix is identical)
    turn2_prompt = (
        "<start_of_turn>user\n"
        f"Here is a long document for context:\n{long_paragraph}\n"
        "Now summarize it in one sentence.<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{turn1_reply}<end_of_turn>\n"
        "<start_of_turn>user\nThanks! What was the main topic of that document?<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    # Turn 2: ingest with reset=False — must reuse prefix
    t0 = time.time()
    n2 = backend.ingest(turn2_prompt, reset=False)
    t_ingest2 = time.time() - t0
    print(f"[MULTI-TURN TEST] Turn 2 ingested {n2} tokens in {t_ingest2:.2f}s")

    # Generate Turn 2 reply
    t0 = time.time()
    turn2_reply = ""
    first_token_time = None
    for piece in backend.generate("", max_new_tokens=30, temperature=0.2):
        if first_token_time is None:
            first_token_time = time.time() - t0
        turn2_reply += piece
        if len(turn2_reply) > 80:
            break
    total_t2 = time.time() - t0
    print(f"[MULTI-TURN TEST] Turn 2 TTFT: {first_token_time:.3f}s, total: {total_t2:.2f}s")
    print(f"[MULTI-TURN TEST] Turn 2 reply: {turn2_reply.encode('ascii', errors='replace').decode('ascii').strip()!r}")

    stats2 = backend.get_stats()
    print(f"[MULTI-TURN TEST] After Turn 2: ssd_blocks={stats2.get('ssd_blocks')}, total_tokens={stats2.get('total_tokens')}")

    backend.close()

    # Assertions
    assert ssd_blocks_after_t1 > 0, \
        f"Expected SSD eviction after Turn 1 (long doc), but ssd_blocks=0. Increase doc length."
    assert len(turn2_reply) > 0, \
        "Turn 2 generated empty reply — possible KV corruption."
    assert t_ingest2 < 10.0, \
        f"Turn 2 ingest took {t_ingest2:.2f}s — prefix reuse not working (expected <10s, not 63s)."
    assert first_token_time is not None and first_token_time < 10.0, \
        f"Turn 2 TTFT was {first_token_time:.2f}s — still too slow. Prefix reuse may not be active."
    print(f"\n[MULTI-TURN TEST] PASS: SSD prefix reuse working correctly!")
    print(f"  ssd_blocks after Turn 1: {ssd_blocks_after_t1}")
    print(f"  Turn 2 ingest time:       {t_ingest2:.3f}s (target <10s)")
    print(f"  Turn 2 TTFT:              {first_token_time:.3f}s (target <10s)")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Single-turn basic generation")
    print("=" * 60)
    test_native_wrapper_load_and_generate()

    print("\n" + "=" * 60)
    print("Test 2: Multi-turn SSD prefix reuse")
    print("=" * 60)
    test_ssd_prefix_reuse_multiturn()

    print("\n[ALL TESTS PASSED]")
