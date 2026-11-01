import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend

def test_longform():
    model_path = r"C:\Users\tragu\.cache\oprel\models\models--unsloth--gemma-3-4b-it-GGUF\snapshots\5a3566e716d80f709ed7b79817eaf7733d2a1fce\gemma-3-4b-it-Q5_K_M.gguf"
    print("=== Testing 10k Token Continuous Generation & SSD KV-Cache ===")
    
    backend = LlamaSSDNativeBackend()
    backend.load_model(model_path, n_gpu_layers=33)
    
    # Option B: Active working window of 2,048 tokens in VRAM, SSD context up to 131,072
    backend.create_context(active_ctx_size=2048, max_ssd_context=131072)
    
    prompt = (
        "Write an exhaustive, encyclopedic manual of modern aeronautical engineering, "
        "atmospheric fluid dynamics, and rocket propulsion systems. "
        "Cover Navier-Stokes equations, boundary layer theory, supersonic shockwaves, "
        "turbofan thermodynamic cycles (Brayton cycle), and fly-by-wire avionics architecture. "
        "Provide thorough equations, derivations, and engineering explanations in full depth."
    )
    print(f"Ingesting prompt ({len(prompt)} chars)...")
    t0_prefill = time.perf_counter()
    n = backend.ingest(prompt, reset=True)
    t_prefill = time.perf_counter() - t0_prefill
    print(f"Ingested {n} prompt tokens in {t_prefill*1000:.1f}ms. Prefill speed: {n/t_prefill:.1f} t/s")
    
    target_tokens = 10000
    print(f"\n--- Starting 10,000-Token Continuous Generation Benchmark ---")
    start = time.perf_counter()
    tokens_generated = 0
    checkpoint_time = start
    checkpoint_tokens = 0
    ttft_measured = False
    
    for piece in backend.generate(prompt, max_new_tokens=target_tokens, temperature=0.7, ignore_eos=True):
        if not ttft_measured:
            ttft = (time.perf_counter() - start) * 1000
            print(f"[TTFT] Time To First Token: {ttft:.1f}ms\n", flush=True)
            ttft_measured = True
            
        tokens_generated += 1
        checkpoint_tokens += 1
        
        if tokens_generated % 250 == 0:
            now = time.perf_counter()
            step_speed = checkpoint_tokens / (now - checkpoint_time)
            avg_speed = tokens_generated / (now - start)
            st = backend.get_stats()
            print(
                f"\n[Token {tokens_generated:5d}/{target_tokens}] "
                f"Instant Speed: {step_speed:4.1f} t/s | Avg Speed: {avg_speed:4.1f} t/s | "
                f"Active VRAM Tokens: {st['active_tokens']:4d}/2048 | "
                f"Total Context: {st['total_tokens']:5d} | "
                f"SSD Blocks Archived: {st['ssd_blocks']:2d}",
                flush=True
            )
            checkpoint_time = now
            checkpoint_tokens = 0
        elif tokens_generated <= 150:
            print(piece, end="", flush=True)
    
    elapsed = time.perf_counter() - start
    print(f"\n\n================ Benchmark Summary ================")
    print(f"Total Tokens Generated:   {tokens_generated}")
    print(f"Total Time:               {elapsed:.2f}s ({elapsed / 60:.2f} mins)")
    print(f"Average Generation Speed: {tokens_generated / elapsed:.2f} tokens/s")
    stats = backend.get_stats()
    print(f"Final Context Telemetry:  {stats}")
    print(f"SSD Blocks Created:       {stats['ssd_blocks']}")
    print(f"====================================================\n")
    
    backend.close()

if __name__ == "__main__":
    test_longform()
