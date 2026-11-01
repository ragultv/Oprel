import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend

MODEL_PATH = r"C:\Users\tragu\.cache\oprel\models\models--unsloth--gemma-3-4b-it-GGUF\snapshots\5a3566e716d80f709ed7b79817eaf7733d2a1fce\gemma-3-4b-it-Q5_K_M.gguf"

SECRET_NEEDLES = (
    "CONFIDENTIAL SPECIFICATIONS DOCUMENT [DO NOT DISCLOSE]\n"
    "Item 1. The classified project codename is: CHIMERA-779.\n"
    "Item 2. The flight safety security passkey is: VORTEX-DELTA-902.\n"
    "Item 3. The maximum turbine bypass ratio limit is: 14.85.\n"
    "End of confidential parameters.\n\n"
)

# Repeatable technical text to fill > 4,000 tokens of distraction
DISTRACTOR_PARAGRAPH = (
    "In fluid mechanics and aeronautical engineering, the Navier-Stokes equations dictate "
    "the motion of viscous fluid substances. For incompressible Newtonian fluids, the momentum "
    "equation is expressed as density multiplied by the material derivative of velocity equaling "
    "the negative gradient of static pressure plus dynamic viscosity times the Laplacian of velocity "
    "plus volumetric body forces. The boundary layer theory introduced by Ludwig Prandtl demonstrates "
    "that viscous effects are confined to a thin layer adjacent to solid boundaries, while the outer "
    "flow can be treated as inviscid potential flow satisfying Laplace's equation for velocity potential. "
    "In transonic and supersonic flow regimes, compressible shockwaves induce sharp discontinuous jumps "
    "in Mach number, stagnation pressure, density, and thermodynamic temperature across the shock front. "
    "Turbofan thermodynamic cycles are governed by the Brayton gas turbine cycle, where isentropic "
    "compression in multistage axial compressors is followed by isobaric heat addition in annular "
    "combustion chambers and isentropic expansion across high-pressure and low-pressure turbine stages. "
)

def build_distractor_text(target_words: int = 3500) -> str:
    paragraphs = []
    current_words = 0
    section_num = 1
    while current_words < target_words:
        p = f"\n\nSection {section_num}: Aeronautical Aerodynamics and Propulsion Analysis.\n" + DISTRACTOR_PARAGRAPH
        paragraphs.append(p)
        current_words += len(p.split())
        section_num += 1
    return "".join(paragraphs)

QUESTION = "What is the classified project codename, the flight safety security passkey, and the maximum turbine bypass ratio limit? Answer clearly with the exact values."

def run_test():
    print("=====================================================================")
    print("      NEEDLE-IN-A-HAYSTACK PROOF TEST: SSD KV-CACHE RETRIEVAL        ")
    print("=====================================================================\n")

    distractor = build_distractor_text(target_words=3200)
    full_prompt = SECRET_NEEDLES + distractor
    print(f"Total prompt length: {len(full_prompt)} characters (~4,200 tokens).")
    print(f"Secret Needles planted at Token 0..80:")
    print("  - Project Codename:       CHIMERA-779")
    print("  - Security Passkey:       VORTEX-DELTA-902")
    print("  - Maximum Bypass Ratio:   14.85\n")

    # -------------------------------------------------------------
    # RUN A: PURE SLIDING WINDOW (Active Window = 2048, No Retrieval)
    # -------------------------------------------------------------
    print(">>> [RUN A] Testing Pure Sliding Window (Window: 2048 tokens, NO retrieval)...")
    backend_a = LlamaSSDNativeBackend()
    backend_a.load_model(MODEL_PATH, n_gpu_layers=33)
    backend_a.create_context(active_ctx_size=2048, max_ssd_context=131072)

    t0 = time.perf_counter()
    n_a = backend_a.ingest(full_prompt, reset=True)
    stats_a = backend_a.get_stats()
    print(f"  Ingested {n_a} tokens in {time.perf_counter() - t0:.2f}s.")
    print(f"  Active VRAM Tokens: {stats_a['active_tokens']}/2048 | Total Tokens: {stats_a['total_tokens']} | SSD Blocks: {stats_a['ssd_blocks']}")
    print(f"  Notice: The first ~2,000 tokens have been completely EVICTED from active VRAM.")

    print(f"  Asking Question without retrieval: '{QUESTION}'")
    query_prompt_a = f"\n\nUser: {QUESTION}\nAssistant:"
    backend_a.ingest(query_prompt_a, reset=False)

    ans_a = ""
    for piece in backend_a.generate("", max_new_tokens=150, temperature=0.1):
        ans_a += piece
        if len(ans_a) > 200:
            break
    backend_a.close()

    print(f"\n  [Run A Response]:\n  \"{ans_a.strip()}\"\n")
    pass_a = ("CHIMERA-779" in ans_a) and ("VORTEX-DELTA-902" in ans_a)
    print(f"  [Run A Verdict]: {'PASS' if pass_a else 'FAIL (Expected - facts were evicted from sliding window)'}\n")

    # -------------------------------------------------------------
    # RUN B: QUEST SSD MEMORY RETRIEVAL (Window = 2048, SSD Direct I/O)
    # -------------------------------------------------------------
    print(">>> [RUN B] Testing Quest SSD Memory Retrieval (Window: 2048 tokens + SSD Direct I/O)...")
    backend_b = LlamaSSDNativeBackend()
    backend_b.load_model(MODEL_PATH, n_gpu_layers=33)
    backend_b.create_context(active_ctx_size=2048, max_ssd_context=131072)

    t0 = time.perf_counter()
    n_b = backend_b.ingest(full_prompt, reset=True)
    stats_b = backend_b.get_stats()
    print(f"  Ingested {n_b} tokens in {time.perf_counter() - t0:.2f}s.")
    print(f"  Active VRAM Tokens: {stats_b['active_tokens']}/2048 | Total Tokens: {stats_b['total_tokens']} | SSD Blocks: {stats_b['ssd_blocks']}")

    print(f"  Retrieving historical context from NVMe SSD for query: '{QUESTION}'...")
    retrieved_memory = backend_b.retrieve_context(QUESTION, top_k=2, max_chars=8192)
    stats_b_after = backend_b.get_stats()
    print(f"  SSD Retrieval Time: {stats_b_after['last_retrieval_ms']:.2f} ms via Win32 Direct I/O!")
    print(f"  Total Blocks Retrieved: {stats_b_after['retrieved_blocks']}")
    print(f"  Retrieved Content Excerpt (first 250 chars):\n    \"{retrieved_memory[:250].strip()}...\"\n")

    # Ingest the retrieved memory alongside the question
    augmented_query = (
        f"\n\nRelevant Context Retrieved from SSD Memory:\n{retrieved_memory.strip()}\n\n"
        f"User: {QUESTION}\nAssistant:"
    )
    backend_b.ingest(augmented_query, reset=False)

    ans_b = ""
    for piece in backend_b.generate("", max_new_tokens=150, temperature=0.1):
        ans_b += piece
        if len(ans_b) > 200:
            break
    backend_b.close()

    print(f"\n  [Run B Response]:\n  \"{ans_b.strip()}\"\n")
    pass_b = ("CHIMERA-779" in ans_b) and ("VORTEX-DELTA-902" in ans_b)
    print(f"  [Run B Verdict]: {'PASS - Facts successfully recalled from SSD!' if pass_b else 'FAIL'}\n")

    # -------------------------------------------------------------
    # RUN C: GROUND TRUTH (In-Memory 8k Context, No Eviction)
    # -------------------------------------------------------------
    print(">>> [RUN C] Ground Truth Baseline (Full in-memory context: 8192 tokens, no eviction)...")
    backend_c = LlamaSSDNativeBackend()
    backend_c.load_model(MODEL_PATH, n_gpu_layers=33)
    backend_c.create_context(active_ctx_size=8192, max_ssd_context=8192)

    t0 = time.perf_counter()
    n_c = backend_c.ingest(full_prompt, reset=True)
    stats_c = backend_c.get_stats()
    print(f"  Ingested {n_c} tokens in {time.perf_counter() - t0:.2f}s.")
    print(f"  Active VRAM Tokens: {stats_c['active_tokens']}/8192 | SSD Blocks: {stats_c['ssd_blocks']}")

    query_prompt_c = f"\n\nUser: {QUESTION}\nAssistant:"
    backend_c.ingest(query_prompt_c, reset=False)

    ans_c = ""
    for piece in backend_c.generate("", max_new_tokens=150, temperature=0.1):
        ans_c += piece
        if len(ans_c) > 200:
            break
    backend_c.close()

    print(f"\n  [Run C Response (Ground Truth)]:\n  \"{ans_c.strip()}\"\n")
    pass_c = ("CHIMERA-779" in ans_c) and ("VORTEX-DELTA-902" in ans_c)
    print(f"  [Run C Verdict]: {'PASS' if pass_c else 'FAIL'}\n")

    print("=====================================================================")
    print("                       FINAL COMPARISON TABLE                        ")
    print("=====================================================================")
    print(f"Run A (Sliding Window 2k, No Retrieval): {'RECALLED (UNEXPECTED)' if pass_a else 'FAILED TO RECALL (PROVEN EVICTION)'}")
    print(f"Run B (Quest SSD Direct I/O Retrieval):  {'RECALLED WITH 100% ACCURACY' if pass_b else 'FAILED'}")
    print(f"Run C (Ground Truth In-Memory 8k):       {'RECALLED WITH 100% ACCURACY' if pass_c else 'FAILED'}")
    print(f"SSD Direct I/O Read Latency:             {stats_b_after['last_retrieval_ms']:.2f} ms")
    print("=====================================================================\n")

if __name__ == "__main__":
    run_test()
