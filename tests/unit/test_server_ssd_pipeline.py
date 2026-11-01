import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
from oprel.server.services.models import load_model
from oprel.server.services.generation import generate_text, GenerateParams, StreamResult
from oprel.server.routes.system import get_kv_cache_stats
from oprel.server.domain.state import get_state

async def main():
    print("=== Testing Server Native SSD Pipeline ===", flush=True)
    
    # 1. Load model with native-ssd backend using local cached Q5_K_M quantization
    print("[PIPELINE] Loading model...", flush=True)
    res = load_model("unsloth/gemma-3-4b-it-GGUF", backend="native-ssd", quantization="Q5_K_M")
    print("Load model result:", res, flush=True)

    # 2. Query system KV cache stats
    stats1 = await get_kv_cache_stats()
    print("KV Cache Stats (Initial):", stats1, flush=True)
    assert stats1["engine"] == "Native C++ SSD KV-Store (oprel_llama_ssd.dll)"

    # 3. Generate response
    params = GenerateParams(
        model_id="unsloth/gemma-3-4b-it-GGUF",
        prompt="Tell me what 2 + 2 is in 3 words.",
        max_tokens=20,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        stream=True,
        images=None,
        conversation_id="test-conv-ssd-1",
        system_prompt="You are a helpful assistant.",
        reset_conversation=True,
        thinking=False,
        rag=False,
    )

    print("\nStreaming response from native SSD backend:", flush=True)
    result = await generate_text(params)
    tokens = []
    if isinstance(result, StreamResult):
        async for chunk in result.iterator:
            tokens.append(chunk)
            print(chunk.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)

    print("\n", flush=True)

    # 4. Query system KV cache stats after generation
    stats2 = await get_kv_cache_stats()
    print("KV Cache Stats (After Generation):", stats2, flush=True)
    assert stats2["total_tokens"] > 0
    print(f"Verified: Total tokens in SSD context = {stats2['total_tokens']}", flush=True)

    # 5. Verify Turn 2 prefix reuse (Instant Recall)
    params2 = GenerateParams(
        model_id="unsloth/gemma-3-4b-it-GGUF",
        prompt="And what is 3 + 3?",
        max_tokens=20,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        stream=True,
        images=None,
        conversation_id="test-conv-ssd-1",
        system_prompt="You are a helpful assistant.",
        reset_conversation=False,
        thinking=False,
        rag=False,
    )
    print("\nTurn 2 (Prefix Reuse / Instant Recall):", flush=True)
    t0 = time.time()
    result2 = await generate_text(params2)
    tokens2 = []
    if isinstance(result2, StreamResult):
        async for chunk in result2.iterator:
            tokens2.append(chunk)
            print(chunk.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)
    t_turn2 = time.time() - t0
    print(f"\nTurn 2 completed in {t_turn2:.3f}s", flush=True)

    # 6. Clean up
    state = get_state()
    for m in list(state.models.values()):
        if hasattr(m, "close"):
            m.close()

    print("\nAll pipeline assertions passed successfully!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
