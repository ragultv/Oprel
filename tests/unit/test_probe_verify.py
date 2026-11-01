import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import ctypes
import time
from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend

MODEL_PATH = r"C:\Users\tragu\.cache\oprel\models\models--unsloth--gemma-3-4b-it-GGUF\snapshots\5a3566e716d80f709ed7b79817eaf7733d2a1fce\gemma-3-4b-it-Q5_K_M.gguf"

def probe():
    print("=== PROBING LLAMA.CPP KV CACHE BEHAVIOR ===")
    backend = LlamaSSDNativeBackend()
    backend.load_model(MODEL_PATH, n_gpu_layers=35)
    backend.create_context(active_ctx_size=2048, max_ssd_context=131072)

    # Ingest 1000 tokens
    prompt = "The quick brown fox jumps over the lazy dog. " * 80
    n = backend.ingest(prompt, reset=True)
    print(f"Ingested {n} tokens into sequence 0.")

    # Access internal context pointer
    # Let's inspect ctypes calls directly on backend.dll
    # We can get context pointer from backend.context_handle
    # In llama_ssd_wrapper.cpp, context_handle is OprelContext*
    # Inside OprelContext:
    # struct OprelContext {
    #     OprelModel* model_ref;
    #     llama_context* ctx;
    #     ...
    # }
    # Let's read ctx pointer at offset 8 (on 64-bit Windows, pointer size is 8 bytes)
    ctx_ptr = ctypes.c_void_p.from_address(backend.context_handle + 8).value
    print(f"Native llama_context pointer: 0x{ctx_ptr:x}")

    # Load llama.dll functions directly
    llama_dll_path = r"C:\Users\tragu\.cache\oprel\bin\llama_cpp\cuda\llama.dll"
    llama_dll = ctypes.CDLL(llama_dll_path)

    llama_get_memory = llama_dll.llama_get_memory
    llama_get_memory.argtypes = [ctypes.c_void_p]
    llama_get_memory.restype = ctypes.c_void_p

    llama_memory_seq_cp = llama_dll.llama_memory_seq_cp
    llama_memory_seq_cp.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    llama_memory_seq_cp.restype = None

    llama_memory_seq_rm = llama_dll.llama_memory_seq_rm
    llama_memory_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    llama_memory_seq_rm.restype = ctypes.c_bool

    llama_memory_seq_pos_min = llama_dll.llama_memory_seq_pos_min
    llama_memory_seq_pos_min.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    llama_memory_seq_pos_min.restype = ctypes.c_int32

    llama_memory_seq_pos_max = llama_dll.llama_memory_seq_pos_max
    llama_memory_seq_pos_max.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    llama_memory_seq_pos_max.restype = ctypes.c_int32

    llama_state_seq_get_size_ext = llama_dll.llama_state_seq_get_size_ext
    llama_state_seq_get_size_ext.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32]
    llama_state_seq_get_size_ext.restype = ctypes.c_size_t

    llama_state_seq_get_data_ext = llama_dll.llama_state_seq_get_data_ext
    llama_state_seq_get_data_ext.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_uint32]
    llama_state_seq_get_data_ext.restype = ctypes.c_size_t

    llama_state_seq_set_data_ext = llama_dll.llama_state_seq_set_data_ext
    llama_state_seq_set_data_ext.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_uint32]
    llama_state_seq_set_data_ext.restype = ctypes.c_size_t

    mem = llama_get_memory(ctx_ptr)
    print(f"llama_memory_t: 0x{mem:x}")

    min_pos_0 = llama_memory_seq_pos_min(mem, 0)
    max_pos_0 = llama_memory_seq_pos_max(mem, 0)
    print(f"Sequence 0: min_pos = {min_pos_0}, max_pos = {max_pos_0}")

    sz_seq0_full = llama_state_seq_get_size_ext(ctx_ptr, 0, 0)
    print(f"Sequence 0 full state size: {sz_seq0_full} bytes ({sz_seq0_full / (1024*1024):.2f} MiB)")

    # Test Check (b): Partial range seq_cp from seq 0 to scratch seq 1
    p0 = 100
    p1 = 612  # exactly 512 tokens
    print(f"\n--- Testing Check (b): seq_cp(0 -> 1, p0={p0}, p1={p1}) ---")
    llama_memory_seq_cp(mem, 0, 1, p0, p1)
    min_pos_1 = llama_memory_seq_pos_min(mem, 1)
    max_pos_1 = llama_memory_seq_pos_max(mem, 1)
    print(f"Scratch Sequence 1 after seq_cp: min_pos = {min_pos_1}, max_pos = {max_pos_1}")

    # Test Check (c): Scratch seq 1 state blob size
    print(f"\n--- Testing Check (c): Blob size of 512-token scratch copy ---")
    sz_seq1 = llama_state_seq_get_size_ext(ctx_ptr, 1, 0)
    print(f"Scratch Sequence 1 (512 tokens) state size: {sz_seq1} bytes ({sz_seq1 / (1024*1024):.2f} MiB)")

    # Also test with flags = 1 (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
    sz_seq1_partial = llama_state_seq_get_size_ext(ctx_ptr, 1, 1)
    print(f"Scratch Sequence 1 with FLAGS_PARTIAL_ONLY (flags=1): {sz_seq1_partial} bytes ({sz_seq1_partial / (1024*1024):.2f} MiB)")

    # Test Check (a): Does state_seq_set_data_ext into seq 0 remove other cells in seq 0?
    print(f"\n--- Testing Check (a): state_seq_set_data_ext cell removal behavior ---")
    # Serialize seq 1 into a buffer
    buf1 = ctypes.create_string_buffer(sz_seq1)
    llama_state_seq_get_data_ext(ctx_ptr, buf1, sz_seq1, 1, 0)

    # Clean up seq 1
    llama_memory_seq_rm(mem, 1, -1, -1)
    print(f"Cleaned up seq 1: min_pos = {llama_memory_seq_pos_min(mem, 1)}, max_pos = {llama_memory_seq_pos_max(mem, 1)}")

    # Before restoring into seq 0, check seq 0 pos range:
    print(f"Seq 0 before restore: min_pos = {llama_memory_seq_pos_min(mem, 0)}, max_pos = {llama_memory_seq_pos_max(mem, 0)}")

    # Now restore the 512-token buffer directly into seq 0:
    llama_state_seq_set_data_ext(ctx_ptr, buf1, sz_seq1, 0, 0)
    print(f"Seq 0 AFTER direct restore: min_pos = {llama_memory_seq_pos_min(mem, 0)}, max_pos = {llama_memory_seq_pos_max(mem, 0)}")
    cells_purged = (llama_memory_seq_pos_max(mem, 0) < max_pos_0)
    print(f"Did direct restore into seq 0 purge the remaining cells of seq 0? {cells_purged}")

    # Now test restoring into seq 1, then seq_cp(1 -> 0, p0, p1), then seq_rm(1):
    print(f"\n--- Testing Safe Scratch Flow: restore into seq 1 -> seq_cp(1 -> 0) -> seq_rm(1) ---")
    # First re-ingest to restore clean state
    backend.ingest(prompt, reset=True)
    orig_max_0 = llama_memory_seq_pos_max(mem, 0)
    print(f"Clean Seq 0: min_pos = {llama_memory_seq_pos_min(mem, 0)}, max_pos = {orig_max_0}")

    # Evict p0..p1 from seq 0
    llama_memory_seq_rm(mem, 0, p0, p1)
    print(f"Evicted [{p0}, {p1}) from seq 0.")

    # Restore buffer into scratch seq 1:
    llama_state_seq_set_data_ext(ctx_ptr, buf1, sz_seq1, 1, 0)
    print(f"Restored into scratch seq 1: min_pos = {llama_memory_seq_pos_min(mem, 1)}, max_pos = {llama_memory_seq_pos_max(mem, 1)}")

    # Copy from scratch seq 1 into seq 0:
    llama_memory_seq_cp(mem, 1, 0, p0, p1)
    print(f"Copied [{p0}, {p1}) from scratch seq 1 into seq 0.")

    # Remove scratch seq 1:
    llama_memory_seq_rm(mem, 1, -1, -1)
    print(f"Seq 0 after safe scratch restore: min_pos = {llama_memory_seq_pos_min(mem, 0)}, max_pos = {llama_memory_seq_pos_max(mem, 0)}")
    print(f"Did live end stay intact at {orig_max_0}? {llama_memory_seq_pos_max(mem, 0) == orig_max_0}")

    backend.close()
    print("\n=== PROBE COMPLETE ===")

if __name__ == "__main__":
    probe()
