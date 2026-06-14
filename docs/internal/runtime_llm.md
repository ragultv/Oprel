# LLM Runtime Internals

Oprel wraps and orchestrates the `llama.cpp` inference engine (`oprel/runtime/process.py`) to execute Large Language Models efficiently.

## 1. Process Management
Unlike simple wrappers that compile Python bindings for `llama.cpp` (which can cause GIL locking and threading issues), Oprel spawns the `llama-server` binary as an entirely independent **subprocess**.

1. **Port Allocation**: Oprel finds an open port dynamically.
2. **Subprocess Spawning**: `process.py` executes the binary with highly tuned arguments (e.g., `-m <path> -c 8192 -ngl 999`).
3. **IPC via HTTP**: The Python daemon communicates with the C++ subprocess over a localized, high-speed HTTP connection.
4. **Lifecycle Hooks**: If the subprocess crashes (e.g., due to an out-of-memory error), Oprel's `model_state.py` catches the connection refusal, clears the state, and automatically attempts to respawn the process on the next user request.

## 2. Hardware Optimization

### VRAM Offloading (`offloading.py`)
Oprel parses the GGUF metadata to determine the exact number of transformer layers in a model. It then queries `vram_monitor.py` to check available GPU memory. It calculates precisely how many layers (`-ngl`) can fit into VRAM, leaving a safe 500MB buffer for the KV cache. The remaining layers are automatically pinned to system RAM.

### CPU Threading (`cpu_optimizer.py`)
If a model spills over into system RAM, CPU inference speed becomes the bottleneck. `cpu_optimizer.py` profiles the host CPU architecture (e.g., Apple Silicon vs Intel vs AMD) and determines the optimal number of physical threads to use, avoiding hyperthreading penalties.

### KV Cache Management (`kv_cache.py`)
The Key-Value (KV) cache stores the pre-computed mathematical states of previous tokens in a conversation. 
- Oprel instructs `llama.cpp` to allocate a fixed-size context window (default: 8192).
- When multiple models are loaded, or when switching contexts rapidly, Oprel can serialize the KV cache matrix to the hard drive and restore it instantly on the next turn, preventing expensive prompt re-evaluations.
