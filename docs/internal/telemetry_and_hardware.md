# Telemetry and Hardware Awareness

The hallmark of Oprel is its ability to run massive AI models gracefully on severely constrained consumer hardware. This is achieved through the `oprel/telemetry/` package.

## 1. Hardware Profiling (`hardware.py`)
On startup, Oprel conducts a deep scan of the host machine:
- **System RAM**: Detects total available memory via `psutil`.
- **GPU Profiling**: Detects CUDA devices, Apple Silicon Unified Memory architectures, or AMD ROCm availability. It accurately probes dedicated VRAM vs shared memory.
- **CPU Architecture**: Parses CPU flags to determine AVX, AVX2, or NEON support, heavily influencing how `llama.cpp` should be compiled or run.

## 2. VRAM Monitoring (`vram_monitor.py`)
Before spawning any new process, the VRAM monitor polls the GPU.
- If a user tries to load a 70B model that requires 40GB of VRAM onto a 12GB RTX 3060, Oprel calculates the exact number of transformer layers (`-ngl`) that will safely fit inside the 12GB envelope.
- The remaining layers are automatically pinned to the system RAM.
- **Memory Pressure (`memory_pressure.py`)**: If multiple models are loaded (e.g., an LLM, an Embedder, and an OCR engine) and VRAM hits critical capacity (e.g., > 95%), Oprel triggers an automatic eviction protocol. It kills the longest-idle model's subprocess to free up space for the incoming request, completely abstracting hardware constraints from the end user.

## 3. Recommender Engine (`recommender.py`)
When a user attempts to download a generic model alias (e.g., `oprel pull llama3`), the recommender calculates the optimal quantization.
- A user with 8GB RAM will automatically be given a highly compressed `Q4_K_M` binary.
- A user with 64GB RAM will receive a pristine `Q8_0` binary.
