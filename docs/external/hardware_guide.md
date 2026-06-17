# Hardware & Deployment Guide

How Oprel uses your hardware — and how to get the best performance.

Oprel runs large language models and image generation on your own machine by wrapping two battle-tested inference engines: [llama.cpp](https://github.com/ggml-org/llama.cpp) for text generation and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) for image generation. Oprel automatically downloads the right pre-built binary for your platform, detects your GPU, calculates optimal settings, and manages the subprocess lifecycle. You don't need to configure backends or compile anything from source.

---

## Supported Platforms

| Platform | Text (llama.cpp) | Image (stable-diffusion.cpp) |
|---|---|---|
| **Linux x86_64** | CPU, Vulkan | CPU, Vulkan, ROCm |
| **Windows x86_64** | CPU, CUDA, Vulkan | CPU, CUDA, Vulkan |
| **Windows arm64** | CPU | — |
| **macOS arm64** (Apple Silicon) | Metal | Metal |
| **macOS x86_64** (Intel) | CPU | — |

---

## How Oprel Detects Your Hardware

On startup, Oprel scans your machine in this order:

1. **PyTorch CUDA** — if `torch` is installed and `torch.cuda.is_available()`, uses it directly.
2. **nvidia-smi** — falls back to the NVIDIA driver utility (works without PyTorch).
3. **rocm-smi** — checks for AMD ROCm on Linux.
4. **lspci fallback** — detects AMD GPUs even without ROCm installed (VRAM defaults conservatively).
5. **Metal** — on Apple Silicon, probes `system_profiler` for GPU info and calculates usable unified memory.
6. **CPU** — if no GPU is found, runs on CPU with automatic thread detection.

The binary that gets downloaded is chosen to match: if you have an NVIDIA GPU on Windows, you get the CUDA binary. If you're on Linux with an AMD GPU but no ROCm installed, you get the CPU binary (or Vulkan, if your GPU supports it).

---

## GPU Acceleration: What Each Backend Gives You

### CUDA (NVIDIA)

**llama.cpp:** CUDA binaries are available on **Windows only**. The binary registry does not include a Linux CUDA variant for llama.cpp. On Linux with an NVIDIA GPU, Oprel falls back to CPU or Vulkan.

**stable-diffusion.cpp:** CUDA binaries are available on **Windows**.

### Vulkan

Vulkan support is available for both backends on **Linux x86_64** and **Windows x86_64**. Vulkan provides GPU acceleration via the Vulkan API and is typically faster than pure CPU. If you have an NVIDIA GPU on Linux, Vulkan is your best available option for text generation.

### Metal (Apple Silicon)

Metal binaries are available for both backends on **macOS arm64** (M1/M2/M3/M4). Because Apple Silicon uses unified memory, the GPU and CPU share the same pool. Oprel estimates how much of your total RAM can safely be used for model inference — typically 50–75% depending on your chip.

### ROCm (AMD)

ROCm binaries exist only for **stable-diffusion.cpp on Linux**. There is no ROCm variant for llama.cpp in the binary registry. On Linux with an AMD GPU, text generation runs on CPU or Vulkan; image generation can use ROCm if `rocm-smi` is detected.

---

## RAM and VRAM: The Practical Limits

Oprel does not implement its own inference engine, memory manager, or GPU scheduler. It passes models to llama.cpp and stable-diffusion.cpp with the best settings it can calculate from your hardware. The real memory limits are set by those engines.

### System RAM (all platforms)

Model weights live in RAM. Even with GPU offloading, the full model file must fit in system memory (or be memory-mapped). As a rule of thumb, you need at least the GGUF file size plus 1–2 GB for context and overhead.

### VRAM (dedicated GPUs)

Only the layers you offload to GPU (`n_gpu_layers`) consume VRAM. The remaining layers stay in system RAM and run on CPU. Oprel automatically calculates how many layers fit:

- It reads total VRAM from `nvidia-smi`.
- It estimates total model layers from file size (e.g., ~32 layers for a 7B-class model, ~80 for a 70B-class).
- It reserves headroom for KV cache and CUDA overhead (typically 0.5–1 GB depending on GPU type).
- It outputs a recommended `n_gpu_layers` value.

You can override this manually with `--n-gpu-layers <N>` if you want to force more or fewer layers onto GPU.

### Unified Memory (Apple Silicon)

On Apple Silicon, there is no separate VRAM pool. llama.cpp uses Metal for GPU acceleration and draws from unified memory. Oprel's heuristics estimate usable memory based on total RAM (50% for 8 GB base models, up to 75% for 64 GB Max chips). This is conservative to leave room for macOS and other applications.

---

## GGUF Model Size Expectations

Oprel downloads models from the HuggingFace Hub in GGUF format. File size depends on the original model parameters and the quantization level.

| Model class | Q4_K_M (typical) | Q8_0 (high quality) |
|---|---|---|
| 1–3B params | 1–2 GB | 2–4 GB |
| 7–8B params | 4–5 GB | 7–9 GB |
| 13–14B params | 7–9 GB | 13–16 GB |
| 33–34B params | 18–22 GB | 33–38 GB |
| 70–72B params | 38–44 GB | 70–77 GB |

These are approximate. Exact sizes vary by model architecture and vocabulary.

---

## What Oprel Controls vs What llama.cpp Controls

**Oprel handles:**
- Hardware detection (GPU type, VRAM, RAM, CPU features)
- Binary selection and download (platform + GPU variant)
- `n_gpu_layers` calculation (how many layers to offload)
- Thread count recommendation (physical cores minus 1–2 for system)
- KV cache type selection (`f16`, `q8_0`, `q4_0`)
- Server lifecycle (spawn, health-check, restart on crash)
- VRAM monitoring during model loading

**llama.cpp handles:**
- Actual inference (token generation, attention, matrix math)
- Memory allocation for model weights
- GPU kernel execution (CUDA, Vulkan, Metal)
- Context window management
- Flash attention and mmap

---

## Cache Locations

Oprel stores content under `~/.cache/oprel/`:

| Directory | Contents |
|---|---|
| `~/.cache/oprel/models/` | Downloaded GGUF model files |
| `~/.cache/oprel/bin/` | llama-server and sd-cli binaries |
| `~/.cache/oprel/ocr/` | PaddleOCR models (downloaded on first OCR use) |
| `~/.cache/oprel/models/chat_history.db` | SQLite database for conversations, provider configs, and settings |

The `OPREL_HOME` environment variable controls the **knowledge directory only** (`OPREL_HOME / "knowledge"`). It does not affect the model cache, binary cache, OCR cache, or database paths — those are configured independently via `Config.cache_dir` and `Config.binary_dir` in `oprel/core/config.py`.

---

## Server and Network

By default, the Oprel server binds to `localhost` and picks a port from the range **54321–54420**. This means:
- Only local applications can reach the server.
- Remote access is blocked by default.

**Do not expose the server to a network** without an authentication layer, reverse proxy, and firewall. The Oprel API has no built-in auth. If you need remote access, put it behind a reverse proxy (nginx, Caddy) with TLS and authentication.

---

## Hardware Tiers: What You Can Run

These are guidelines based on how much memory a model needs. Actual performance depends on your specific hardware, quantization, context length, and backend.

### CPU-only / Low RAM (≤8 GB)

- **Good fit:** 1–3B models (Q4_K_M). Expect slower generation than GPU-backed systems, with speed depending heavily on CPU, quantization, context length, and background load.
- **Possible with patience:** 7B models at Q4_K_M, but system may swap.
- **Not practical:** Models above 7B.

### 8 GB VRAM (e.g., RTX 3070, RTX 4060 Ti)

- **Comfortable:** 7–8B models with most layers on GPU. Responsive chat speeds.
- **Partial offload:** 13B models — some layers on GPU, rest on CPU.
- **Not enough for:** 33B+ models entirely on GPU.

### 12 GB VRAM (e.g., RTX 3080/4070, RTX 3060 12GB)

- **Comfortable:** 7–13B models fully on GPU.
- **Partial offload:** 33B models with most layers on GPU.
- **Stretched:** 70B models at Q4_K_M — fits if you accept CPU fallback for some layers.

### 16 GB VRAM (e.g., RTX 4080, RX 6800/6900 XT)

- **Comfortable:** 7–13B models, plus room for large context windows.
- **Mostly GPU:** 33B models with high GPU layer count.
- **Partial offload:** 70B models at Q4_K_M.

### 24 GB VRAM (e.g., RTX 4090, RTX 3090)

- **Comfortable:** 7–34B models fully on GPU with generous context.
- **Mostly GPU:** 70B models at Q4_K_M fit with most layers on GPU.
- **Watch context:** Very long contexts (>32K) consume significant VRAM for the KV cache.

### Apple Silicon (unified memory)

- **8 GB (base M1/M2):** 1–3B models comfortably. 7B models at Q4_K_M with reduced context.
- **16 GB (M1/M2 with upgrade, M1 Pro base):** 7B models comfortably. 13B at Q4_K_M.
- **32–36 GB (M1/M2 Pro with upgrade):** 13–34B models.
- **64–96 GB (M1/M2 Max):** 34–70B models.

On Apple Silicon, KV cache quantization (`q8_0` or `q4_0`) can significantly reduce memory pressure on larger models.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `OPREL_HOME` | Override knowledge directory base path (default: `~/.cache/oprel`). Does **not** control model, binary, OCR, or database paths |
| `OPREL_SSL_NO_VERIFY` | Set to `1` to disable SSL verification for binary downloads (useful behind corporate proxies) |
| `OPREL_SKIP_RUNTIME_DOWNLOAD` | Set to `1`, `true`, or `yes` to skip runtime binary downloads during `pip install` |

---

## Troubleshooting

### "No GPU detected" but I have one

- **NVIDIA:** Run `nvidia-smi` in a terminal. If it fails, your drivers aren't installed or aren't in PATH. On Linux, also check that `nvidia-smi` is available.
- **AMD on Linux:** Install ROCm if you want ROCm-backed image generation. Without `rocm-smi`, Oprel may still detect your GPU via `lspci`, but backend availability depends on the binary selected for your platform.
- **Apple Silicon:** Make sure you're on macOS arm64 (not Intel). Run `uname -m` — it should say `arm64`.

### Model fails to load / out of memory

1. Check available RAM: `free -h` (Linux) or Activity Monitor (macOS).
2. Compare model file size to free RAM — the full file must fit.
3. Reduce context size (`--ctx-size 4096` or `--ctx-size 2048`).
4. Use a lower quantization (`Q4_K_M` instead of `Q8_0`).
5. If on Apple Silicon, close memory-heavy applications first.

### Binary download fails / "backend mismatch"

- Oprel auto-downloads binaries on first use. If downloads fail behind a corporate proxy, set `OPREL_SSL_NO_VERIFY=1` or configure `ssl_verify` and `ssl_cert_file` in your Oprel config.
- If you switch GPUs (e.g., from AMD to NVIDIA), delete the old binary from `~/.cache/oprel/bin/` — Oprel will re-download the correct variant on next run.

### Slow generation on CPU

- Check your CPU supports AVX2: `grep avx2 /proc/cpuinfo` (Linux). Pre-built binaries target modern CPUs; very old CPUs may fall back to slower code paths.
- Reduce thread count if the system is unresponsive during generation — Oprel uses physical cores minus 2 by default, which is usually right, but you can tune it with `--n-threads`.

### GPU detected but generation is slow

- On Linux with NVIDIA, recall that Oprel's llama.cpp registry currently has no Linux CUDA binary — you're running on CPU or Vulkan. Vulkan provides GPU acceleration through the Vulkan API but may not match native CUDA performance.
- Check `nvidia-smi` during generation — if GPU utilization is low, most layers may be running on CPU. Increase `n_gpu_layers`.
- On AMD with llama.cpp, Vulkan is your only GPU option on Linux — ROCm binaries don't exist for text generation.

---

## Related Topics

- Text generation with llama.cpp
- Image generation with stable-diffusion.cpp
- Model downloads, quantization, and aliases
- Hardware telemetry and GPU detection internals
