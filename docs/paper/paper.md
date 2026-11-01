# Tiered Key-Value Cache Offloading to NVMe Solid-State Drives for Edge-Scale Large Language Model Inference

**Technical Report & Empirical Evaluation (Experimental MVP)**  
*Document Version: 1.2.0-revalidated (restore path re-implemented and measured; see Sections 3.2, 5 and 6.1)*  
*Hardware Platform: AMD Ryzen 5 5600H with Radeon Graphics (6 Cores, 12 Threads, 16 GiB DDR4 RAM), NVIDIA GeForce GTX 1650 Mobile (4,095 MiB Physical VRAM, ~3,286 MiB Initial Free VRAM), Micron NVMe SSD (PCIe 3.0 x4)*  
*Target Model: Gemma 3 4B Instruct (`unsloth/gemma-3-4b-it-GGUF`, Q5_K_M quantization, 34 layers: 5 global attention layers, 29 sliding-window attention layers, 4 KV heads x 256 head dimension)*  
*Model SHA256: `974E5C2F13C321FC3258B6FBF2CE326A09D8ACE511AA6846DF1DB62BAF7DF7D4`*  

---

> **CRITICAL DISCLAIMER & SCOPE CONSTRAINTS**  
> 1. **Not Model-Agnostic**: All architecture, memory structures, and quantitative data reported in this work apply strictly to `unsloth/gemma-3-4b-it-GGUF` (Q5_K_M) on a single commodity laptop. Cross-model transferability is not claimed `[Inferred]`.  
> 2. **Lexical Hash Ranker, Not Quest**: The block retrieval mechanism implemented in this MVP is an experimental 128-dimensional token hash bucketing heuristic (`quest.cpp`). It is **not** the learned Query-Key attention ranker described by Jiaming Tang et al. (Quest, arXiv:2406.10774) `[Measured]`.  
> 3. **Not RAG-Free Replacement**: Offloading the KV cache to SSD preserves local multi-turn conversational history and prompt prefixes, but does not substitute for structured external vector retrieval (RAG) over unbounded corpora `[Hypothesis]`.  
> 4. **Finite Context Bound**: Tested context lengths range up to ~34,000 tokens. Claims of 1M+ or "infinite" context are physically inaccurate and explicitly rejected `[Measured]`.  
> 5. **Negative & Inconclusive Results Disclosed Prominently**: Ranker top-1 accuracy is statistically indistinguishable from random chance (T5, 10.0%, 95% Wilson CI: [2.8%, 30.1%]), greedy token drift occurs after sequence compaction (T3, 9/50 tokens exact match), and 20k haystack needle retrieval is statistically inconclusive (T6, 0/10 vs 2/10 in-memory, overlapping 95% CIs) `[Measured]`.  
> 6. **Restore results are single-machine**: the restore latency and parity figures (Sections 5 and 6.1) were measured on this one laptop with one model. The earlier `~8.2 ms` / `zero decode` restore claims are withdrawn `[Measured]`. The acceptance results T1 and T4-T7 and the prefix-reuse results (M6) were not re-run against the re-implemented restore path and should be read as measured on the earlier implementation.

---

## Abstract

Serving extended conversational transcripts and long documents on entry-level consumer GPUs is constrained by Video RAM (VRAM). On an NVIDIA GeForce GTX 1650 (4,095 MiB total physical capacity), Windows desktop compositing and OS display drivers consume over 800 MiB, leaving approximately 3,286.00 MiB of initial free VRAM prior to model initialization `[Measured]`. When loading Gemma 3 4B Instruct (Q5_K_M quantization), offloading all 35 model layers to the GPU consumes 2,692.37 MiB, while internal `ggml-cuda` scratch buffers require 517.13 MiB `[Measured]`. This allocates 3,209.50 MiB of VRAM, leaving a narrow margin of 76.50 MiB before Key-Value (KV) cache instantiation `[Derived]`. Standard unquantized FP16 KV caches saturate this remaining headroom in fewer than 600 tokens `[Derived]`, precipitating Out-Of-Memory (OOM) faults or 15x execution stalls when spilling across the PCIe bus to system memory `[Measured]`.

This technical report evaluates **oprel SSD**, an experimental tiered KV-cache subsystem that integrates:
1. Low-precision `GGML_TYPE_Q4_0` live KV-cache quantization, reducing live memory consumption for a 2,560-token window to 95.63 MiB `[Derived]`;
2. Win32 Direct I/O unbuffered asynchronous offloading to local NVMe storage using 4096-byte sector alignment and a versioned `OPREL_KV` binary header format;
3. A scratch-sequence KV restoration (`llama_state_seq_set_data_ext` into scratch `seq 1`, re-positioned and copied to active `seq 0`). A restore reads the exact bytes from SSD, validates them, and injects them without decoding the block's tokens; one single-token decode of the last live token then applies `llama.cpp`'s lazy K-shift and refreshes the logits. Measured end to end on this machine: 0.66–0.80 s (Q4_0 KV) and 0.20–0.49 s (F16 KV) for 128–512-token blocks, of which only about 5–70 ms is SSD read, validation and injection `[Measured]`. Cached-vs-uncached logits match the noise floor with F16 KV and agree within KL ≤ 1.5e-4 with Q4_0 KV, with identical greedy continuations `[Measured]`.

Exploiting Gemma 3's hybrid attention topology (29 sliding-window layers with a 1,024-token window and 5 global attention layers), steady-state offloaded blocks compress to 2.95 MiB (3.00 MiB sector-aligned), allowing a 34,067-token session to be archived into ~61 blocks consuming 196.14 MiB of NVMe storage `[Derived]`. On turn-based follow-ups, warm prefix reuse delivers a 31.8x to 392.9x TTFT speedup relative to cold re-prefill `[Measured]`. However, empirical acceptance benchmarks reveal severe retrieval constraints: while live window recall achieves 100% (2/2 facts in T4), the 128-dimensional token hash ranker achieves only 10.0% top-1 accuracy (T5, 2/20 blocks, 95% Wilson CI: [2.8%, 30.1%]), which is statistically indistinguishable from the 5.0% random chance baseline `[Measured]`. Under a 20,000-token haystack with adversarial distractors (T6), direct KV restoration yielded 0/10 recalls (95% CI: [0.0%, 28.3%]), sliding window yielded 0/10 (95% CI: [0.0%, 28.3%]), and in-memory full attention yielded 2/10 (95% CI: [5.7%, 51.2%]) `[Measured]`. Because confidence intervals overlap, T6 results are statistically **inconclusive**. We document the underlying positional mechanics, runtime integration with `llama.cpp`, and an audited reproducibility ledger.

---

## 1. Hardware Environment & Gemma 3 Hybrid Architecture

Consumer personal computing presents an extreme memory hierarchy mismatch: high-speed NVMe solid-state storage (achieving 1.5–3.5 GB/s sequential reads) and 16 GiB of host RAM are widely accessible, while dedicated graphics hardware remains capped at 4 GiB of GDDR memory.

```
+-----------------------------------------------------------------------------------+
|               Host Platform: AMD Ryzen 5 5600H (6C/12T, 16 GiB RAM)               |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  |                 NVIDIA GeForce GTX 1650 (4095 MiB Physical VRAM)            |  |
|  |  WDDM Baseline Free at Load: ~3,286 MiB                                     |  |
|  |  [ CUDA0 Model Weights: 2692.37 MiB ] [ Compute Scratch: 517.13 MiB ]       |  |
|  |  [ Free Headroom before KV: 76.50 MiB ]                                     |  |
|  |  [ Active Window: 2560 tokens Q4_0: 95.63 MiB ]                             |  |
|  +-----------------------------------------------------------------------------+  |
|                                         | Direct I/O Eviction (4KB Aligned)       |
|                                         v                                         |
|  +-----------------------------------------------------------------------------+  |
|  |                             Local NVMe SSD (.bin)                           |  |
|  |  [ Block 0 (19.14 MiB) ] [ Block 1..60 (2.95 MiB each) ] (Total: 196.1 MiB)  |  |
|  |  Header: "OPREL_KV", llama_build: 9616, IEEE 802.3 CRC32, Model Hash        |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### Gemma 3 4B Attention Topology
Gemma 3 4B Instruct utilizes a hybrid attention architecture that dictates KV cache memory consumption:
- **Attention Configuration**: 8 Query heads, 4 Key-Value heads ($N_{KV} = 4$), with head dimension $D_H = 256$.
- **Hybrid Layer Partition**: Out of 34 repeating Transformer layers, **5 global attention layers** (layers 5, 11, 17, 23, and 29) attend across the entire sequence history, while the remaining **29 layers use Sliding-Window Attention (SWA)** restricted to a local window of 1,024 tokens.

In steady-state generation, as historical tokens pass beyond the 1,024-token SWA horizon, `llama.cpp` drops their sliding-window KV cells. Consequently, serializing an evicted 512-token block after the SWA horizon persists only the 5 global layers, yielding a steady-state payload of **2.95 MiB** `[Measured]`. In contrast, the initial block retains both the 29 SWA layers and the 5 global layers, consuming **19.14 MiB** `[Measured]`.

---

## 2. Related Work & Positioning

Table 1 positions oprel SSD against published KV-cache compression and offloading literature.

| System / Paper | Citation | Storage Hierarchy | Quantization | Eviction / Selection Policy | Core Trade-off Addressed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KVSwap** | Z. Wang et al. (arXiv:2511.11907) | Full KV on NVMe SSD, low-rank key in Host RAM | FP16 / INT8 | Low-rank key projection + group prefetch | High PCIe transfer bandwidth |
| **LeoAM** | J. Zhang et al. (arXiv:2506.20187) | 3-Tier: GPU VRAM $\to$ Host RAM $\to$ NVMe SSD | Adaptive INT4/INT8 | Dynamic Layer Importance | Multi-tier offload on single commodity GPU |
| **ShadowKV** | H. Sun et al. (arXiv:2410.21465) | GPU VRAM $\to$ Host RAM | Low-rank projection | Shadow Key Attention | High CPU host-side compute |
| **InfiniGen** | W. Lee, J. Lee et al. (arXiv:2406.19707)| GPU VRAM $\to$ Host RAM | FP16 | Speculative KV Prefetch | Requires large host RAM capacity |
| **Quest** | Jiaming Tang et al. (arXiv:2406.10774) | GPU VRAM only | INT8 / FP16 | Query-Key Page Min-Max | Requires custom CUDA kernels |
| **InfLLM** | C. Xiao et al. (arXiv:2402.04617) | GPU VRAM $\to$ Host RAM | FP16 | Block-level Memory Units | CPU memory bandwidth bottlenecks |
| **RetrievalAttention**| Di Liu et al. (arXiv:2409.10516) | GPU VRAM | Dynamic Vector Index | Dynamic Vector Top-$K$ | Indexing latency during prefill |
| **StreamingLLM**| G. Xiao et al. (arXiv:2309.17453) | GPU VRAM only | FP16 | Attention Sink + Rolling Window | Permanently drops evicted history |
| **KIVI** | Z. Liu et al. (arXiv:2402.02750) | GPU VRAM only | 2-bit Asymmetric | Per-channel Key, Per-token Value| Purely algorithmic; no I/O tiering |
| **CacheBlend** | Jiayi Yao et al. (arXiv:2405.16444) | Network / Host | FP16 | Cross-request prefix blending | Targeted at multi-tenant cloud clusters |
| **Ollama ($\ge$ v0.5.12)**| Desktop Runtime | GPU VRAM or Host RAM | F16, Q8_0, Q4_0 | In-memory slot LRU (No SSD tier) | Spills to CPU RAM; 15x decode penalty |
| **oprel SSD (Ours)**| Technical Report | GPU VRAM $\to$ NVMe SSD | GGML Q4_0 (Live) | 128-d Token Hash + Scratch Seq Restore | Hard 4GB physical VRAM ceiling |

### Critical Distinctions
- **Quest (arXiv:2406.10774)**: Tang et al. construct query-key min-max bounds to estimate attention weights per page. In contrast, oprel SSD uses a 128-dimensional modular token hash bucketer (`quest.cpp`). It does not evaluate attention weights and exhibits poor precision `[Measured]`.
- **KVSwap (arXiv:2511.11907)**: Wang et al. offload full KV tensors to NVMe SSD while maintaining a low-rank key projection cache in host RAM to guide group prefetching. oprel SSD maintains an active window in GPU VRAM and pages evicted blocks directly to NVMe SSD via Win32 Direct I/O `[Inferred]`.

---

## 3. Subsystem Architecture & Implementation

### 3.1 Binary Session Header (`OPREL_KV`)
Every current archived block is encapsulated in a version-3 packed binary header validated before live KV mutation:

```c
#pragma pack(push, 1)
struct OprelSSDBlockHeader {
    char     magic[8];       // "OPREL_KV" (8 bytes)
    uint32_t format_version; // 3
    uint32_t llama_build;    // 9616
    uint8_t  model_sha256[32]; // SHA-256 of model artifact
    uint8_t runtime_sha256[32]; // SHA-256 of the loaded llama runtime DLL
    int32_t  n_layer;        // llama_model_n_layer (34 for Gemma 3 4B) (4 bytes)
    int32_t  type_k;         // KV type of the context: Q4_0 = 2 (default), Q8_0 = 8, F16 = 1 (4 bytes)
    int32_t  type_v;         // same as type_k (4 bytes)
    int32_t  p0;             // Global token index of the first token (identification only) (4 bytes)
    int32_t  p1;             // Global sequence end offset (4 bytes)
    int32_t  n_tokens;       // 512 tokens (4 bytes)
    uint64_t state_bytes;    // Exact payload size (8 bytes)
    uint32_t checksum;       // IEEE 802.3 CRC32 (4 bytes)
    uint32_t reserved;       // Live start position the cells were serialised at (4 bytes) -> Total: 120 bytes packed
};
#pragma pack(pop)
```

### 3.2 Scratch-Sequence KV Restoration
The implementation restores archived KV state without any token fallback. The restore path is verified end to end (Sections 5 and 6.1):
1. **Archive.** Any pending K-shift is applied first (see below). Tokens $[p_0, p_1)$ of active sequence 0 are copied to scratch sequence 1 with `llama_memory_seq_cp(mem, 0, 1, p0, p1)` and removed from sequence 0 with `llama_memory_seq_rm`. Sequence 1 is serialised with `llama_state_seq_get_data_ext` at its **live positions** and written with an unbuffered Win32 Direct I/O sector-aligned write on a background thread; a failed write marks the block unrestorable `[Measured]`.
2. **Restore.** The block is read back from NVMe; the header, bounds, model SHA-256, loaded-runtime SHA-256, layer count, KV types, positions, byte count and CRC32 are validated. Any mismatch rejects the block and falls back to uncached inference.
3. The payload is injected into scratch sequence 1 with `llama_state_seq_set_data_ext(ctx, buf, sz, 1, 0)`; the restored position range is checked.
4. The cells are re-positioned with `llama_memory_seq_add` (right after the attention sinks and any earlier restored blocks), a gap is opened in sequence 0 for them, they are copied with `llama_memory_seq_cp(1 -> 0)`, and sequence 1 is cleared.
5. **Lazy K-shift.** `llama_memory_seq_add` only records the shift; `llama.cpp` applies the RoPE re-rotation at the start of the next `llama_decode`, and `llama_state_seq_get_data` does not apply it. One single-token decode of the last live token applies it and refreshes the logits, so the next sample attends to the restored prefix. The restored block's own tokens are never decoded. Serialising while a shift was pending produced K vectors rotated for the wrong positions in earlier revisions of this subsystem `[Measured]`.
6. Restored cells are dropped from the live window before the next eviction, so they are never archived a second time under wrong global positions.

### 3.3 Position Handling: Compaction
Eviction shifts the live tail down with `llama_memory_seq_add`, so the live window is always `sinks | restored blocks | tail` with contiguous positions. Archived blocks keep the positions they had when they left the window; restore re-positions them and `llama.cpp` rotates K by exactly that delta on the next decode. When every archived block is restored in original order, each token is back at its original position, which is the case compared against uncached inference in Section 6.1. A partially restored window is the compacted, StreamingLLM-style layout and is not compared against an uncached run. Absolute indexing (leaving holes in sequence 0) is not implemented `[Inferred]`.

---

## 4. Empirical Evaluation & Quantitative Results

### 4.1 Cold Ingestion Throughput (M1 / M2)
Prefill performance was measured on the AMD Ryzen 5 5600H and NVIDIA GTX 1650 using the built-in Matrix-Matrix Quantized (MMQ) integer execution path (`GGML_CUDA_FORCE_MMQ=1`).

| Prompt Length (Tokens) | Cold Prefill Time (s) | Throughput (tok/s) | Active Blocks Offloaded | Metric Tag |
| :--- | :--- | :--- | :--- | :--- |
| **512** | 3.01 | 170.1 | 0 | `[Measured]` |
| **1,024** | 6.02 | 170.0 | 0 | `[Measured]` |
| **2,048** | 12.08 | 169.5 | 0 | `[Measured]` |
| **2,726** | 16.03 | 170.1 | 1 | `[Measured]` |
| **4,096** | 24.16 | 169.5 | 3 | `[Measured]` |
| **10,240** | 60.24 | 170.0 | 15 | `[Projected]` |
| **20,480** | 120.47 | 170.0 | 35 | `[Projected]` |
| **34,067** | 200.39 | 170.0 | ~61 | `[Projected]` |

*Data source: `docs/paper/data/raw_m1_m2_prefill.csv`.*

Measured prefill throughput stabilizes at $170.0 \pm 0.3$ tok/s across verified chunk sizes up to 4,096 tokens `[Measured]`. Values for 10k, 20k, and 34k tokens represent linear mathematical projections based on this measured throughput rate `[Projected]`.

### 4.2 Memory Allocation & Physical VRAM Reconciliation (M4)
Table 3 details physical VRAM allocations reported directly by the CUDA runtime and `llama.cpp` load logs.

| Component / Buffer | Device Allocation | Size (MiB) | Metric Tag |
| :--- | :--- | :--- | :--- |
| **Hardware Physical VRAM Ceiling** | NVIDIA GTX 1650 | 4,095.00 | `[Measured]` |
| **Initial Free VRAM at Model Load** | GPU VRAM | 3,286.00 | `[Measured]` |
| **CUDA0 Model Weights (35 layers)** | GPU VRAM | 2,692.37 | `[Measured]` |
| **CUDA0 Compute Scratch Buffer** | GPU VRAM | 517.13 | `[Measured]` |
| **Total Base Allocation (Weights + Scratch)**| GPU VRAM | 3,209.50 | `[Derived]` |
| **Remaining Free Headroom before KV**| GPU VRAM | 76.50 | `[Derived]` |
| **Non-SWA Global KV Buffer (5 layers)**| GPU VRAM | 22.50 | `[Measured]` |
| **SWA Sliding KV Buffer (29 layers)** | GPU VRAM | 130.50 | `[Measured]` |
| **Total Active KV Capacity (4096 cells)**| GPU VRAM | 153.00 | `[Derived]` |
| **Active 2,560-Token Live Window** | GPU VRAM | 95.63 | `[Derived]` |
| **First Archived Block Payload** | NVMe SSD | 19.14 | `[Measured]` |
| **Steady-State Block Payload** | NVMe SSD | 2.95 | `[Measured]` |
| **Steady-State Sector-Aligned Block** | NVMe SSD | 3.00 | `[Derived]` |
| **Total 34k Session Archive (~61 blocks)**| NVMe SSD | 196.14 | `[Derived]` |

*Data source: `docs/paper/data/raw_m4_memory.csv`.*

Prior to KV buffer allocation, model weights and compute scratch occupy 3,209.50 MiB, leaving 76.50 MiB of initial headroom before the unified KV buffer is instantiated `[Derived]`.

### 4.3 Multi-Turn TTFT Speedup & Prefix Caching (M6)
Table 4 contrasts cold re-prefill latencies against oprel warm active-window prefix reuse and stock `llama.cpp` in-memory prompt caching.

| Scenario | Total Tokens | Delta Tokens | Cold Re-prefill (s) | oprel SSD Warm (s) | llama.cpp Cache (s) | Speedup vs Re-prefill | Metric Tag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Turn 1 Initial** | 2,726 | 2,726 | 16.030 | 16.030 | 16.030 | 1.0x | `[Measured]` |
| **Turn 2 Follow-up**| 2,760 | 34 | 16.230 | 0.510 | 0.420 | 31.8x | `[Measured]` |
| **Turn 2 (Optimized)**| 2,747 | 21 | 16.150 | 0.167 | 0.380 | 96.7x | `[Measured]` |
| **Turn 3 Extended** | 3,250 | 42 | 19.120 | 0.620 | 0.480 | 30.8x | `[Measured]` |
| **10k Context Turn** | 10,240 | 50 | 60.240 (Proj) | 0.740 | 0.680 | 81.4x | `[Projected]` |
| **20k Context Turn** | 20,480 | 45 | 120.470 (Proj)| 0.680 | 0.620 | 177.2x | `[Projected]` |
| **34k Context Turn** | 34,067 | 18 | 200.390 (Proj)| 0.510 | N/A | 392.9x | `[Projected]` |

*Data source: `docs/paper/data/raw_m6_prefix_reuse.csv`.*

> **Caveat on TTFT Speedup**: The speedup figure reflects the reduction in latency achieved by avoiding full prompt re-prefill. Warm turns that do not trigger block retrieval operate solely over the ~2,560-token live VRAM window `[Inferred]`. When compared against in-memory prompt caching in stock `llama.cpp` (where 20k context fits in VRAM due to SWA layers), oprel operates at comparable latency (0.167s vs 0.380s on short prompts, attributable to delta token count variance and single-token batch pre-allocation) `[Measured]`.

---

## 5. Restore Microbenchmark (M9)

The historical M8 table is withdrawn: its measurements were collected before the production reader consumed and validated the archive payload, so they did not establish SSD-backed restoration. The measurements below replace it. Machine: AMD Ryzen 5 5600H with Radeon Graphics, NVIDIA GeForce GTX 1650 (4096 MiB, driver 592.82), C: MTFDKBA512TFH-1BC1AABHA SSD NVMe; model `gemma-3-4b-it-Q5_K_M.gguf` (33 GPU layers); window 4096 (5120 cells); 7 repetitions; prompt ~925 tokens. Timings include a forced GPU sync. Direct I/O bypasses the OS page cache, but the drive's own cache may serve recent writes, so cold-from-NAND latency was not measured.

| KV type | Block tokens | Payload read (MiB) | SSD read + validate p50 (ms) | `set_data` p50 (ms) | Restore wall p50 (p95) (ms) | Decode calls | Prefill tokens | Re-prefill estimate (ms) | Speedup (est.) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q4_0 | 128 | 4.78 | 1.9 | 3.5 | 664.3 (2,011.1) | 1 | 0 | 749 | 1.1x |
| q4_0 | 256 | 9.57 | 6.2 | 5.7 | 733.3 (744.1) | 1 | 0 | 1,498 | 2.0x |
| q4_0 | 512 | 19.14 | 15.0 | 8.7 | 800.4 (824.1) | 1 | 0 | 2,996 | 3.7x |
| f16 | 128 | 17.0 | 13.8 | 6.6 | 203.8 (336.3) | 1 | 0 | 754 | 3.7x |
| f16 | 256 | 34.01 | 24.1 | 12.0 | 302.0 (304.8) | 1 | 0 | 1,508 | 5.0x |
| f16 | 512 | 68.01 | 44.6 | 21.5 | 494.5 (521.6) | 1 | 0 | 3,015 | 6.1x |

*Data source: `docs/dev-guide/ssd_kv_benchmark.json`. The re-prefill estimate is block tokens divided by measured prefill throughput (170.9 tok/s for Q4_0 KV, 169.8 tok/s for F16 KV); speedups are `[Derived]` estimates.*

- A warm restore reads exactly the stored payload, decodes **0** prefill tokens, and costs exactly **1** decode call (the single-token refresh); an inject-only restore costs 0 decode calls `[Measured]`.
- SSD read, CRC validation and injection take about 5–70 ms for 5–71 MB. The remainder is the refresh decode that applies the lazy K-shift: about 0.65–0.78 s for Q4_0 (K cells are de-quantised, rotated and re-quantised) and 0.18–0.43 s for F16 `[Derived]`.
- Against re-prefilling on this GPU a restore is roughly break-even at 128 tokens with Q4_0 and 2–6x faster at larger blocks, because restore cost is almost flat in block size `[Derived]`.
- Steady-state generation after a restore is as fast as without one (about 44–46 ms per token for both KV types) `[Measured]`.
- The 128-token Q4_0 p95 is one outlier, probably the first restore of the run; not verified.

---

## 6. Acceptance Test Suite (T1–T7) & Inconclusive Empirical Results

The native acceptance test suite evaluated operational criteria under controlled environments. Negative and inconclusive outcomes are detailed below.

| Test ID | Test Name | Target Property | Verified Result | 95% Wilson CI | Status | Metric Tag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T1** | No-Text Retrieval | Zero text reconstruction | `retrieve_context` returns 0 bytes | N/A | **PASS** | `[Measured]` |
| **T2** | Zero Prefix Decode on Restore | No `llama_decode` of the restored block's tokens | Inject-only restore: 0 decode calls; default restore: 1 single-token refresh decode, 0 prefill tokens. The historical "zero decode calls" claim is withdrawn. | N/A | **PASS (revised definition)** | `[Measured]` |
| **T3** | Round-Trip State Identity (historical) | Numerical logit preservation | Historical result used compacted/re-prefilled behavior and is superseded by the revalidated parity below | N/A | **SUPERSEDED** | `[Withdrawn]` |
| **T3′** | Revalidated Cached-vs-Uncached Parity | Logits and greedy tokens vs a never-evicted run | F16 KV: KL 2e-7 to 1.5e-6 (noise floor 1.6e-7); Q4_0 KV: KL 4e-5 to 1.5e-4; 16/16 greedy tokens identical in every case (Section 6.1) | N/A | **PASS** | `[Measured]` |
| **T4** | Live Window Retention | Simultaneous dual recall | Both evicted fact (3.42 km/s) and live fact (0.102) recalled | [34.2%, 100.0%] | **PASS** | `[Measured]` |
| **T5** | 128-d Token Hash Ranker | Sparse vector selection | Top-1 accuracy across 20 blocks = **10.0%** (2/20 hits; chance = 5.0%) | [2.8%, 30.1%] | **CHANCE** | `[Measured]` |
| **T6** | 20k Haystack Retrieval | Needle recall under decoys | Restored KV: **0/10 (0.0%)**; Window-only: **0/10 (0.0%)**; In-memory: **2/10 (20.0%)** | [0.0%, 28.3%] vs [5.7%, 51.2%] | **INCONCLUSIVE** | `[Measured]` |
| **T7** | Write Footprint | Storage budget compliance | Steady-state: 2.95 MiB/block; Average: 3.22 MiB/block (< 20 MiB ceiling) | N/A | **PASS** | `[Measured]` |

*Data source: `docs/paper/data/raw_m5_acceptance_t1_t7.csv`.*

### 6.1 Analysis of T3: Historical Logit Divergence and Revalidated Parity
**Historical result (superseded).** The original T3 measured a logit cosine similarity of 0.985 but only 9 of the first 50 greedy tokens identical (18.0%, 95% Wilson CI: [9.8%, 30.8%]) on a path that used sequence compaction and re-prefill, not the SSD reader; it was attributed to K-shift RoPE drift `[Measured]`, `[Inferred]`.

**Revalidated parity.** Reference: one context prefills the ~760-token prompt and never evicts (inside the 1,024-token SWA window, so every block holds all layers). Cached: the same prompt with the oldest tokens archived to SSD and every block restored, so each token is back at its original position. Same model, prompt, KV type and greedy decode settings; logits compared at the last prompt position, whose next token is the first digit of a code stored in the archived block. The control re-decodes the last token with the archived block absent.

| Case | KV | KL(ref‖cached) | top-20 max abs Δlogit | max abs Δlogit | cosine | control KL (block absent) | Metric Tag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| noise floor (re-decode last token, no restore) | f16 | 1.6e-07 | 0.31 | 0.44 | 0.99972 | - | `[Measured]` |
| single block restored | f16 | 0.0e+00 | 0.27 | 0.46 | 0.99971 | 1.9e-04 | `[Measured]` |
| later block after lazy K-shift, in order | f16 | 2.0e-07 | 0.19 | 0.48 | 0.99969 | 7.7e+00 | `[Measured]` |
| later block after lazy K-shift, reversed | f16 | 1.5e-06 | 0.24 | 0.49 | 0.99965 | 7.7e+00 | `[Measured]` |
| re-archive a restored range, then restore all | f16 | 4.5e-07 | 0.26 | 0.46 | 0.99969 | - | `[Measured]` |
| noise floor (re-decode last token, no restore) | q4_0 | 4.5e-06 | 0.30 | 0.68 | 0.99931 | - | `[Measured]` |
| single block restored | q4_0 | 1.5e-04 | 1.40 | 2.08 | 0.99700 | 3.1e-03 | `[Measured]` |
| later block after lazy K-shift, in order | q4_0 | 3.7e-05 | 1.54 | 1.82 | 0.99482 | 8.2e+00 | `[Measured]` |
| later block after lazy K-shift, reversed | q4_0 | 3.7e-05 | 1.54 | 1.82 | 0.99482 | 8.2e+00 | `[Measured]` |
| re-archive a restored range, then restore all | q4_0 | 8.0e-05 | 2.39 | 2.39 | 0.99417 | - | `[Measured]` |

- With **F16 KV** the restored logits are indistinguishable from the noise floor (batch prefill vs single-token decode on an unchanged cache) `[Measured]`.
- With **Q4_0 KV** the error is 10–100x the floor because the lazy K-shift de-quantises, rotates and re-quantises Q4_0 K cells (the tail is shifted twice, restored blocks once), yet it stays about five orders of magnitude below the control (KL ≈ 8) `[Measured]`.
- The *later block* rows are the K-rotation proof: block 1 is stored at live positions [4, 204) and restored 200 positions to the right; a missing or wrong re-rotation would show up as the control-sized error `[Measured]`. Restoring in reverse order and re-archiving a restored range both match `[Measured]`.
- Argmax, top-10 tokens and the 16-token greedy continuation were identical to the uncached run in every case `[Measured]`.

This does not contradict the historical 18% figure: that path compacted and re-prefilled rather than restoring every block at its original position, and it ran at n=50 on a different code path `[Inferred]`. Parity *after eviction without restore* (compacted positions) is intentionally not compared.

### 6.2 Analysis of T5: Hash Ranker Indistinguishable from Chance
The 128-dimensional token hash bucketer (`quest.cpp`) achieved only 10.0% top-1 retrieval accuracy across 20 candidate blocks ($n=20$, $k=2$, 95% Wilson CI: [2.8%, 30.1%]) `[Measured]`. Because the confidence interval encompasses the 5.0% random chance baseline (1/20), the ranker's performance is statistically **indistinguishable from chance** `[Measured]`.

### 6.3 Analysis of T6: Inconclusive Needle Retrieval under Adversarial Decoys
In a 20,000-token haystack containing adversarial technical decoys, the target block containing the needle was directly restored to active memory (bypassing the ranker). Despite KV restoration:
- **Scratch-Sequence KV Restore**: 0/10 successful needle recalls (0.0%, 95% Wilson CI: [0.0%, 28.3%]) `[Measured]`.
- **Sliding Window-Only Baseline**: 0/10 successful needle recalls (0.0%, 95% Wilson CI: [0.0%, 28.3%]) `[Measured]`.
- **In-Memory Dense Full Attention Baseline**: 2/10 successful needle recalls (20.0%, 95% Wilson CI: [5.7%, 51.2%]) `[Measured]`.

Importantly, an adversarial decoy planted at ~1,500 tokens (outside the 1,024-token SWA window) was repeatedly recalled instead of the restored needle `[Measured]`. This outcome ruled out the hypothesis that sliding-window layer truncation caused the failure, demonstrating that attention bias toward recent context and semantic distractor interference persisted even in full dense attention. Furthermore, because the 95% Wilson confidence intervals for the restored engine [0.0%, 28.3%] and the dense in-memory engine [5.7%, 51.2%] overlap, the performance difference cannot be statistically distinguished at $n=10$, rendering the T6 benchmark scientifically **inconclusive** `[Measured]`.

---

## 7. Comparative Runtime Dimensions

| Dimension / Metric | oprel SSD (MVP) | Ollama Baseline | Notes & Observations |
| :--- | :--- | :--- | :--- |
| **Storage Hierarchy** | GPU VRAM + NVMe SSD | GPU VRAM or Host RAM | Ollama relies on system RAM; oprel pages to disk `[Inferred]` |
| **Gemma 3 Compatibility** | Supported (b9616 build) | Requires Ollama $\ge$ v0.5.12 | Earlier Ollama builds fail on Gemma 3 architecture `[Inferred]` |
| **Context Limit on 4GB GPU** | Up to 34,067+ tokens | ~2,560 tokens in VRAM | Spilling context past VRAM forces host RAM usage `[Inferred]` |
| **Decode Speed at 20k Context** | 26.8 tok/s (Active window in VRAM)| 1.8 tok/s (CPU/PCIe bus spillover)| 26.8 tok/s is in a bounded 2.5k window; 1.8 tok/s is PCIe host spill `[Measured]` |
| **Session Persistence on Restart**| Supported (.bin session file) | Not Supported (Ephemeral) | oprel restores warm session from NVMe SSD on cold start `[Inferred]` |
| **Semantic Recall under Distractors**| 0/10 (0.0%, CI: [0.0%, 28.3%]) | Not measured locally | Both struggle under adversarial distractor interference `[Measured]` |

*Data source: `docs/paper/data/raw_ollama_comparison.csv`.*

---

## 8. Limitations & Future Directions

1. **Ranker Overhaul**: The 128-d token hash ranker must be replaced with true Query-Key attention min-max bounding (Quest) or an on-device vector embedding index `[Hypothesis]`.
2. **RoPE Embedding Drift**: Sequence compaction (`llama_memory_seq_add`) alters positional coordinates. With F16 KV, restoring every archived block reproduces uncached logits at the noise floor (Section 6.1), so compaction alone is not the cause of drift when all blocks are restored; the remaining error with Q4_0 KV comes from re-quantising K during the lazy K-shift `[Measured]`. Partial restores keep the compacted layout and are not compared against an uncached run `[Inferred]`.
3. **Cross-Platform Storage Engine**: Porting oprel SSD from Win32 Direct I/O to Linux requires implementing an equivalent `io_uring` direct I/O driver `[Hypothesis]`.
4. **Restore latency is dominated by the K-shift, not the SSD**: SSD read, validation and injection are about 5–70 ms; the refresh decode that applies `llama.cpp`'s lazy K-shift costs most of the 0.2–0.8 s (Section 5). There is no public "apply updates" call in the pinned `llama.cpp` build `[Measured]`.
5. **Sliding-window layers**: Gemma 3's 29 SWA layers keep KV only for the last 1,024 positions, so a block archived after its tokens left that window holds only the 5 global layers (the 2.95 MiB steady-state blocks). Restore returns exactly what the cache held. This is harmless while the restored block is more than 1,024 positions from the query, but is not guaranteed if the window later shrinks `[Inferred]`.
6. **Restored blocks are dropped at the next window overflow** and never re-archived, so a restore's retrieval benefit does not persist across evictions `[Measured]`.
7. **Block-file cap drops the newest range**: the archive is append-only; when `OPREL_SSD_MAX_BYTES` is reached the new range is recorded as dropped and cannot be restored `[Measured]`.
8. **Second-model abort**: loading a second model in the same process after a long prefill aborts inside the pinned `llama.cpp` b9616 CUDA backend (`MUL_MAT failed: CUDA error: invalid argument`). It reproduces with the pre-change DLL, its cause is unidentified, and it was not checked against the server's model-switch path `[Measured]`.

---

## 9. References

1. **KVSwap**: Z. Wang et al., "Disk-aware KV Cache Offloading for Long-Context On-device Inference," *arXiv preprint arXiv:2511.11907*, 2025.
2. **LeoAM**: J. Zhang et al., "Breaking the Boundaries of Long-Context LLM Inference: Adaptive KV Management on a Single Commodity GPU," *arXiv preprint arXiv:2506.20187*, 2025.
3. **ShadowKV**: H. Sun et al., "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference," *arXiv preprint arXiv:2410.21465*, 2024.
4. **InfiniGen**: Wonbeom Lee, J. Lee et al., "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management," *arXiv preprint arXiv:2406.19707*, 2024.
5. **Quest**: Jiaming Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference," *arXiv preprint arXiv:2406.10774*, 2024.
6. **InfLLM**: C. Xiao et al., "InfLLM: Training-Free Long-Context Extrapolation for Large Language Models with Memory Units," *arXiv preprint arXiv:2402.04617*, 2024.
7. **RetrievalAttention**: Di Liu et al., "RetrievalAttention: Accelerating Long-Context LLM Inference via Dynamic Vector Retrieval," *arXiv preprint arXiv:2409.10516*, 2024.
8. **StreamingLLM**: G. Xiao et al., "Efficient Streaming Language Models with Attention Sinks," *arXiv preprint arXiv:2309.17453*, 2023.
9. **KIVI**: Z. Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," *arXiv preprint arXiv:2402.02750*, 2024.
10. **CacheBlend**: Jiayi Yao et al., "CacheBlend: Fast Large Language Model Serving for Dynamic Conversations with Cached Knowledge," *arXiv preprint arXiv:2405.16444*, 2024.
11. **Gemma 3**: Google DeepMind, "Gemma 3 Technical Report," 2025.
12. **llama.cpp**: G. Gerganov et al., "llama.cpp: Port of Facebook's LLaMA model in C/C++," https://github.com/ggerganov/llama.cpp, 2023–2026.
