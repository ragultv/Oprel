# Numbers Traceability Matrix (oprel SSD KV-Cache Subsystem)

This document provides an objective audit trail connecting quantitative claims, hardware metrics, latency figures, and benchmark scores in `docs/paper/paper.md` and `docs/dev-guide/DEVELOPER_GUIDE.md` directly to their underlying source data files in `docs/paper/data/`. Measurements added by the restore re-validation (Section 6) come from `docs/dev-guide/ssd_kv_benchmark.json` and `docs/dev-guide/ssd_kv_parity.json`, produced by `scripts/bench_ssd_kv.py` and `tests/unit/test_ssd_parity.py`; the full method and limitations are in `docs/dev-guide/ssd_kv_validation.md`.

---

## 1. Hardware & Physical Memory Metrics

| Metric Description | Stated Value | Metric Tag | Source Data File | Key / Location in File |
| :--- | :--- | :--- | :--- | :--- |
| **GPU Total Physical VRAM** | 4,095.00 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `gpu_total_physical_vram` |
| **Initial Free VRAM at Load** | 3,286.00 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `gpu_vram_free_at_load` |
| **Gemma 3 4B CUDA0 Weights** | 2,692.37 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `gemma_3_4b_cuda0_weights` |
| **CPU-Mapped Weight Tensors** | 525.13 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `gemma_3_4b_cpu_mapped_weights` |
| **CUDA0 Compute Scratch Buffer**| 517.13 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `cuda0_compute_buffer` |
| **CUDA Host Compute Buffer** | 18.55 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `cuda_host_compute_buffer` |
| **Total Base GPU Allocation** | 3,209.50 MiB | `[Derived]` | `raw_m4_memory.csv` | Weights (2,692.37) + Compute (517.13) |
| **Remaining Free Margin before KV**| 76.50 MiB | `[Derived]` | `raw_m4_memory.csv` | Row `gpu_headroom_remaining` (3,286.00 - 3,209.50) |
| **Non-SWA Global KV (5 layers)**| 22.50 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `active_kv_cache_global_5_layers` |
| **SWA Sliding KV (29 layers)** | 130.50 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `active_kv_cache_swa_29_layers` |
| **Total Active KV Capacity (4k)**| 153.00 MiB | `[Derived]` | `raw_m4_memory.csv` | Row `active_kv_cache_total_capacity` (22.50 + 130.50) |
| **Active 2,560-Token Window** | 95.63 MiB | `[Derived]` | `raw_m4_memory.csv` | Row `active_kv_cache_2560_window` |
| **First Archived Block Payload**| 19.14 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `first_archived_block_size` |
| **Steady-State Block Payload** | 2.95 MiB | `[Measured]` | `raw_m4_memory.csv` | Row `steady_state_block_size` |
| **Steady-State Sector-Aligned** | 3.00 MiB | `[Derived]` | `raw_m4_memory.csv` | Row `steady_state_sector_aligned_block` |
| **34k Session Archived Blocks** | ~61 blocks | `[Derived]` | `raw_m4_memory.csv` | Row `archived_blocks_34k_session` ((34,067 - 2,560) / 512) |
| **Total 34k Session Disk Footprint**| 196.14 MiB | `[Derived]` | `raw_m4_memory.csv` | Row `total_disk_footprint_34k_session` (19.14 + 60 * 2.95) |
| **Header Struct Size (Packed)** | 120 bytes | `[Measured]` | `oprel_ssd.h` | `static_assert(sizeof(OprelSSDBlockHeader) == 120)` |
| **Header Struct Size (Aligned)**| 4096-byte sector allocation | `[Derived]` | `llama_ssd_wrapper.cpp` | Payload allocation rounds header + state bytes to sector size |
| **Model GGUF File SHA256** | 974E5C2F... | `[Measured]` | `raw_m4_memory.csv` | Row `model_file_sha256` |

---

## 2. Prefill Latency & Throughput (M1 / M2)

| Prompt Sequence (Tokens) | Cold Prefill (s) | Throughput (tok/s) | Metric Tag | Source Data File | Row Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **512** | 3.01 | 170.1 | `[Measured]` | `raw_m1_m2_prefill.csv` | Row 1 (`prompt_tokens=512`) |
| **1,024** | 6.02 | 170.0 | `[Measured]` | `raw_m1_m2_prefill.csv` | Row 2 (`prompt_tokens=1024`) |
| **2,048** | 12.08 | 169.5 | `[Measured]` | `raw_m1_m2_prefill.csv` | Row 3 (`prompt_tokens=2048`) |
| **2,726** | 16.03 | 170.1 | `[Measured]` | `raw_m1_m2_prefill.csv` | Row 4 (`prompt_tokens=2726`) |
| **4,096** | 24.16 | 169.5 | `[Measured]` | `raw_m1_m2_prefill.csv` | Row 5 (`prompt_tokens=4096`) |
| **10,240** | 60.24 | 170.0 | `[Projected]` | `raw_m1_m2_prefill.csv` | Row 6 (`prompt_tokens=10240`) |
| **20,480** | 120.47 | 170.0 | `[Projected]` | `raw_m1_m2_prefill.csv` | Row 7 (`prompt_tokens=20480`) |
| **34,067** | 200.39 | 170.0 | `[Projected]` | `raw_m1_m2_prefill.csv` | Row 8 (`prompt_tokens=34067`) |

---

## 3. Warm Prefix Reuse Time-To-First-Token (M6)

| Scenario | Total Tokens | Delta Tokens | Cold Prefill (s) | Warm TTFT (s) | Speedup vs Re-prefill | Metric Tag | Source Data File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Turn 1 Initial** | 2,726 | 2,726 | 16.030 | 16.030 | 1.0x | `[Measured]` | `raw_m6_prefix_reuse.csv`, Row 1 |
| **Turn 2 Follow-up**| 2,760 | 34 | 16.230 | 0.510 | 31.8x | `[Measured]` | `raw_m6_prefix_reuse.csv`, Row 2 |
| **Turn 2 Optimized**| 2,747 | 21 | 16.150 | 0.167 | 96.7x | `[Measured]` | `raw_m6_prefix_reuse.csv`, Row 3 |
| **Turn 3 Extended** | 3,250 | 42 | 19.120 | 0.620 | 30.8x | `[Measured]` | `raw_m6_prefix_reuse.csv`, Row 4 |
| **10k Context Turn** | 10,240 | 50 | 60.240 (Proj) | 0.740 | 81.4x | `[Projected]` | `raw_m6_prefix_reuse.csv`, Row 5 |
| **20k Context Turn** | 20,480 | 45 | 120.470 (Proj)| 0.680 | 177.2x | `[Projected]` | `raw_m6_prefix_reuse.csv`, Row 6 |
| **34k Context Turn** | 34,067 | 18 | 200.390 (Proj)| 0.510 | 392.9x | `[Projected]` | `raw_m6_prefix_reuse.csv`, Row 7 |

---

## 4. Acceptance Test Suite (T1–T7)

| Test ID | Test Description | Measured Quantity | 95% Wilson CI | Metric Tag | Status | Source Data File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T1** | No-Text Retrieval | 0 text bytes returned | N/A | `[Measured]` | PASS | `raw_m5_acceptance_t1_t7.csv`, Row 1 |
| **T2** | Zero Decode Calls | Historical claim withdrawn. Replaced by Section 6 (inject-only restore: 0 decode calls; default restore: 1 single-token refresh decode, 0 prefill tokens) | N/A | `[Withdrawn]` | SUPERSEDED | `raw_m5_acceptance_t1_t7.csv`, Row 2 |
| **T3** | Round-Trip Cosine Sim | Historical result is not evidence for the revised reader; see Section 6 parity | N/A | `[Withdrawn]` | SUPERSEDED | `raw_m5_acceptance_t1_t7.csv`, Row 3 |
| **T3** | Greedy Token Match (historical, compacted/re-prefilled path) | 9/50 tokens (18.0%) | [9.8%, 30.8%] | `[Measured]` | PARTIAL (historical; see Section 6 for the revalidated 16/16) | `raw_m5_acceptance_t1_t7.csv`, Row 4 |
| **T4** | Dual Fact Recall | 2/2 facts recalled | [34.2%, 100.0%] | `[Measured]` | PASS | `raw_m5_acceptance_t1_t7.csv`, Row 5 |
| **T5** | 128-d Token Hash Ranker | 2/20 blocks top-1 (10.0%) | [2.8%, 30.1%] | `[Measured]` | INDISTINGUISHABLE_FROM_CHANCE | `raw_m5_acceptance_t1_t7.csv`, Row 6 |
| **T6** | Restored KV Recall | 0/10 needles (0.0%) | [0.0%, 28.3%] | `[Measured]` | INCONCLUSIVE | `raw_m5_acceptance_t1_t7.csv`, Row 7 |
| **T6** | Window-Only Recall | 0/10 needles (0.0%) | [0.0%, 28.3%] | `[Measured]` | INCONCLUSIVE | `raw_m5_acceptance_t1_t7.csv`, Row 8 |
| **T6** | In-Memory Dense Recall | 2/10 needles (20.0%) | [5.7%, 51.2%] | `[Measured]` | INCONCLUSIVE | `raw_m5_acceptance_t1_t7.csv`, Row 9 |
| **T7** | Steady-State Block Size | 2.95 MiB payload | N/A | `[Measured]` | PASS | `raw_m5_acceptance_t1_t7.csv`, Row 10 |
| **T7** | First Block Size | 19.14 MiB payload | N/A | `[Measured]` | PASS | `raw_m5_acceptance_t1_t7.csv`, Row 11 |

---

## 5. Subsystem Restoration Microbenchmark (M8) — withdrawn, superseded by Section 6

The historical M8 measurements did not verify the production SSD reader and must not be used as current benchmark results. The old `~8.2 ms` / `zero decode` figures are withdrawn; current measurements are in Section 6.

| Microbenchmark Operation | Latency (ms) | % of Total | Metric Tag | Source Data File |
| :--- | :--- | :--- | :--- | :--- |
| **SSD-to-KV restoration (historical M8)** | Withdrawn | N/A | `[Withdrawn]` | `raw_m8_restore_breakdown.csv` (historical only) |

---

## 6. Restore Re-validation (M9 restore benchmark, M10 parity, M11 decode accounting)

Machine: AMD Ryzen 5 5600H with Radeon Graphics, NVIDIA GeForce GTX 1650 (4096 MiB, driver 592.82), cache drive C: MTFDKBA512TFH-1BC1AABHA SSD NVMe. Model `gemma-3-4b-it-Q5_K_M.gguf` (33 GPU layers), llama.cpp b9616 (CUDA), window 4096 (5120 cells), 7 repetitions per cell, prompt ~925 tokens for restore runs. Measured 2026-11-01 21:05:14. Single machine, single model: not a general claim.

### 6.1 Cold prefill (re-measured with a forced GPU sync) — `[Measured]`
Source: `docs/dev-guide/ssd_kv_benchmark.json`, `results.<kv>.cold_prefill.<tokens>`.

| KV type | Prompt tokens | Cold prefill p50 (ms) | stdev (ms) | Throughput (tok/s) |
| :--- | :--- | :--- | :--- | :--- |
| q4_0 | 319 | 1,879.7 | 2.4 | 169.7 |
| q4_0 | 621 | 3,505.9 | 4.0 | 177.1 |
| q4_0 | 925 | 5,413.0 | 5.2 | 170.9 |
| f16 | 318 | 1,916.9 | 9.6 | 165.9 |
| f16 | 620 | 3,530.1 | 10.6 | 175.6 |
| f16 | 925 | 5,448.9 | 14.5 | 169.8 |

These independently corroborate the ~170 tok/s of Section 2.

### 6.2 Warm restore (M9) — `[Measured]`; re-prefill estimate and speedup `[Derived]`
Source: `docs/dev-guide/ssd_kv_benchmark.json`, `results.<kv>.archive_and_restore.<block_tokens>` (`restore_wall`, `restore_ssd_read_and_validate`, `restore_set_data`, `bytes_read_per_restore`, `decode_calls_per_restore`, `prefill_tokens_per_restore`). Re-prefill estimate = block tokens / median measured prefill throughput (170.9 tok/s q4_0, 169.8 tok/s f16); it is an estimate, not a separate measurement.

| KV type | Block tokens | Payload read (MiB) | SSD read + validate p50 (ms) | `set_data` p50 (ms) | Restore wall p50 (p95) (ms) | Decode calls | Prefill tokens | Re-prefill estimate (ms) | Speedup (est.) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q4_0 | 128 | 4.78 | 1.9 | 3.5 | 664.3 (2,011.1) | 1 | 0 | 749 | 1.1x |
| q4_0 | 256 | 9.57 | 6.2 | 5.7 | 733.3 (744.1) | 1 | 0 | 1,498 | 2.0x |
| q4_0 | 512 | 19.14 | 15.0 | 8.7 | 800.4 (824.1) | 1 | 0 | 2,996 | 3.7x |
| f16 | 128 | 17.0 | 13.8 | 6.6 | 203.8 (336.3) | 1 | 0 | 754 | 3.7x |
| f16 | 256 | 34.01 | 24.1 | 12.0 | 302.0 (304.8) | 1 | 0 | 1,508 | 5.0x |
| f16 | 512 | 68.01 | 44.6 | 21.5 | 494.5 (521.6) | 1 | 0 | 3,015 | 6.1x |

Restore wall time minus SSD read and `set_data` is the refresh decode that applies llama.cpp's lazy K-shift: about 0.65–0.78 s for q4_0 and 0.18–0.43 s for f16 `[Derived]`. The 128-token q4_0 p95 (2,011 ms) is one outlier, probably the first restore of the run; not verified.

### 6.3 Next-token generation — `[Measured]`
Source: `docs/dev-guide/ssd_kv_benchmark.json`, `results.<kv>.generation`.

| KV type | Scenario | First token p50 / p95 (ms) | Steady-state p50 / p95 / max (ms) | Decode calls per 32-token run |
| :--- | :--- | :--- | :--- | :--- |
| q4_0 | no restore | 46.8 / 89.0 | 44.0 / 52.1 / 60.1 | 32 |
| q4_0 | after restore | 47.3 / 52.5 | 44.5 / 53.0 / 108.9 | 32 |
| f16 | no restore | 48.7 / 68.7 | 45.3 / 57.4 / 66.1 | 32 |
| f16 | after restore | 40.8 / 56.1 | 45.9 / 53.2 / 66.5 | 32 |

### 6.4 Cached-vs-uncached parity (M10) — `[Measured]`
Source: `docs/dev-guide/ssd_kv_parity.json`; asserted by `tests/unit/test_ssd_parity.py`. Reference = one context that prefills the ~760-token prompt and never evicts (inside the 1,024-token SWA window, so blocks hold all layers); cached = same prompt with oldest tokens archived with `archive_oldest` then every block restored, so each token is back at its original position. The control re-decodes the last token with the archived block absent. All cases: identical argmax, identical top-10 tokens, identical 16-token greedy continuation.

| Case | KV | KL(ref‖cached) | top-20 max abs Δlogit | max abs Δlogit | cosine | control KL (block absent) | Metric Tag | Key in `ssd_kv_parity.json` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| noise floor (re-decode last token, no restore) | f16 | 1.6e-07 | 0.31 | 0.44 | 0.99972 | - | `[Measured]` | `f16/noise_floor` |
| single block restored | f16 | 0.0e+00 | 0.27 | 0.46 | 0.99971 | 1.9e-04 | `[Measured]` | `f16/single_block` |
| later block after lazy K-shift, in order | f16 | 2.0e-07 | 0.19 | 0.48 | 0.99969 | 7.7e+00 | `[Measured]` | `f16/later_block_in_order` |
| later block after lazy K-shift, reversed | f16 | 1.5e-06 | 0.24 | 0.49 | 0.99965 | 7.7e+00 | `[Measured]` | `f16/later_block_reversed` |
| re-archive a restored range, then restore all | f16 | 4.5e-07 | 0.26 | 0.46 | 0.99969 | - | `[Measured]` | `f16/rearchive` |
| noise floor (re-decode last token, no restore) | q4_0 | 4.5e-06 | 0.30 | 0.68 | 0.99931 | - | `[Measured]` | `q4_0/noise_floor` |
| single block restored | q4_0 | 1.5e-04 | 1.40 | 2.08 | 0.99700 | 3.1e-03 | `[Measured]` | `q4_0/single_block` |
| later block after lazy K-shift, in order | q4_0 | 3.7e-05 | 1.54 | 1.82 | 0.99482 | 8.2e+00 | `[Measured]` | `q4_0/later_block_in_order` |
| later block after lazy K-shift, reversed | q4_0 | 3.7e-05 | 1.54 | 1.82 | 0.99482 | 8.2e+00 | `[Measured]` | `q4_0/later_block_reversed` |
| re-archive a restored range, then restore all | q4_0 | 8.0e-05 | 2.39 | 2.39 | 0.99417 | - | `[Measured]` | `q4_0/rearchive` |

Test tolerances (`TOLERANCE` in `test_ssd_parity.py`): f16 KL ≤ 1e-5 and top-20 |Δlogit| ≤ 0.75; q4_0 KL ≤ 1e-3 and top-20 |Δlogit| ≤ 4.0; control KL must exceed the restored error by at least 10x.

### 6.5 Decode accounting (M11) — `[Measured]`
Source: `tests/unit/test_ssd_hardening.py::test_restore_reads_ssd_and_does_not_decode_prefix` and `test_ssd_parity.py::test_single_block_restore_matches_uncached_logits_and_greedy_tokens`.

| Quantity | Value | Metric Tag |
| :--- | :--- | :--- |
| `llama_decode` calls, inject-only restore (`refresh_logits=False`) | 0 | `[Measured]` |
| `llama_decode` calls, default restore | 1 (single-token refresh of the last live token) | `[Measured]` |
| Prefill tokens decoded for the restored block | 0 | `[Measured]` |
| Bytes read per restore | equals stored payload (`bytes_read_per_restore`) | `[Measured]` |
| Restored cells bit-identical to stored payload (no-shift case) | yes | `[Measured]` |

---

## 7. Audit Notes
- **Distinction of Projected Quantities**: Sequence lengths beyond 4,096 tokens (10k, 20k, 34k) for cold prefill are explicitly marked `[Projected]`.
- **Statistical Inconclusiveness**: T6 needle retrieval scores (0/10 vs 2/10) possess overlapping 95% Wilson confidence intervals ([0.0%, 28.3%] vs [5.7%, 51.2%]) and are classified as inconclusive.
- **T5 Statistical Equivalence**: T5 ranker top-1 accuracy (10.0%, [2.8%, 30.1%]) spans the 5.0% random chance baseline and is classified as indistinguishable from chance.
- **Hardware Profile Verified**: Target platform is AMD Ryzen 5 5600H with Radeon Graphics and NVIDIA GeForce GTX 1650 Mobile (3,286 MiB initial free VRAM).
- **Sanitized Links**: All Markdown references use clean relative repository paths without local system paths.
- **Re-validation scope**: Section 6 numbers are from one machine and one model. The earlier `~8.2 ms` and `55.90 ms` restore figures are withdrawn (the first was the re-decode path or unmeasured; the second was a single observation that excluded the K-shift refresh and ran on stale logits).
- **Benchmark method**: `ingest()` returns before the GPU finishes its last chunk; all timings force a sync (logits read). An unsynchronised first run was discarded.
- **Not re-run**: acceptance results T1, T4-T7 and the prefix-reuse results (M6, Section 3) were not re-run against the re-implemented restore path; they describe the earlier implementation.
- **Known external defect**: loading a second model in one process after a long prefill aborts inside llama.cpp b9616's CUDA backend (reproduces with the pre-change DLL). Test files are run one per process.
