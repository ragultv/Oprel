# Native SSD KV archive: validation, measurements and limitations

Measured 2026-11-01 21:05:14 on the machine below. Everything in the tables comes from `ssd_kv_benchmark.json` and `ssd_kv_parity.json` in this folder; the scripts that produced them are `scripts/bench_ssd_kv.py`, `tests/unit/test_ssd_parity.py` and `tests/unit/test_ssd_hardening.py`.

## 1. What was wrong, and what changed

| # | Defect | Fix |
|---|---|---|
| 1 | The restore path in the shipped DLL re-decoded `history_tokens` at new positions (not a KV restore). The unstaged rewrite that used `llama_state_seq_set_data_ext` had never been built. | `restore_block` / `internal_load_and_inject_block` now read the block from SSD, validate it, inject the serialised KV, and never decode the block's tokens. |
| 2 | The `llama_context_params` mirror in `llama_dyn.h` did not match llama.cpp b9616 (two extra fields, one missing). Only `n_ctx`/`type_k`/`type_v` happened to line up, so the scratch sequence could not be configured. The first real restore failed with `invalid seq_id-agnostic kv cell`. | Struct corrected field-for-field; `static_assert`s pin the offsets and size. `n_seq_max = 2`, `kv_unified = true`. |
| 3 | llama.cpp applies K-shifts lazily (next `llama_decode`); `state_seq_get_data` does **not**. Blocks were serialised after a pending `seq_add`, so their K vectors were rotated for the wrong positions. | Blocks are serialised at their live positions after any pending shift is applied (`apply_pending_shift`). Restore re-positions with `seq_add` and the next decode applies the rotation. Block format v3. |
| 4 | Restored cells sat at the front of the window and were archived again by the next eviction, under the wrong global positions. | `drop_all_restored_blocks` runs before every archive; live positions of the other restored blocks are re-tracked. |
| 5 | Eviction removed cells before the payload was written; any failure lost tokens and broke position bookkeeping. | Transactional eviction with rollback; write failures mark the block unrestorable. |
| 6 | Session files were written in place, unchecked on read, and lost the restored-block list; stats were stale after a session switch. | Atomic temp-file + fsync + rename, CRC32, bounds-checked load that only mutates state after validation; restored blocks and stats persisted/synced. Session format v6. |
| 7 | Session IDs became file names unsanitised; the block file and session count were unbounded; `n_layer == 34` was hardcoded. | Sanitised IDs (collision-safe), `OPREL_SSD_MAX_BYTES` (default 16 GiB), `OPREL_SSD_MAX_SESSIONS` (default 64), real layer count, `OPREL_SSD_KV_TYPE` (`q4_0` default, `q8_0`, `f16`). |
| 8 | No locking; `direct_io` ignored partial/short transfers. | Per-context recursive mutex; looped reads/writes. |

New native entry points: `oprel_archive_oldest`, `oprel_refresh_logits`, `oprel_restore_kv_block(…, refresh_logits)`.

## 2. Trace of the real path

Archive (`oprel_ingest_prompt` / `oprel_generate_next_token_ex` → `archive_and_evict_kv`, `llama_ssd_wrapper.cpp`):

1. `drop_all_restored_blocks` then `apply_pending_shift` (one single-token decode if a K-shift is pending).
2. `llama_memory_seq_cp(0→1)`, `llama_memory_seq_rm(0, range)`, `llama_state_seq_get_size_ext` / `get_data_ext` (seq 1).
3. Header (magic, format v3, llama build, model SHA-256, runtime DLL SHA-256, layers, KV types, positions, byte count, CRC32) + payload, sector-aligned, staged in a shadow buffer.
4. `std::async` → `direct_io_write` (`direct_io.cpp`, Direct I/O, write-through). `join_bg_write` marks the block unrestorable if the write failed.

Restore (`oprel_restore_kv_state` → `restore_block` → `internal_load_and_inject_block`):

1. `join_bg_write`, bounds check against the file size.
2. `direct_io_read` of the exact bytes written.
3. Validate magic, version, build, model SHA, runtime SHA, layer count, KV types, positions, byte count and CRC32. Any mismatch: reject, count `archive_validation_failures`, fall back to uncached inference.
4. `llama_state_seq_set_data_ext` into scratch seq 1; verify the restored position range.
5. `seq_add` re-positions the cells and opens a gap in seq 0; `seq_cp(1→0)`; `seq_rm(1)`.
6. `apply_pending_shift`: one single-token decode of the last live token applies the lazy K-shift and refreshes the logits. The block's own tokens are never decoded.
7. The next `generate_next_token_ex` samples from those logits and decodes against restored prefix + live tail.

## 3. Cached vs uncached parity

**Method.** Reference: one context prefills the prompt (~760 tokens, inside Gemma-3's 1024-token sliding window so every block holds the KV of *all* layers) and never evicts. Cached: the same prompt, oldest tokens archived with `archive_oldest`, then restored. Same model, prompt, KV type and greedy decode settings. The prompt ends so that the next token is the first digit of a code stored in the archived block. When every archived block is restored each token is back at its original position, so logits are compared at the same logical position. Parity *after eviction without restore* (compacted positions) is intentionally not compared; it appears only as the "block absent" control.

**Noise floor.** Re-decoding the last token alone against an unchanged cache (batch prefill vs single-token decode) already differs slightly; the restore error is judged against that.

**Control.** "Block absent" = the last token re-decoded with the archived block missing (the tail's own KV was computed with the block present, so this is a lower bound on the damage of not restoring).

| Case | KV | KL(ref‖cached) | top-20 max abs Δlogit | max abs Δlogit | cosine | control KL |
|---|---|---|---|---|---|---|
| noise floor | f16 | 1.6e-07 | 0.31 | 0.44 | 0.99972 | - |
| single block | f16 | 0.0e+00 | 0.27 | 0.46 | 0.99971 | 1.9e-04 |
| later block, in order | f16 | 2.0e-07 | 0.19 | 0.48 | 0.99969 | 7.7e+00 |
| later block, reversed | f16 | 1.5e-06 | 0.24 | 0.49 | 0.99965 | 7.7e+00 |
| re-archive restored range | f16 | 4.5e-07 | 0.26 | 0.46 | 0.99969 | - |
| noise floor | q4_0 | 4.5e-06 | 0.30 | 0.68 | 0.99931 | - |
| single block | q4_0 | 1.5e-04 | 1.40 | 2.08 | 0.99700 | 3.1e-03 |
| later block, in order | q4_0 | 3.7e-05 | 1.54 | 1.82 | 0.99482 | 8.2e+00 |
| later block, reversed | q4_0 | 3.7e-05 | 1.54 | 1.82 | 0.99482 | 8.2e+00 |
| re-archive restored range | q4_0 | 8.0e-05 | 2.39 | 2.39 | 0.99417 | - |

All cases: argmax identical, top-10 tokens identical, and the 16-token greedy continuation identical to the uncached run (asserted in the tests).

**Tolerances used by the tests** (`TOLERANCE` in `test_ssd_parity.py`, 2-7x headroom over the measurements): f16 KL <= 1e-5 and top-20 |Δlogit| <= 0.75; q4_0 KL <= 1e-3 and top-20 |Δlogit| <= 4.0; control KL must be at least 10x the restored error.

**Reading the numbers.** With f16 KV the restored logits are indistinguishable from the noise floor. With q4_0 the error is 10-100x the floor because the lazy K-shift de-quantises, rotates and re-quantises Q4_0 K cells (the tail is shifted twice, restored blocks once); it stays about five orders of magnitude below the "block absent" control (KL ~8). The "later block" and "reversed order" cases are the K-rotation proof: block 1 is stored at live positions [4, 204) and restored 200 positions to the right, so a missing or wrong re-rotation would show up as the control-sized error, not as noise-sized error.

**Position coverage** (`test_ssd_parity.py`): a later block archived after the first block's tail shift and restored in both orders; re-archiving while a restored block is live (restored cells are dropped first, the new block's global start is 204 = original token index, and live-token accounting matches); block bytes round-trip bit-identical for the no-shift case (`test_ssd_hardening.py`).

## 4. Benchmarks

**Hardware:** AMD Ryzen 5 5600H with Radeon Graphics (12 logical cores), 15.3 GB RAM, NVIDIA GeForce GTX 1650 (4096 MiB, driver 592.82), cache drive C: MTFDKBA512TFH-1BC1AABHA SSD NVMe, Windows-11-10.0.26200-SP0, Python 3.12.6.
**Model:** gemma-3-4b-it-Q5_K_M.gguf (2.64 GB), 33 GPU layers, llama.cpp b9616 (CUDA), window 4096 (cache 5120 cells), 7 repetitions per cell.

**Method.**

- *prefill*: ingest() of a salted prompt (no prefix reuse), reset=True, plus a logits read that forces the GPU sync, wall clock
- *archive*: archive_oldest(n): serialise + stage; the disk write runs in the background
- *restore*: restore_kv_state(): Direct I/O read + header/CRC validation + state_seq_set_data_ext + re-positioning + one single-token refresh decode; measured 0.4 s after the archive so the asynchronous write has finished. Direct I/O bypasses the OS page cache; the drive's own cache may still serve recent writes.
- *generate*: per generate_next_token(temperature=0.0) call (sample + 1 decode), greedy

Cold prefill includes a forced GPU sync (logits read). `ingest()` alone returns before the GPU finishes the last chunk; an earlier run that did not sync was discarded because it under-counted prefill and inflated the next call (see section 6).


### KV type q4_0

**Cold prefill** (nothing cached)

| prompt tokens | p50 ms | p95 ms | min | max | stdev | tokens/s | decode calls |
|---|---|---|---|---|---|---|---|
| 319 | 1,879.7 | 1,885.7 | 1,877.1 | 1,885.7 | 2.4 | 169.7 | 1 |
| 621 | 3,505.9 | 3,512.5 | 3,501.1 | 3,512.5 | 4.0 | 177.1 | 2 |
| 925 | 5,413.0 | 5,421.5 | 5,405.9 | 5,421.5 | 5.2 | 170.9 | 2 |

**Archive and warm restore** (prompt ~925 tokens)

| block tokens | cache bytes read | SSD read+validate p50 | set_data p50 | restore wall p50 | p95 | min | max | decode calls | prefill tokens | archive call p50 | re-prefill estimate | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 5,017,600 (4.78 MiB) | 1.9 ms | 3.5 ms | 664.3 ms | 2,011.1 | 630.7 | 2,011.1 | 1 | 0 | 25.6 ms | 749 ms | 1.1x |
| 256 | 10,035,200 (9.57 MiB) | 6.2 ms | 5.7 ms | 733.3 ms | 744.1 | 718.3 | 744.1 | 1 | 0 | 55.1 ms | 1,498 ms | 2.0x |
| 512 | 20,070,400 (19.14 MiB) | 15.0 ms | 8.7 ms | 800.4 ms | 824.1 | 781.7 | 824.1 | 1 | 0 | 105.8 ms | 2,996 ms | 3.7x |

Re-prefill estimate = block tokens / measured prefill throughput (170.9 tok/s, median of the three prompt sizes); it is an estimate, not a separate measurement. Restore wall time is dominated by the refresh decode (lazy K-shift), not by SSD I/O.

**Next-token generation** (greedy, 32 tokens per run, per-call latency = sample + 1 decode)

| scenario | first token p50 / p95 ms | steady-state p50 / p95 / max ms | decode calls per run |
|---|---|---|---|
| no restore | 46.8 / 89.0 | 44.0 / 52.1 / 60.1 | 32 |
| after restore | 47.3 / 52.5 | 44.5 / 53.0 / 108.9 | 32 |

### KV type f16

**Cold prefill** (nothing cached)

| prompt tokens | p50 ms | p95 ms | min | max | stdev | tokens/s | decode calls |
|---|---|---|---|---|---|---|---|
| 318 | 1,916.9 | 1,939.7 | 1,912.4 | 1,939.7 | 9.6 | 165.9 | 1 |
| 620 | 3,530.1 | 3,556.0 | 3,529.6 | 3,556.0 | 10.6 | 175.6 | 2 |
| 925 | 5,448.9 | 5,469.5 | 5,430.3 | 5,469.5 | 14.5 | 169.8 | 2 |

**Archive and warm restore** (prompt ~925 tokens)

| block tokens | cache bytes read | SSD read+validate p50 | set_data p50 | restore wall p50 | p95 | min | max | decode calls | prefill tokens | archive call p50 | re-prefill estimate | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 17,829,888 (17.0 MiB) | 13.8 ms | 6.6 ms | 203.8 ms | 336.3 | 174.5 | 336.3 | 1 | 0 | 98.7 ms | 754 ms | 3.7x |
| 256 | 35,659,776 (34.01 MiB) | 24.1 ms | 12.0 ms | 302.0 ms | 304.8 | 260.7 | 304.8 | 1 | 0 | 179.7 ms | 1,508 ms | 5.0x |
| 512 | 71,319,552 (68.01 MiB) | 44.6 ms | 21.5 ms | 494.5 ms | 521.6 | 464.3 | 521.6 | 1 | 0 | 348.9 ms | 3,015 ms | 6.1x |

Re-prefill estimate = block tokens / measured prefill throughput (169.8 tok/s, median of the three prompt sizes); it is an estimate, not a separate measurement. Restore wall time is dominated by the refresh decode (lazy K-shift), not by SSD I/O.

**Next-token generation** (greedy, 32 tokens per run, per-call latency = sample + 1 decode)

| scenario | first token p50 / p95 ms | steady-state p50 / p95 / max ms | decode calls per run |
|---|---|---|---|
| no restore | 48.7 / 68.7 | 45.3 / 57.4 / 66.1 | 32 |
| after restore | 40.8 / 56.1 | 45.9 / 53.2 / 66.5 | 32 |

**Observations.**
- A warm restore never decodes the block's tokens (prefill tokens 0) and costs exactly one decode call. Bytes read per restore equal the stored payload.
- SSD read + CRC validation + `set_data` is about 5-70 ms for 5-71 MB. The rest of the restore is the K-shift refresh decode: about 0.6-0.8 s for q4_0 (re-quantising Q4_0 K cells across the whole 5120-cell cache) and 0.2-0.45 s for f16.
- Against re-prefilling on this GPU (~170 tok/s) a restore is roughly break-even at 128 tokens with q4_0 and 2-6x faster for larger blocks; the gain grows with block size because restore cost is almost flat.
- Steady-state generation after a restore is the same speed as without one; there is no per-token penalty from restored cells.
- One q4_0 restore at 128 tokens took ~2 s against a ~0.65 s median. It was probably the first restore of the run (warm-up), but I did not verify that. It is the only outlier and is included in the p95/max.

## 5. Test results

Run on the rebuilt DLL after restarting the `oprel start` server, each file in its own process:

| File | Result |
|---|---|
| `tests/unit/test_ssd_hardening.py` | 11 passed |
| `tests/unit/test_ssd_parity.py` | 8 passed |
| `tests/unit/test_session_persistence.py` | 1 passed |
| `tests/unit/test_native_wrapper.py` | 2 passed |

Covered: cache miss/hit, write+read, successful restore with decode-count evidence, bit-identical round trip, corrupt block, truncated block file, corrupt/truncated/tiny session files, incompatible metadata (model/layers/types/version), disk cap (drop instead of write), reset reclaiming the block file, session pruning, ID traversal, concurrent access, fallback to uncached inference after every rejection, parity and position cases above.

## 6. Limitations and things not to over-claim

1. **Second model in one process can abort.** Loading a second model in the same process after a long prefill (~3900 tokens) aborts inside llama.cpp's CUDA backend (`MUL_MAT failed: CUDA error: invalid argument`, `ggml-cuda.cu:3163`). It reproduces with the pre-change DLL, so it is not caused by these changes, and it needs no restore or eviction (a plain long ingest is enough). Cause not identified. I did not verify whether the `oprel start` server hits it when it switches models; it loaded a first model fine after the restart. The test files are therefore run one per process.
2. **Sliding-window models.** Gemma-3's SWA layers keep KV only for the last 1024 positions, so a block archived after its tokens left that window holds only the global layers. Restore returns exactly what the cache held. The parity tests stay inside the window. With a live window larger than 1024 this is harmless while the restored block is farther than 1024 positions from the query (the SWA layers mask it anyway), but it is not guaranteed if the window later shrinks.
3. **Restore latency is dominated by the K-shift, not the SSD** (section 4). It is inherent to llama.cpp's lazy shift (no public "apply updates" call), and is paid once per restore/eviction event. `restore_kv_state(…, refresh_logits=False)` skips the refresh decode, but then the logits still describe the cache without the block: only use it when a decode follows before sampling.
4. **Q4_0 KV adds requantisation error** when cells are shifted (section 3). f16 does not measurably.
5. **Layout after eviction is compacted** (sinks | restored blocks | tail). Restoring every archived block reproduces the original positions only for tokens that were contiguous in the original order; partial restores are the StreamingLLM-style approximation the design intends and are not compared against an uncached run.
6. **Restored blocks are dropped at the next eviction** (never re-archived); retrieval benefit does not persist across window overflows.
7. **Block file cap drops the newest range, not the oldest.** The file is append-only; when the cap is hit the new range is recorded as dropped and is not restorable.
8. **Direct I/O bypasses the OS page cache, but the drive's own cache may serve recent writes**, so "cold from NAND" latency was not measured.
9. **Single GPU, single model, one prompt family.** The benchmark and parity numbers are for this GTX 1650 / Gemma-3-4B Q5_K_M setup, not a general claim. `llama_build` (9616) and the `llama_context_params` layout are pinned to b9616.
10. `get_stats` is unlocked, and `oprel_free_context` must not run concurrently with other calls on the same context.

**Retractions.** The earlier "8.20 ms SSD retrieval / zero decode" and "55.90 ms restore" figures are withdrawn: the first was the re-decode path or an unmeasured claim, the second was a single observation that excluded the K-shift refresh and ran against stale logits. A short-lived claim that GPU-to-host serialisation runs at ~26 MB/s was also wrong: it was the serialise call waiting for outstanding GPU work after an unsynchronised ingest. With a proper sync, archiving a 512-token q4_0 block takes about 106 ms.
