"""Benchmark the native SSD KV archive: cold prefill, warm restore, next-token generation.

Each phase is measured separately, with repeated runs and latency distributions:

  prefill   ingest of a fresh prompt (nothing cached), per prompt size
  archive   archive_oldest(n): serialise + stage the block (the disk write is asynchronous)
  restore   restore_kv_state(block): SSD read + validate + inject + 1 refresh decode, per block size
  generate  per-token latency of greedy generation, with and without a restored block

Usage:  python scripts/bench_ssd_kv.py [--reps 7] [--out docs/dev-guide/ssd_kv_benchmark.json]
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend  # noqa: E402

MODEL_PATH = r"C:\Users\tragu\.cache\oprel\models\models--unsloth--gemma-3-4b-it-GGUF\snapshots\5a3566e716d80f709ed7b79817eaf7733d2a1fce\gemma-3-4b-it-Q5_K_M.gguf"
N_GPU_LAYERS = 33
PROMPT_RECORDS = [16, 32, 48]          # ~300 / ~600 / ~900 tokens, all inside the SWA window
BLOCK_SIZES = [128, 256, 512]
GEN_TOKENS = 32


def prompt(n_records: int, salt: str) -> str:
    # The salt differs in the first tokens so a repeated ingest never reuses the cached prefix.
    body = " ".join(f"Record {i}: the vault code for room {i} is {(i * 37) % 1000}." for i in range(n_records))
    return (f"<start_of_turn>user\nRun {salt}. {body}\nWhat is the vault code for room 7?<end_of_turn>\n"
            f"<start_of_turn>model\nThe vault code for room 7 is ")


def dist(samples_ms):
    xs = sorted(samples_ms)
    n = len(xs)

    def pct(p):
        return xs[min(n - 1, int(round(p * (n - 1))))]

    return {
        "n": n,
        "min_ms": round(xs[0], 3),
        "p50_ms": round(pct(0.5), 3),
        "p95_ms": round(pct(0.95), 3),
        "max_ms": round(xs[-1], 3),
        "mean_ms": round(statistics.fmean(xs), 3),
        "stdev_ms": round(statistics.pstdev(xs), 3),
    }


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, shell=False).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable ({e})"


def hardware(cache_dir: str) -> dict:
    ps = lambda q: sh(["powershell", "-NoProfile", "-Command", q])
    drive = os.path.splitdrive(os.path.abspath(cache_dir))[0].rstrip(":")
    return {
        "os": platform.platform(),
        "cpu": ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"),
        "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "cache_drive": f"{drive}: " + ps(
            f"$p = Get-Partition -DriveLetter {drive} -ErrorAction SilentlyContinue; "
            "if ($p) { (Get-PhysicalDisk | Where-Object DeviceId -eq $p.DiskNumber | "
            "Select-Object -First 1 | ForEach-Object { $_.FriendlyName + ' ' + $_.MediaType + ' ' + $_.BusType }) }"),
        "python": platform.python_version(),
    }


class Bench:
    def __init__(self, backend, kv_type, reps):
        self.b = backend
        self.kv = kv_type
        self.reps = reps
        self.tmp = tempfile.mkdtemp(prefix=f"oprel_bench_{kv_type}_")
        os.environ["OPREL_SSD_KV_TYPE"] = kv_type
        if backend.context_handle:
            backend.dll.oprel_free_context(backend.context_handle)
            backend.context_handle = None
        backend.create_context(ssd_cache_dir=self.tmp, active_ctx_size=4096, max_ssd_context=32768)
        backend.set_autonomous_restore(False)
        assert backend.set_session(f"bench-{kv_type}")

    def ingest(self, n_records):
        t0 = time.perf_counter()
        n = self.b.ingest(prompt(n_records, uuid.uuid4().hex[:8]), reset=True)
        # ingest() returns before the GPU has finished the last chunk (CUDA is asynchronous and nothing
        # has read the logits yet). Reading them forces the sync, so the timing covers the whole prefill.
        self.b.get_logits(max_vocab=8)
        return n, (time.perf_counter() - t0) * 1000.0

    def prefill(self):
        out = {}
        self.ingest(8)  # warm-up (graph reservation, allocator)
        for rec in PROMPT_RECORDS:
            times, toks, dec = [], 0, []
            for _ in range(self.reps):
                before = self.b.get_stats()["decode_calls"]
                toks, ms = self.ingest(rec)
                times.append(ms)
                dec.append(self.b.get_stats()["decode_calls"] - before)
            d = dist(times)
            d.update(prompt_tokens=toks, decode_calls_per_run=dec[0],
                     tokens_per_s=round(toks / (d["p50_ms"] / 1000.0), 1))
            out[str(toks)] = d
        return out

    def archive_restore(self):
        out = {}
        for n_block in BLOCK_SIZES:
            arch, rest, read, setd, whole, b_read, d_calls, pref, payload = [], [], [], [], [], [], [], [], 0
            for _ in range(self.reps):
                n_tok, _ = self.ingest(PROMPT_RECORDS[-1])
                t0 = time.perf_counter()
                blk = self.b.archive_oldest(n_block)
                arch.append((time.perf_counter() - t0) * 1000.0)
                assert blk >= 0, "archive failed"
                payload = self.b.get_stats()["last_block_bytes"]
                time.sleep(0.4)  # let the asynchronous write finish so restore does not wait on it
                before = self.b.get_stats()
                t0 = time.perf_counter()
                ok = self.b.restore_kv_state(blk)
                rest.append((time.perf_counter() - t0) * 1000.0)
                assert ok, "restore failed"
                after = self.b.get_stats()
                read.append(after["last_read_ms"])
                setd.append(after["last_set_data_ms"])
                whole.append(after["last_restore_latency_ms"])
                b_read.append(after["total_ssd_bytes_read"] - before["total_ssd_bytes_read"])
                d_calls.append(after["decode_calls"] - before["decode_calls"])
                pref.append(after["restore_prefill_tokens"])
            out[str(n_block)] = {
                "prompt_tokens": n_tok,
                "block_tokens": n_block,
                "payload_bytes": payload,
                "payload_mib": round(payload / 2**20, 2),
                "archive_call": dist(arch),
                "restore_wall": dist(rest),
                "restore_native_total": dist(whole),
                "restore_ssd_read_and_validate": dist(read),
                "restore_set_data": dist(setd),
                "bytes_read_per_restore": sorted(set(b_read)),
                "decode_calls_per_restore": sorted(set(d_calls)),
                "prefill_tokens_per_restore": sorted(set(pref)),
            }
        return out

    def generate(self):
        def run(restore):
            first, steady, dec = [], [], []
            for _ in range(self.reps):
                self.ingest(PROMPT_RECORDS[1])
                if restore:
                    blk = self.b.archive_oldest(252)
                    assert blk >= 0 and self.b.restore_kv_state(blk)
                before = self.b.get_stats()["decode_calls"]
                for i in range(GEN_TOKENS):
                    t0 = time.perf_counter()
                    tok = self.b.generate_next_token(temperature=0.0)
                    ms = (time.perf_counter() - t0) * 1000.0
                    assert tok >= 0
                    (first if i == 0 else steady).append(ms)
                dec.append(self.b.get_stats()["decode_calls"] - before)
            return {"first_token": dist(first), "steady_state_token": dist(steady),
                    "decode_calls_per_run": sorted(set(dec)), "tokens_per_run": GEN_TOKENS}

        return {"no_restore": run(False), "after_restore": run(True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "docs" / "dev-guide" / "ssd_kv_benchmark.json"))
    ap.add_argument("--kv", nargs="+", default=["q4_0", "f16"])
    args = ap.parse_args()

    backend = LlamaSSDNativeBackend()
    backend.load_model(MODEL_PATH, n_gpu_layers=N_GPU_LAYERS)
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {"file": os.path.basename(MODEL_PATH), "size_gb": round(os.path.getsize(MODEL_PATH) / 2**30, 2),
                  "n_gpu_layers": N_GPU_LAYERS},
        "llama_cpp_build": "b9616 (CUDA)",
        "reps": args.reps,
        "method": {
            "prefill": "ingest() of a salted prompt (no prefix reuse), reset=True, plus a logits read that forces the GPU sync, wall clock",
            "archive": "archive_oldest(n): serialise + stage; the disk write runs in the background",
            "restore": "restore_kv_state(): Direct I/O read + header/CRC validation + state_seq_set_data_ext + "
                       "re-positioning + one single-token refresh decode; measured 0.4 s after the archive so the "
                       "asynchronous write has finished. Direct I/O bypasses the OS page cache; the drive's own "
                       "cache may still serve recent writes.",
            "generate": "per generate_next_token(temperature=0.0) call (sample + 1 decode), greedy",
        },
        "hardware": None,
        "results": {},
    }
    for kv in args.kv:
        bench = Bench(backend, kv, args.reps)
        if result["hardware"] is None:
            result["hardware"] = hardware(bench.tmp)
        print(f"\n=== KV type {kv} ===", flush=True)
        result["results"][kv] = {
            "cold_prefill": bench.prefill(),
            "archive_and_restore": bench.archive_restore(),
            "generation": bench.generate(),
        }
        r = result["results"][kv]
        print(json.dumps({"prefill_p50_ms": {k: v["p50_ms"] for k, v in r["cold_prefill"].items()},
                          "restore_p50_ms": {k: v["restore_wall"]["p50_ms"] for k, v in r["archive_and_restore"].items()}}),
              flush=True)
    backend.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
