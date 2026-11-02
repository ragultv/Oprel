import ctypes
import os
import sys
import time

# Allow OpenMP runtime coexistence on Windows (PyTorch libiomp5 + llama libomp140)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["GGML_CUDA_FORCE_MMQ"] = "1"

from pathlib import Path
from typing import Generator, Optional, Dict, Any

class OprelStats(ctypes.Structure):
    _fields_ = [
        ("total_tokens", ctypes.c_int32),
        ("active_tokens", ctypes.c_int32),
        ("ssd_blocks", ctypes.c_int32),
        ("retrieved_blocks", ctypes.c_int32),
        ("last_retrieval_ms", ctypes.c_double),
        ("last_read_ms", ctypes.c_double),
        ("last_set_data_ms", ctypes.c_double),
        ("read_throughput_mb_s", ctypes.c_double),
        ("decode_calls", ctypes.c_int64),
        ("last_block_bytes", ctypes.c_int64),
        ("total_ssd_bytes_written", ctypes.c_int64),
        ("total_ssd_bytes_read", ctypes.c_int64),
        ("archive_validation_passes", ctypes.c_int64),
        ("archive_validation_failures", ctypes.c_int64),
        ("restore_decode_calls", ctypes.c_int64),
        ("restore_prefill_tokens", ctypes.c_int64),
        ("last_restore_pos_start", ctypes.c_int32),
        ("last_restore_pos_end", ctypes.c_int32),
        ("last_restore_kv_match", ctypes.c_int32),
        ("last_restore_latency_ms", ctypes.c_double),
        ("last_prefill_tokens", ctypes.c_int32),
    ]

class LlamaSSDNativeBackend:
    """
    Modular native C++ wrapper around upstream llama.dll with Win32 Direct I/O
    unbuffered NVMe SSD KV caching and Quest sparse retrieval.
    """
    def __init__(
        self,
        llama_dll_path: Optional[str] = None,
        wrapper_dll_path: Optional[str] = None,
    ):
        if not wrapper_dll_path:
            # Default location where CMake builds the release DLL
            sdk_root = Path(__file__).resolve().parents[3]
            wrapper_dll_path = str(sdk_root / "oprel" / "native" / "build" / "bin" / "Release" / "oprel_llama_ssd.dll")

        if not llama_dll_path:
            # Check standard Oprel cache location with versioning priority
            home = Path.home()
            try:
                from oprel.core.config import Config
                cfg_version = getattr(Config(), "binary_version", "b9616")
            except Exception:
                cfg_version = "b9616"

            candidates = [
                home / ".cache" / "oprel" / "bin" / "llama_cpp" / cfg_version / "cuda" / "llama.dll",
                home / ".cache" / "oprel" / "bin" / "llama_cpp" / "b9616" / "cuda" / "llama.dll",
                home / ".cache" / "oprel" / "bin" / "llama_cpp" / "cuda" / "llama.dll",
                home / ".cache" / "oprel" / "bin" / "cuda" / "llama.dll",
            ]
            for c in candidates:
                if c.exists():
                    llama_dll_path = str(c)
                    break
            if not llama_dll_path:
                llama_dll_path = str(candidates[0])

        if not os.path.exists(wrapper_dll_path):
            raise FileNotFoundError(f"Native wrapper DLL not found at: {wrapper_dll_path}")

        # Add cuda dll directory to PATH so llama.dll can resolve ggml-cuda.dll, cublas, etc.
        llama_dir = os.path.dirname(llama_dll_path)
        if os.path.exists(llama_dir):
            os.environ["PATH"] = llama_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(llama_dir)
                except Exception:
                    pass

        self.dll = ctypes.CDLL(wrapper_dll_path)
        self._setup_signatures()

        # Initialize dynamic linking to the upstream llama.dll
        ok = self.dll.oprel_init(llama_dll_path.encode("utf-8"))
        if not ok:
            raise RuntimeError(f"Failed to bind native wrapper to llama.dll at {llama_dll_path}")

        self.model_handle = None
        self.context_handle = None
        self.last_run: Dict[str, Any] = {}
        self.current_session_id: Optional[str] = None
        self.ssd_cache_dir: Optional[str] = None
        self.retrieval_top_k = 2

    def _setup_signatures(self):
        self.dll.oprel_init.argtypes = [ctypes.c_char_p]
        self.dll.oprel_init.restype = ctypes.c_bool

        self.dll.oprel_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        self.dll.oprel_load_model.restype = ctypes.c_void_p

        self.dll.oprel_create_context.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32]
        self.dll.oprel_create_context.restype = ctypes.c_void_p

        self.dll.oprel_ingest_prompt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        self.dll.oprel_ingest_prompt.restype = ctypes.c_int32

        self.dll.oprel_generate_next_token.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
        self.dll.oprel_generate_next_token.restype = ctypes.c_int32

        if hasattr(self.dll, "oprel_generate_next_token_ex"):
            self.dll.oprel_generate_next_token_ex.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
            ]
            self.dll.oprel_generate_next_token_ex.restype = ctypes.c_int32

        self.dll.oprel_token_to_text.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
        self.dll.oprel_token_to_text.restype = ctypes.c_int32

        self.dll.oprel_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.dll.oprel_is_eog.restype = ctypes.c_bool

        self.dll.oprel_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(OprelStats)]
        self.dll.oprel_get_stats.restype = None

        self.dll.oprel_retrieve_context.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
        self.dll.oprel_retrieve_context.restype = ctypes.c_int32

        self.dll.oprel_restore_kv_state.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.dll.oprel_restore_kv_state.restype = ctypes.c_bool
        if hasattr(self.dll, "oprel_restore_kv_block"):
            self.dll.oprel_restore_kv_block.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_bool]
            self.dll.oprel_restore_kv_block.restype = ctypes.c_bool
        if hasattr(self.dll, "oprel_refresh_logits"):
            self.dll.oprel_refresh_logits.argtypes = [ctypes.c_void_p]
            self.dll.oprel_refresh_logits.restype = ctypes.c_bool
        if hasattr(self.dll, "oprel_archive_oldest"):
            self.dll.oprel_archive_oldest.argtypes = [ctypes.c_void_p, ctypes.c_int32]
            self.dll.oprel_archive_oldest.restype = ctypes.c_int32

        self.dll.oprel_find_best_block.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.dll.oprel_find_best_block.restype = ctypes.c_int32

        self.dll.oprel_score_all_blocks.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32
        ]
        self.dll.oprel_score_all_blocks.restype = ctypes.c_int32

        self.dll.oprel_get_logits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
        self.dll.oprel_get_logits.restype = ctypes.c_int32

        self.dll.oprel_reserialize_range.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32]
        self.dll.oprel_reserialize_range.restype = ctypes.c_int32

        self.dll.oprel_get_block_payload.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32]
        self.dll.oprel_get_block_payload.restype = ctypes.c_int32

        self.dll.oprel_set_session.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.dll.oprel_set_session.restype = ctypes.c_bool
        if hasattr(self.dll, "oprel_save_session"):
            self.dll.oprel_save_session.argtypes = [ctypes.c_void_p]
            self.dll.oprel_save_session.restype = ctypes.c_bool

        if hasattr(self.dll, "oprel_set_autonomous_restore"):
            self.dll.oprel_set_autonomous_restore.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.dll.oprel_set_autonomous_restore.restype = None

        if hasattr(self.dll, "oprel_set_retrieval_top_k"):
            self.dll.oprel_set_retrieval_top_k.argtypes = [ctypes.c_void_p, ctypes.c_int32]
            self.dll.oprel_set_retrieval_top_k.restype = None

        if hasattr(self.dll, "oprel_get_active_restored_blocks"):
            self.dll.oprel_get_active_restored_blocks.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
            self.dll.oprel_get_active_restored_blocks.restype = ctypes.c_int32

        self.dll.oprel_free_context.argtypes = [ctypes.c_void_p]
        self.dll.oprel_free_context.restype = None

        self.dll.oprel_free_model.argtypes = [ctypes.c_void_p]
        self.dll.oprel_free_model.restype = None

    def load_model(self, model_path: str, n_gpu_layers: int = 16):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model_handle = self.dll.oprel_load_model(model_path.encode("utf-8"), n_gpu_layers)
        if not self.model_handle:
            raise RuntimeError(f"Failed to load model through llama.dll: {model_path}")

    def create_context(
        self,
        ssd_cache_dir: Optional[str] = None,
        active_ctx_size: int = 4096,
        max_ssd_context: int = 131072,
        retrieval_top_k: int = 2,
    ):
        if not self.model_handle:
            raise RuntimeError("Must load model before creating context")

        if not ssd_cache_dir:
            ssd_cache_dir = os.path.join(Path.home(), ".cache", "oprel", "ssd_kv")
        os.makedirs(ssd_cache_dir, exist_ok=True)
        self.ssd_cache_dir = ssd_cache_dir
        self.retrieval_top_k = retrieval_top_k

        self.context_handle = self.dll.oprel_create_context(
            self.model_handle,
            ssd_cache_dir.encode("utf-8"),
            active_ctx_size,
            max_ssd_context,
        )
        if not self.context_handle:
            raise RuntimeError("Failed to create SSD-backed context")

        if retrieval_top_k and hasattr(self.dll, "oprel_set_retrieval_top_k"):
            self.dll.oprel_set_retrieval_top_k(self.context_handle, ctypes.c_int32(retrieval_top_k))

    def ingest(self, prompt: str, reset: bool = True) -> int:
        if not self.context_handle:
            raise RuntimeError("Context not initialized")
        n = self.dll.oprel_ingest_prompt(self.context_handle, prompt.encode("utf-8"), reset)
        return int(n)

    def generate_next_token(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.15,
        frequency_penalty: float = 0.35,
        presence_penalty: float = 0.30,
    ) -> int:
        if not self.context_handle:
            raise RuntimeError("Context not initialized")
        if hasattr(self.dll, "oprel_generate_next_token_ex"):
            return int(self.dll.oprel_generate_next_token_ex(
                self.context_handle,
                ctypes.c_float(temperature),
                ctypes.c_float(top_p),
                ctypes.c_float(repeat_penalty),
                ctypes.c_float(frequency_penalty),
                ctypes.c_float(presence_penalty),
            ))
        return int(self.dll.oprel_generate_next_token(self.context_handle, temperature, top_p))

    def token_to_text(self, token: int) -> str:
        if not self.context_handle:
            return ""
        buf = ctypes.create_string_buffer(512)
        length = self.dll.oprel_token_to_text(self.context_handle, token, buf, 512)
        if length > 0:
            return buf.value[:length].decode("utf-8", errors="replace")
        return ""

    def is_eog(self, token: int) -> bool:
        if not self.context_handle:
            return True
        return bool(self.dll.oprel_is_eog(self.context_handle, token))

    def get_stats(self) -> Dict[str, Any]:
        if not self.context_handle:
            return {}
        s = OprelStats()
        self.dll.oprel_get_stats(self.context_handle, ctypes.byref(s))
        return {
            "total_tokens": s.total_tokens,
            "active_tokens": s.active_tokens,
            "ssd_blocks": s.ssd_blocks,
            "retrieved_blocks": s.retrieved_blocks,
            "active_restored_blocks": self.get_active_restored_blocks(),
            "last_retrieval_ms": round(s.last_retrieval_ms, 2),
            "last_read_ms": round(s.last_read_ms, 2),
            "last_set_data_ms": round(s.last_set_data_ms, 2),
            "read_throughput_mb_s": round(s.read_throughput_mb_s, 2),
            "decode_calls": s.decode_calls,
            "last_block_bytes": s.last_block_bytes,
            "total_ssd_bytes_written": s.total_ssd_bytes_written,
            "total_ssd_bytes_read": s.total_ssd_bytes_read,
            "archive_validation_passes": s.archive_validation_passes,
            "archive_validation_failures": s.archive_validation_failures,
            "restore_decode_calls": s.restore_decode_calls,
            "restore_prefill_tokens": s.restore_prefill_tokens,
            "last_restore_pos_start": s.last_restore_pos_start,
            "last_restore_pos_end": s.last_restore_pos_end,
            "last_restore_kv_match": s.last_restore_kv_match,
            "last_restore_latency_ms": round(s.last_restore_latency_ms, 2),
            "last_prefill_tokens": s.last_prefill_tokens,
        }

    def restore_kv_state(self, block_index: int, refresh_logits: bool = True) -> bool:
        """
        Restore an archived KV block from SSD (Direct I/O + llama_state_seq_set_data_ext). The block's
        tokens are never decoded. With ``refresh_logits`` (default) one single-token decode of the
        last live token applies the lazy K-shift and makes the next sample attend to the block;
        pass False only when a decode follows before sampling.
        """
        if not self.context_handle:
            return False
        if not refresh_logits and hasattr(self.dll, "oprel_restore_kv_block"):
            return bool(self.dll.oprel_restore_kv_block(self.context_handle, ctypes.c_int32(block_index), False))
        return bool(self.dll.oprel_restore_kv_state(self.context_handle, ctypes.c_int32(block_index)))

    def refresh_logits(self) -> bool:
        """Re-decode the last live token against the current cache (also applies a pending K-shift)."""
        if not self.context_handle or not hasattr(self.dll, "oprel_refresh_logits"):
            return False
        return bool(self.dll.oprel_refresh_logits(self.context_handle))

    def archive_oldest(self, n_tokens: int) -> int:
        """
        Archive the oldest ``n_tokens`` live tokens (after the attention sinks) to SSD now.
        Returns the new block index, or -1 if nothing was archived.
        """
        if not self.context_handle or not hasattr(self.dll, "oprel_archive_oldest"):
            return -1
        return int(self.dll.oprel_archive_oldest(self.context_handle, ctypes.c_int32(n_tokens)))

    def set_session(self, session_id: str) -> bool:
        """
        Switch active conversation session with zero cold re-prefill.
        Persists previous session and loads requested session in ~15 ms.
        """
        if not self.context_handle or not session_id:
            return False
        ok = bool(self.dll.oprel_set_session(self.context_handle, session_id.encode("utf-8")))
        if ok:
            self.current_session_id = session_id
        return ok

    def save_session(self) -> bool:
        """
        Persist active session state (live KV and SSD block index) to disk.
        """
        if not self.context_handle or not hasattr(self.dll, "oprel_save_session"):
            return False
        return bool(self.dll.oprel_save_session(self.context_handle))

    def set_autonomous_restore(self, enabled: bool):
        """
        Enable or disable autonomous query-time SSD block restoration.
        """
        if self.context_handle and hasattr(self.dll, "oprel_set_autonomous_restore"):
            self.dll.oprel_set_autonomous_restore(self.context_handle, ctypes.c_bool(enabled))

    def set_retrieval_top_k(self, top_k: int = 2):
        """
        Set number of top-K historical blocks autonomously restored from SSD.
        """
        if self.context_handle and hasattr(self.dll, "oprel_set_retrieval_top_k"):
            self.dll.oprel_set_retrieval_top_k(self.context_handle, ctypes.c_int32(top_k))

    def get_active_restored_blocks(self, max_blocks: int = 8) -> list:
        """
        Get list of block indices currently active in sequence 0.
        """
        if not self.context_handle or not hasattr(self.dll, "oprel_get_active_restored_blocks"):
            return []
        arr = (ctypes.c_int32 * max_blocks)()
        n = self.dll.oprel_get_active_restored_blocks(self.context_handle, arr, ctypes.c_int32(max_blocks))
        return list(arr[:n])

    def find_best_block(self, query: str) -> int:
        """
        Query Quest sparse index to find the best matching SSD block index.
        """
        if not self.context_handle:
            return -1
        return int(self.dll.oprel_find_best_block(self.context_handle, query.encode("utf-8")))

    def score_all_blocks(self, query: str, max_blocks: int = 128) -> list:
        """
        Score all archived blocks against query and return list of dicts:
        [{'block_index': int, 'score': float, 'p0': int, 'p1': int}, ...]
        """
        if not self.context_handle:
            return []
        blk_arr = (ctypes.c_int32 * max_blocks)()
        scores_arr = (ctypes.c_float * max_blocks)()
        start_arr = (ctypes.c_int32 * max_blocks)()
        ntok_arr = (ctypes.c_int32 * max_blocks)()

        n = self.dll.oprel_score_all_blocks(
            self.context_handle,
            query.encode("utf-8"),
            blk_arr,
            scores_arr,
            start_arr,
            ntok_arr,
            ctypes.c_int32(max_blocks)
        )
        res = []
        for i in range(n):
            res.append({
                "block_index": blk_arr[i],
                "score": round(float(scores_arr[i]), 4),
                "p0": start_arr[i],
                "p1": start_arr[i] + ntok_arr[i],
                "n_tokens": ntok_arr[i],
            })
        return res

    def retrieve_context(self, query: str, top_k: int = 1, max_chars: int = 4096) -> str:
        if not self.context_handle:
            return ""
        buf = ctypes.create_string_buffer(max_chars)
        length = self.dll.oprel_retrieve_context(
            self.context_handle,
            query.encode("utf-8"),
            ctypes.c_int32(top_k),
            buf,
            ctypes.c_int32(max_chars),
        )
        if length > 0:
            return buf.value[:length].decode("utf-8", errors="replace")
        return ""

    def get_logits(self, max_vocab: int = 262208) -> list:
        if not self.context_handle:
            return []
        buf = (ctypes.c_float * max_vocab)()
        n = self.dll.oprel_get_logits(self.context_handle, buf, ctypes.c_int32(max_vocab))
        return list(buf[:n])

    def reserialize_range(self, p0: int, p1: int, max_bytes: int = 64 * 1024 * 1024) -> bytes:
        if not self.context_handle:
            return b""
        buf = (ctypes.c_uint8 * max_bytes)()
        n = self.dll.oprel_reserialize_range(self.context_handle, ctypes.c_int32(p0), ctypes.c_int32(p1), buf, ctypes.c_int32(max_bytes))
        if n > 0:
            return bytes(buf[:n])
        return b""

    def get_block_payload(self, block_index: int, max_bytes: int = 64 * 1024 * 1024) -> bytes:
        if not self.context_handle:
            return b""
        buf = (ctypes.c_uint8 * max_bytes)()
        n = self.dll.oprel_get_block_payload(self.context_handle, ctypes.c_int32(block_index), buf, ctypes.c_int32(max_bytes))
        if n > 0:
            return bytes(buf[:n])
        return b""

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.15,
        frequency_penalty: float = 0.35,
        presence_penalty: float = 0.30,
        ignore_eos: bool = False,
    ) -> Generator[str, None, None]:
        limit = max_new_tokens if max_new_tokens and max_new_tokens > 0 else 4096
        t_start = time.perf_counter()
        prompt_tokens = self.ingest(prompt, reset=True) if prompt else 0
        prefill_tokens = self.get_stats().get("last_prefill_tokens", prompt_tokens) if prompt else 0
        completion_tokens = 0
        t_first = None
        # Timings of the last run, read by the server for the Dev tab speed metrics.
        # ingest returns before the GPU finishes the last prompt chunk (CUDA is asynchronous), so the
        # first sampled token absorbs that work: prompt time runs until the first token, and decode
        # speed is measured over the tokens after it.
        self.last_run = {
            "prompt_tokens": prompt_tokens,
            "prefill_tokens": prefill_tokens,
            "prompt_s": 0.0,
            "completion_tokens": 0,
            "decode_tokens": 0,
            "decode_s": 0.0,
        }
        try:
            for _ in range(limit):
                token = self.generate_next_token(
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
                if t_first is None:
                    t_first = time.perf_counter()
                if token < 0 or (not ignore_eos and self.is_eog(token)):
                    break
                completion_tokens += 1
                piece = self.token_to_text(token)
                yield piece
        finally:
            t_end = time.perf_counter()
            self.last_run["prompt_s"] = (t_first or t_end) - t_start
            self.last_run["completion_tokens"] = completion_tokens
            self.last_run["decode_tokens"] = max(0, completion_tokens - 1)
            self.last_run["decode_s"] = (t_end - t_first) if t_first else 0.0

    def close(self):
        if self.context_handle:
            self.dll.oprel_free_context(self.context_handle)
            self.context_handle = None
        if self.model_handle:
            self.dll.oprel_free_model(self.model_handle)
            self.model_handle = None

    def __del__(self):
        self.close()
