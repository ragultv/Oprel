"""
Global configuration management

Production-ready configuration for Oprel SDK.
Includes all settings for Week 1 features.
"""

import os
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


# Values llama.cpp accepts for --cache-type-k/v, plus Oprel's own "auto" and "turboquant" (= q4_0). 4-bit is the smallest there is.
KV_CACHE_TYPES = ("auto", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "turboquant")


CTX_MIN, CTX_MAX = 512, 262144


def runtime_settings_path() -> Path:
    """``~/.cache/oprel/runtime_settings.json`` (``OPREL_RUNTIME_SETTINGS_FILE`` overrides, used by tests)."""
    override = os.environ.get("OPREL_RUNTIME_SETTINGS_FILE")
    return Path(override) if override else Path.home() / ".cache" / "oprel" / "runtime_settings.json"


def normalize_kv_cache_type(value: Any) -> Optional[str]:
    kv = str(value or "").strip().lower()
    kv = {"q4": "q4_0", "q8": "q8_0", "q5": "q5_0"}.get(kv, kv)
    return kv if kv in KV_CACHE_TYPES else None


def valid_ctx_size(value: Any) -> Optional[int]:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if CTX_MIN <= n <= CTX_MAX else None


def read_runtime_settings() -> dict:
    """User-chosen model runtime settings that survive restarts. Missing/invalid entries mean "automatic"."""
    try:
        import json

        raw = json.loads(runtime_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict = {}
    ctx = valid_ctx_size(raw.get("ctx_size")) if isinstance(raw, dict) else None
    kv = normalize_kv_cache_type(raw.get("kv_cache_type")) if isinstance(raw, dict) else None
    if ctx:
        out["ctx_size"] = ctx
    if kv and kv != "auto":
        out["kv_cache_type"] = kv
    return out


def write_runtime_settings(ctx_size: Optional[int], kv_cache_type: Optional[str]) -> dict:
    """Persist the settings (``None`` / ``"auto"`` = automatic). Raises ValueError on invalid values."""
    import json

    out: dict = {}
    if ctx_size is not None:
        ctx = valid_ctx_size(ctx_size)
        if ctx is None:
            raise ValueError(f"context size must be between {CTX_MIN} and {CTX_MAX} tokens")
        out["ctx_size"] = ctx
    if kv_cache_type not in (None, "", "auto"):
        kv = normalize_kv_cache_type(kv_cache_type)
        if kv is None:
            raise ValueError(f"unsupported KV cache type '{kv_cache_type}'; use one of: {', '.join(KV_CACHE_TYPES)}")
        if kv != "auto":
            out["kv_cache_type"] = kv
    path = runtime_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return out


def _get_default_memory_limit() -> int:
    """
    Calculate a reasonable default memory limit based on available system RAM.
    
    Returns 80% of total system RAM in MB, with a minimum of 8192MB (8GB).
    This matches the approach used in the quantization recommender.
    """
    try:
        import psutil
        ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
        # Use 80% of total RAM (leave 20% for OS and other apps)
        # This is slightly more conservative than the 70% used for quantization selection,
        # but provides a safety margin for the monitor
        limit_mb = int(ram_total_gb * 0.8 * 1024)
        # Ensure minimum of 8GB
        return max(limit_mb, 8192)
    except Exception:
        # Fallback to 8GB if psutil fails
        return 8192


class Config(BaseModel):
    """
    Global configuration for Oprel SDK.

    Can be customized per-model or set globally.
    """

    # Paths
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "oprel" / "models")
    binary_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "oprel" / "bin")
    prompt_cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "oprel" / "prompt_cache")

    # Memory limits
    default_max_memory_mb: int = Field(
        default_factory=_get_default_memory_limit,
        description="Default max memory per model in MB (auto-calculated from system RAM)"
    )

    # Performance
    use_unix_socket: bool = Field(
        default=True, description="Use Unix sockets instead of HTTP (Linux/Mac only)"
    )
    n_threads: Optional[int] = Field(default=None, description="CPU threads (None for auto-detect)")
    n_gpu_layers: int = Field(default=-1, description="GPU layers to offload (-1 for auto)")
    ctx_size: int = Field(default=8192, description="Context size in tokens")
    batch_size: int = Field(default=512, description="Batch size for processing")
    
    # Memory Optimization (key differentiator from Ollama)
    kv_cache_type: str = Field(
        default="auto", 
        description="KV cache precision: auto (match model), turboquant (extreme 4-bit compression), iq4_nl, q4_0, q8_0, f16"
    )
    flash_attention: bool = Field(
        default=True, 
        description="Use Flash Attention for memory efficiency and speed"
    )
    mmap: bool = Field(
        default=True, 
        description="Use memory-mapped model loading for faster startup"
    )

    # Prompt Caching & KV Cache Reuse
    prompt_cache_enabled: bool = Field(
        default=True,
        description="Enable prompt/prefix caching to reuse KV cache across requests"
    )
    cache_reuse_tokens: int = Field(
        default=256,
        description="Minimum chunk size for KV cache prefix reuse in llama-server"
    )

    # Supermemory (Semantic Long-Term Conversation Memory)
    supermemory_enabled: bool = Field(
        default=True,
        description="Enable semantic long-term memory for chat conversations"
    )
    supermemory_recency_window: int = Field(
        default=10,
        description="Number of recent messages to preserve in active prompt context"
    )
    supermemory_top_k: int = Field(
        default=3,
        description="Number of historical memories to retrieve per query"
    )

    # Networking
    default_port_range: tuple[int, int] = Field(
        default=(54321, 54420), description="Port range for HTTP servers"
    )

    # Timeouts (M1.11 - Production-ready timeouts)
    connect_timeout_sec: float = Field(
        default=10.0, description="Connection timeout in seconds"
    )
    read_timeout_sec: float = Field(
        default=300.0, description="Read timeout for non-streaming requests (CPU can be slow)"
    )
    stream_timeout_sec: float = Field(
        default=120.0, description="Per-chunk timeout for streaming responses"
    )
    startup_timeout_sec: int = Field(
        default=60, description="Maximum seconds to wait for backend startup"
    )

    # Monitoring
    health_check_interval_sec: float = Field(
        default=1.0, description="How often to check process health"
    )
    health_check_timeout_sec: float = Field(
        default=5.0, description="Timeout for health check requests"
    )

    # Auto-restart settings (M1.8)
    auto_restart: bool = Field(
        default=False, description="Automatically restart backend on crash"
    )
    max_restarts: int = Field(
        default=3, description="Maximum restart attempts before giving up"
    )
    restart_delay_sec: float = Field(
        default=2.0, description="Delay between restart attempts"
    )

    # VRAM Monitoring (M1.4)
    vram_monitor_enabled: bool = Field(
        default=True, description="Enable VRAM monitoring during model loading"
    )
    vram_warning_threshold: float = Field(
        default=85.0, description="VRAM utilization % to trigger warning"
    )
    vram_critical_threshold: float = Field(
        default=95.0, description="VRAM utilization % to trigger critical alert"
    )

    # Logging
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Binary management
    auto_install_binaries: bool = Field(
        default=True, description="Automatically download backend binaries"
    )
    binary_version: str = Field(default="b9616", description="llama.cpp binary version to use")
    image_binary_version: str = Field(
        default="latest",
        description="stable-diffusion.cpp binary version to use for image generation",
    )
    
    # SSL/TLS settings (for binary downloads)
    ssl_verify: bool = Field(
        default=True, description="Verify SSL certificates when downloading binaries (disable for corporate proxies)"
    )
    ssl_cert_file: Optional[Path] = Field(
        default=None, description="Path to custom CA certificate bundle (for corporate proxies)"
    )

    # Vision preprocessing (image optimization before llama.cpp inference)
    vision_optimization_enabled: bool = Field(
        default=True,
        description="Downscale oversized images before vision model inference to reduce encoding latency",
    )
    vision_max_pixels: int = Field(
        default=409_600,
        ge=1,
        description=(
            "Maximum pixel count (width × height) for vision model inputs. "
            "Images exceeding this budget are downscaled with aspect ratio preserved. "
            "Reference values: 262144 ≈ 512×512, 409600 ≈ 640×640, "
            "589824 ≈ 768×768, 1048576 ≈ 1024×1024."
        ),
    )
    vision_image_quality: int = Field(
        default=90,
        ge=1,
        le=100,
        description="JPEG quality (1–100) used when re-encoding preprocessed vision images",
    )

    # OCR VLM Settings
    ocr_model_id: str = Field(
        default="glm-ocr",
        description="Default GGUF VLM model ID or alias used for OCR text extraction",
    )
    ocr_model_quant: str = Field(
        default="Q8_0",
        description="Default quantization for OCR VLM model",
    )

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def _ctx_size_from_env(cls, data: Any) -> Any:
        """``OPREL_CTX_SIZE`` / ``OPREL_KV_CACHE_TYPE`` or ``runtime_settings.json`` set the context window and KV cache type.

        An explicit value counts as "set by the user": it replaces the automatic hardware-tier default and the
        automatic VRAM cap, so the user's choice is what the backend is launched with.
        """
        if not isinstance(data, dict):
            return data
        # priority: explicit argument > environment variable > runtime settings file > automatic
        saved = read_runtime_settings()
        if "ctx_size" not in data:
            ctx = valid_ctx_size(os.environ.get("OPREL_CTX_SIZE", "").strip()) or saved.get("ctx_size")
            if ctx:
                data = {**data, "ctx_size": ctx}
        if "kv_cache_type" not in data:
            kv = normalize_kv_cache_type(os.environ.get("OPREL_KV_CACHE_TYPE", "")) or saved.get("kv_cache_type")
            if kv and kv != "auto":
                data = {**data, "kv_cache_type": kv}
        return data

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.binary_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_cache_dir.mkdir(parents=True, exist_ok=True)



