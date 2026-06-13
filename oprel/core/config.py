"""
Global configuration management

Production-ready configuration for Oprel SDK.
Includes all settings for Week 1 features.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


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
        description="KV cache precision: auto (match model), f16, q8_0, q4_0"
    )
    flash_attention: bool = Field(
        default=True, 
        description="Use Flash Attention for memory efficiency and speed"
    )
    mmap: bool = Field(
        default=True, 
        description="Use memory-mapped model loading for faster startup"
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
    binary_version: str = Field(default="b7822", description="llama.cpp binary version to use")
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

    class Config:
        arbitrary_types_allowed = True

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.binary_dir.mkdir(parents=True, exist_ok=True)



