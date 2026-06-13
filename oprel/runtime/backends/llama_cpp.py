"""llama.cpp backend implementation."""

from pathlib import Path
from typing import List, Optional

from oprel.core.config import Config
from oprel.runtime.backends.base import BaseBackend
from oprel.telemetry.hardware import calculate_gpu_layers, detect_gpu
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_kv_cache_type(config: Config, model_quantization: str, is_vision: bool) -> str:
    """Pick a KV cache type that balances quality and memory footprint."""
    requested = getattr(config, "kv_cache_type", "auto")
    if requested != "auto":
        return requested

    if is_vision:
        return "f16"

    quant = model_quantization.upper()
    if any(level in quant for level in ["Q2", "Q3", "Q4", "IQ1", "IQ2", "IQ3", "IQ4"]):
        return "q4_0"
    if any(level in quant for level in ["Q5", "Q6", "Q8"]):
        return "q8_0"
    return "f16"


def _recommend_batch_size(ctx_size: int, gpu_enabled: bool) -> int:
    """Keep batch sizing proportional to context to avoid unnecessary memory spikes."""
    if gpu_enabled:
        if ctx_size >= 32768:
            return 128
        if ctx_size >= 16384:
            return 192
        if ctx_size >= 8192:
            return 256
        if ctx_size >= 4096:
            return 384
        return 512

    if ctx_size >= 16384:
        return 128
    if ctx_size >= 8192:
        return 192
    return 256


class LlamaCppBackend(BaseBackend):
    """
    Backend implementation for llama.cpp server.

    Uses the pre-compiled llama-server binary with optimal settings
    for various GPU types (CUDA, ROCm, Metal).
    """

    def __init__(self, binary_path: Path, model_path: Path, config: Config):
        super().__init__(binary_path, model_path, config)
        self.is_vision = False

    def build_command(self, port: int) -> List[str]:
        """
        Build command with Month 2 optimizations + Vision model support + Embedding mode:
        - Auto quantization selection
        - KV cache-aware layer calculation
        - CPU optimization for non-GPU systems
        - Vision model support (mmproj for LLaVA, Qwen-VL, etc.)
        - Embedding mode support (pooling for text embeddings)
        """
        cmd = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

        # Check if this is an embedding model
        model_name_lower = self.model_path.name.lower()
        is_embedding_model = 'embed' in model_name_lower
        
        if is_embedding_model:
            # Enable embedding/pooling mode for embedding models
            cmd.append("--embedding")
            logger.info("Embedding model detected - enabling pooling mode")
            # For embedding models, we can return early with minimal config
            # They don't need GPU layers, context size, etc.
            cmd.extend(["--ctx-size", "2048"])  # Increased for larger chunks
            return cmd

        # Check if this is a vision model and add mmproj if needed
        try:
            from oprel.runtime.backends.vision import is_vision_model, detect_mmproj_file
            model_id = self.model_path.stem  # Get model name from path
            
            if is_vision_model(model_id):
                self.is_vision = True
                # Look for mmproj file in same directory as model
                model_dir = self.model_path.parent
                mmproj_path = detect_mmproj_file(model_dir)
                
                if mmproj_path:
                    cmd.extend(["--mmproj", str(mmproj_path)])
                    logger.info(f"Vision model detected - using mmproj: {mmproj_path.name}")
                else:
                    logger.warning(
                        f"Vision model detected but no mmproj file found. "
                        f"Vision features may not work properly. "
                        f"Download the complete model package including vision encoder."
                    )
        except Exception as e:
            logger.debug(f"Vision model check failed: {e}")

        # Get model metadata for intelligent configuration
        try:
            from oprel.models.gguf_parser import parse_gguf_fast
            metadata = parse_gguf_fast(str(self.model_path))
            logger.debug(f"Model metadata: {metadata.architecture} {metadata.quantization}, {metadata.num_layers} layers")
        except Exception as e:
            logger.warning(f"Could not parse GGUF metadata: {e}")
            metadata = None

        # GPU detection
        gpu_info = detect_gpu()
        gpu_type = gpu_info.get("gpu_type") if gpu_info else None
        vram_gb = gpu_info.get("vram_total_gb", 0) if gpu_info else 0
        
        model_size_gb = self.model_path.stat().st_size / (1024**3)
        resolved_kv_cache_type = _resolve_kv_cache_type(
            self.config,
            metadata.quantization if metadata else "",
            self.is_vision,
        )
        
        # GPU acceleration with Month 2 layer calculator
        if gpu_info and gpu_type in ("cuda", "metal", "rocm") and vram_gb > 0:
            n_gpu_layers = self.config.n_gpu_layers
            
            if n_gpu_layers == -1 and metadata:
                # Use Month 2 layer calculator (accounts for KV cache!)
                from oprel.runtime.layer_calculator import calculate_from_metadata
                ctx_size = getattr(self.config, 'ctx_size', 2048)
                
                layer_calc = calculate_from_metadata(
                    metadata,
                    available_vram_gb=vram_gb,
                    context_length=ctx_size,
                    kv_cache_type=resolved_kv_cache_type,
                    is_vision=self.is_vision
                )
                n_gpu_layers = layer_calc.gpu_layers
                
                logger.info(
                    f"Month 2 layer calc: {n_gpu_layers}/{layer_calc.total_layers} layers on GPU "
                    f"({layer_calc.gpu_percentage:.0f}%) - {layer_calc.recommendation}"
                )
            elif n_gpu_layers == -1:
                # Fallback to old calculation
                n_gpu_layers = calculate_gpu_layers(vram_gb, model_size_gb, gpu_type=gpu_type)
                logger.info(f"GPU layers (fallback): {n_gpu_layers}")
            
            if n_gpu_layers > 0:
                cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
                logger.info(f"GPU: {n_gpu_layers} layers on {gpu_info['gpu_name']}")
            else:
                logger.warning("GPU detected but layers=0, using CPU")
        else:
            # CPU-only: Apply Month 2 CPU optimizations
            logger.info("GPU not detected - applying CPU optimizations")
            
            from oprel.runtime.cpu_optimizer import get_cpu_config
            cpu_config = get_cpu_config(model_size_gb, prefer_speed=True)
            
            # Use optimized thread count (physical cores only)
            cmd.extend(["--threads", str(cpu_config.num_threads)])
            
            # Use optimized batch size for CPU
            cmd.extend(["--batch-size", str(cpu_config.batch_size)])
            
            # Enable memory mapping
            if cpu_config.use_mmap:
                cmd.append("--mmap")
            
            # Memory locking (if enough RAM)
            if cpu_config.use_mlock:
                cmd.append("--mlock")
            
            logger.info(
                f"CPU optimization: {cpu_config.num_threads} threads, "
                f"batch={cpu_config.batch_size}, variant={cpu_config.binary_variant}, "
                f"~{cpu_config.expected_speedup:.1f}x speedup"
            )
            
            # Warn if model too large for CPU
            from oprel.recommendations.cpu import check_model_for_cpu
            warning = check_model_for_cpu(model_size_gb)
            if warning:
                logger.warning(warning)

        gpu_enabled = any(arg == "--n-gpu-layers" for arg in cmd)
        
        # Context size - use a sane cap even if metadata says higher
        if metadata:
            max_safe_ctx = self.config.ctx_size
            if vram_gb > 0:
                from oprel.runtime.kv_cache import recommend_max_context

                kv_safe_ctx = recommend_max_context(
                    available_vram_gb=vram_gb,
                    model_size_gb=model_size_gb,
                    hidden_dim=metadata.embedding_dim,
                    num_layers=metadata.num_layers,
                    num_kv_heads=metadata.num_kv_heads,
                    num_attention_heads=metadata.num_attention_heads,
                    kv_cache_type=resolved_kv_cache_type,
                )
                max_safe_ctx = min(max_safe_ctx, kv_safe_ctx)

            ctx_size = min(metadata.context_length, max_safe_ctx)
            if ctx_size < metadata.context_length:
                logger.info(
                    f"Capped ctx-size from {metadata.context_length} to {ctx_size} "
                    f"(from config, to prevent OOM)"
                )
        else:
            ctx_size = self.config.ctx_size
        cmd.extend(["--ctx-size", str(ctx_size)])
        
        # Batch size (GPU mode only — CPU mode already set it above)
        if gpu_enabled:
            batch_size = min(
                getattr(self.config, "batch_size", 512),
                _recommend_batch_size(ctx_size, gpu_enabled=True),
            )
            cmd.extend(["--batch-size", str(batch_size)])
            logger.info(f"GPU batch size set to {batch_size} for ctx-size={ctx_size}")
        elif "--batch-size" not in cmd:
            cpu_batch_size = min(
                getattr(self.config, "batch_size", 256),
                _recommend_batch_size(ctx_size, gpu_enabled=False),
            )
            cmd.extend(["--batch-size", str(cpu_batch_size)])
        
        if getattr(self.config, "mmap", True) and "--mmap" not in cmd:
            cmd.append("--mmap")

        if resolved_kv_cache_type in ("q8_0", "q4_0", "q5_0", "q5_1"):
            cmd.extend(["--cache-type-k", resolved_kv_cache_type])
            cmd.extend(["--cache-type-v", resolved_kv_cache_type])
            logger.info(f"KV cache optimized to {resolved_kv_cache_type}")
        else:
            logger.debug("KV cache using default f16")
        
        # Flash Attention — this binary version takes 'on'/'off'/'auto' as a value
        flash_attention = getattr(self.config, 'flash_attention', True)
        if flash_attention:
            cmd.extend(["--flash-attn", "auto"])
        
        return cmd

    def get_api_format(self) -> str:
        """llama.cpp uses OpenAI-compatible API"""
        return "openai"
