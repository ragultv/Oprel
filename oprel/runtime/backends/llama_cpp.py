"""llama.cpp backend implementation."""

import functools
import subprocess
from pathlib import Path
from typing import List, Optional, Set

from oprel.core.config import Config
from oprel.runtime.backends.base import BaseBackend
from oprel.telemetry.hardware import calculate_gpu_layers, detect_gpu
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


@functools.lru_cache(maxsize=32)
def _get_supported_binary_flags(binary_path: str) -> Set[str]:
    """Inspect binary --help to detect available CLI arguments (cached)."""
    try:
        proc = subprocess.run(
            [binary_path, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        recognized = {
            "--load-mode",
            "--mmap",
            "--no-mmap",
            "--mlock",
            "--no-mlock",
            "--flash-attn",
            "--no-warmup",
            "--parallel",
        }
        return {flag for flag in recognized if flag in output}
    except Exception as e:
        logger.debug(f"Could not inspect binary flags for {binary_path}: {e}")
        return set()


def _apply_memory_flags(
    cmd: List[str],
    use_mmap: bool,
    use_mlock: bool,
    supported_flags: Set[str],
) -> None:
    """Safely apply memory mapping and locking flags based on binary capabilities."""
    if "--load-mode" in supported_flags:
        if use_mmap and use_mlock:
            cmd.extend(["--load-mode", "mmap+mlock"])
        elif use_mlock:
            cmd.extend(["--load-mode", "mlock"])
        elif not use_mmap:
            cmd.extend(["--load-mode", "none"])
        else:
            # Default auto in modern llama-server uses mmap unless unsupported
            cmd.extend(["--load-mode", "mmap"])
    else:
        # Legacy binary flags (e.g. b9616 and earlier)
        if use_mmap and "--mmap" in supported_flags and "--mmap" not in cmd:
            cmd.append("--mmap")
        elif not use_mmap and "--no-mmap" in supported_flags and "--no-mmap" not in cmd:
            cmd.append("--no-mmap")
        if use_mlock and "--mlock" in supported_flags and "--mlock" not in cmd:
            cmd.append("--mlock")


def _resolve_kv_cache_type(config: Config, model_quantization: str, is_vision: bool) -> str:
    """
    Enforce strictly 4-bit or lower KV cache precision (q4_0, iq4_nl, q4_1).
    Eradicates f16 and q8_0 KV cache allocations across the entire application,
    including for vision/multimodal models, requiring zero modifications to llama.cpp.
    """
    requested = str(getattr(config, "kv_cache_type", "q4_0") or "q4_0").lower()
    if requested in ("turboquant", "turbo", "extreme"):
        return "q4_0"
    if requested in ("q4_0", "iq4_nl", "q4_1"):
        return requested

    # Hard clamp to 4-bit q4_0 for all other cases
    return "q4_0"


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
        self.ctx_size = getattr(config, "ctx_size", 32768)

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

        # Check if this is a vision/OCR model and add mmproj if needed
        try:
            from oprel.runtime.backends.vision import is_vision_model, detect_mmproj_file
            model_id = self.model_path.stem  # Get model name from path
            model_dir = self.model_path.parent
            mmproj_path = detect_mmproj_file(model_dir)

            if mmproj_path is not None:
                self.is_vision = True
                cmd.extend(["--mmproj", str(mmproj_path)])
                logger.info(f"Vision/OCR model detected - using mmproj: {mmproj_path.name}")
            elif is_vision_model(model_id):
                # Model is classified as a vision model but missing mmproj encoder
                logger.warning(
                    f"Vision/OCR model detected ('{model_id}') but no mmproj file found. "
                    f"Vision features may not work without the vision projector. "
                    f"Running in standard text mode."
                )
                self.is_vision = False
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
        
        try:
            model_size_gb = self.model_path.stat().st_size / (1024**3)
        except OSError:
            model_size_gb = 4.0
        # Use what is actually free: another app (browser, game) may hold VRAM.
        if vram_gb > 0:
            try:
                from oprel.telemetry.hardware import get_vram_info

                live = get_vram_info()
                if live:
                    vram_gb = min(vram_gb, max(live["vram_free_gb"], 0.0))
            except Exception:
                pass
            # Strictly preserve 350MB-400MB of VRAM headroom for CUDA context & context page buffers
            reserve_gb = getattr(self.config, "vram_reserve_mb", 400) / 1024.0
            vram_gb = max(0.0, vram_gb - reserve_gb)

        # Tier-based context default, only when the user did not set ctx_size.
        if "ctx_size" not in getattr(self.config, "model_fields_set", {"ctx_size"}):
            self.config = self.config.model_copy(update={"ctx_size": 32768})

        resolved_kv_cache_type = _resolve_kv_cache_type(
            self.config,
            metadata.quantization if metadata else "",
            self.is_vision,
        )
        
        supported_flags = _get_supported_binary_flags(str(self.binary_path))
        use_mmap = getattr(self.config, "mmap", True)
        use_mlock = False

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
            use_mmap = use_mmap and cpu_config.use_mmap
            use_mlock = cpu_config.use_mlock
            
            # Use optimized thread count (physical cores only)
            cmd.extend(["--threads", str(cpu_config.num_threads)])
            
            # Use optimized batch size for CPU
            cmd.extend(["--batch-size", str(cpu_config.batch_size)])
            
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
        
        # Context size: dynamically calculated from dedicated 600MB VRAM budget per model architecture
        target_ctx = getattr(self.config, "ctx_size", 32768)
        if metadata:
            model_max = metadata.context_length or 32768
            if vram_gb > 0:
                from oprel.runtime.kv_cache import recommend_max_context

                ctx_size = recommend_max_context(
                    available_vram_gb=vram_gb,
                    model_size_gb=model_size_gb,
                    hidden_dim=metadata.embedding_dim,
                    num_layers=metadata.num_layers,
                    num_kv_heads=metadata.num_kv_heads,
                    num_attention_heads=metadata.num_attention_heads,
                    kv_cache_type=resolved_kv_cache_type,
                    max_supported_ctx=model_max,
                    vram_budget_mb=600.0,
                )
            else:
                ctx_size = min(model_max, target_ctx)

            if ctx_size < model_max:
                logger.info(f"Context size configured to {ctx_size} tokens based on 600MB VRAM budget (model max: {model_max})")
            else:
                logger.info(f"Context size configured to {ctx_size} tokens (model max: {model_max})")
        else:
            ctx_size = target_ctx

        self.ctx_size = ctx_size
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
        
        _apply_memory_flags(cmd, use_mmap=use_mmap, use_mlock=use_mlock, supported_flags=supported_flags)

        # Strictly enforce 4-bit or lower KV cache (q4_0, iq4_nl, q4_1) natively in stock llama-server
        cache_type = resolved_kv_cache_type if resolved_kv_cache_type in ("q4_0", "iq4_nl", "q4_1") else "q4_0"
        cmd.extend(["--cache-type-k", cache_type])
        cmd.extend(["--cache-type-v", cache_type])
        logger.info(f"Native KV cache locked to {cache_type} (4-bit tiered memory)")

        # Prompt Caching & KV Cache Reuse
        if getattr(self.config, "prompt_cache_enabled", True):
            reuse_tokens = getattr(self.config, "cache_reuse_tokens", 256)
            if reuse_tokens and reuse_tokens > 0:
                cmd.extend(["--cache-reuse", str(reuse_tokens)])

            prompt_cache_dir = getattr(self.config, "prompt_cache_dir", None)
            if prompt_cache_dir:
                slot_save_path = Path(prompt_cache_dir)
                slot_save_path.mkdir(parents=True, exist_ok=True)
                cmd.extend(["--slot-save-path", str(slot_save_path)])
                logger.info(f"Prompt caching enabled with slot save path: {slot_save_path}")
        
        # Flash Attention — this binary version takes 'on'/'off'/'auto' as a value
        flash_attention = getattr(self.config, 'flash_attention', True)
        if flash_attention:
            cmd.extend(["--flash-attn", "auto"])

        # Limit slots to 1 for local desktop single-user inference to avoid multiplying context VRAM by 4
        if "--parallel" in supported_flags and "--parallel" not in cmd:
            cmd.extend(["--parallel", "1"])
        elif "-np" in supported_flags and "-np" not in cmd:
            cmd.extend(["-np", "1"])

        # Skip dummy warmup inference pass during startup to avoid 30-50s delay
        if "--no-warmup" in supported_flags and "--no-warmup" not in cmd:
            cmd.append("--no-warmup")
        
        return cmd

    def get_api_format(self) -> str:
        """llama.cpp uses OpenAI-compatible API"""
        return "openai"
