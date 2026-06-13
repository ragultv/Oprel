"""
Hardware detection and telemetry

Production-ready hardware detection for GPU acceleration.
Supports: NVIDIA CUDA, AMD ROCm, Apple Metal

Key Features:
- Real-time VRAM monitoring during model loading
- Multi-method GPU detection (PyTorch, nvidia-smi, rocm-smi)
- Accurate GPU layer calculation with KV cache accounting
- Memory pressure warnings before OOM
"""

import platform
import sys
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List
import psutil

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# VRAM MONITORING (M1.4)
# =============================================================================

@dataclass
class VRAMSnapshot:
    """Point-in-time VRAM usage snapshot"""
    timestamp: float
    total_mb: float
    used_mb: float
    free_mb: float
    utilization_percent: float
    
    @property
    def is_critical(self) -> bool:
        """Returns True if VRAM usage is critically high (>95%)"""
        return self.utilization_percent > 95.0
    
    @property
    def is_warning(self) -> bool:
        """Returns True if VRAM usage is high (>85%)"""
        return self.utilization_percent > 85.0


class VRAMMonitor:
    """
    Real-time VRAM monitoring during model loading.
    
    Provides:
    - Continuous VRAM tracking in background thread
    - Warning callbacks when memory pressure is high
    - Peak usage tracking for debugging
    
    Usage:
        monitor = VRAMMonitor(on_warning=my_callback)
        monitor.start()
        # ... load model ...
        peak = monitor.get_peak_usage()
        monitor.stop()
    """
    
    def __init__(
        self,
        poll_interval_ms: int = 250,
        warning_threshold: float = 85.0,
        critical_threshold: float = 95.0,
        on_warning: Optional[Callable[[VRAMSnapshot], None]] = None,
        on_critical: Optional[Callable[[VRAMSnapshot], None]] = None,
    ):
        """
        Args:
            poll_interval_ms: How often to check VRAM (milliseconds)
            warning_threshold: Utilization % to trigger warning callback
            critical_threshold: Utilization % to trigger critical callback
            on_warning: Callback when warning threshold exceeded
            on_critical: Callback when critical threshold exceeded
        """
        self.poll_interval = poll_interval_ms / 1000.0
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.on_warning = on_warning
        self.on_critical = on_critical
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._snapshots: List[VRAMSnapshot] = []
        self._peak_usage_mb: float = 0.0
        self._lock = threading.Lock()
        self._warning_fired = False
        self._critical_fired = False
    
    def start(self) -> None:
        """Start monitoring in background thread"""
        if self._thread is not None:
            return
        
        self._stop_event.clear()
        self._warning_fired = False
        self._critical_fired = False
        self._snapshots.clear()
        self._peak_usage_mb = 0.0
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug("VRAM monitoring started")
    
    def stop(self) -> None:
        """Stop monitoring"""
        if self._thread is None:
            return
        
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        logger.debug(f"VRAM monitoring stopped. Peak usage: {self._peak_usage_mb:.1f}MB")
    
    def get_current(self) -> Optional[VRAMSnapshot]:
        """Get current VRAM snapshot"""
        return _get_vram_snapshot()
    
    def get_peak_usage(self) -> float:
        """Get peak VRAM usage in MB during monitoring"""
        with self._lock:
            return self._peak_usage_mb
    
    def get_history(self) -> List[VRAMSnapshot]:
        """Get all VRAM snapshots taken during monitoring"""
        with self._lock:
            return list(self._snapshots)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while not self._stop_event.is_set():
            try:
                snapshot = _get_vram_snapshot()
                if snapshot is None:
                    time.sleep(self.poll_interval)
                    continue
                
                with self._lock:
                    self._snapshots.append(snapshot)
                    # Keep only last 1000 snapshots to prevent memory bloat
                    if len(self._snapshots) > 1000:
                        self._snapshots = self._snapshots[-500:]
                    
                    if snapshot.used_mb > self._peak_usage_mb:
                        self._peak_usage_mb = snapshot.used_mb
                
                # Fire callbacks (only once per threshold crossing)
                if snapshot.utilization_percent >= self.critical_threshold:
                    if not self._critical_fired and self.on_critical:
                        self._critical_fired = True
                        try:
                            self.on_critical(snapshot)
                        except Exception as e:
                            logger.error(f"Critical callback error: {e}")
                elif snapshot.utilization_percent >= self.warning_threshold:
                    if not self._warning_fired and self.on_warning:
                        self._warning_fired = True
                        try:
                            self.on_warning(snapshot)
                        except Exception as e:
                            logger.error(f"Warning callback error: {e}")
                else:
                    # Reset flags when usage drops
                    self._warning_fired = False
                    self._critical_fired = False
                
            except Exception as e:
                logger.debug(f"VRAM monitoring error: {e}")
            
            time.sleep(self.poll_interval)


def _get_vram_snapshot() -> Optional[VRAMSnapshot]:
    """Get current VRAM snapshot using nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,utilization.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        
        if result.returncode != 0:
            return None
        
        line = result.stdout.strip().split('\n')[0]
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            total = float(parts[0])
            used = float(parts[1])
            free = float(parts[2])
            # utilization might have % sign
            util_str = parts[3].replace('%', '').strip()
            util = float(util_str) if util_str else (used / total * 100 if total > 0 else 0)
            
            return VRAMSnapshot(
                timestamp=time.time(),
                total_mb=total,
                used_mb=used,
                free_mb=free,
                utilization_percent=util,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    
    return None


def get_vram_usage() -> Optional[Dict[str, float]]:
    """
    Get current VRAM usage.
    
    Returns:
        Dict with 'total_mb', 'used_mb', 'free_mb', 'utilization_percent'
        or None if not available
    """
    snapshot = _get_vram_snapshot()
    if snapshot:
        return {
            'total_mb': snapshot.total_mb,
            'used_mb': snapshot.used_mb,
            'free_mb': snapshot.free_mb,
            'utilization_percent': snapshot.utilization_percent,
        }
    return None


def get_vram_info() -> Optional[Dict[str, float]]:
    """
    Get VRAM info in GB (Compatibility wrapper for VRAMMonitor).
    
    Returns:
        Dict with 'vram_total_gb', 'vram_used_gb', 'vram_free_gb'
    """
    usage = get_vram_usage()
    if usage:
        return {
            'vram_total_gb': usage['total_mb'] / 1024,
            'vram_used_gb': usage['used_mb'] / 1024,
            'vram_free_gb': usage['free_mb'] / 1024,
            'utilization_percent': usage['utilization_percent']
        }
    return None


def check_vram_for_model(model_size_gb: float, gpu_layers: int, estimated_layers: int = 32) -> Dict[str, Any]:
    """
    Check if there's enough VRAM for a model before loading.
    
    Args:
        model_size_gb: Model file size in GB
        gpu_layers: Number of layers to put on GPU
        estimated_layers: Estimated total layers in model
        
    Returns:
        Dict with 'can_load', 'warning', 'recommended_layers', 'message'
    """
    vram = get_vram_usage()
    if vram is None:
        return {
            'can_load': True,
            'warning': False,
            'recommended_layers': gpu_layers,
            'message': 'Could not check VRAM (nvidia-smi unavailable)',
        }
    
    # Estimate VRAM needed (rough calculation)
    # GPU layer memory = (model_size * layers_ratio) * 0.85 (overhead)
    layers_ratio = gpu_layers / estimated_layers if estimated_layers > 0 else 1.0
    estimated_vram_mb = model_size_gb * 1024 * layers_ratio * 0.85
    
    # Add KV cache estimate (conservative: 512MB for 4K context)
    estimated_vram_mb += 512
    
    free_mb = vram['free_mb']
    
    if estimated_vram_mb > free_mb:
        # Calculate how many layers we CAN fit
        safe_vram_mb = free_mb * 0.85  # Keep 15% buffer
        max_layers = int((safe_vram_mb - 512) / (model_size_gb * 1024 * 0.85) * estimated_layers)
        max_layers = max(0, max_layers)
        
        return {
            'can_load': max_layers > 0,
            'warning': True,
            'recommended_layers': max_layers,
            'message': (
                f"Requested {gpu_layers} GPU layers needs ~{estimated_vram_mb:.0f}MB, "
                f"but only {free_mb:.0f}MB free. Recommended: {max_layers} layers"
            ),
        }
    
    return {
        'can_load': True,
        'warning': estimated_vram_mb > free_mb * 0.85,
        'recommended_layers': gpu_layers,
        'message': f"VRAM OK: {estimated_vram_mb:.0f}MB needed, {free_mb:.0f}MB free",
    }


def get_hardware_info() -> Dict[str, Any]:
    """
    Detect system hardware capabilities.

    Returns:
        Dictionary with CPU, RAM, GPU, and OS information
    """
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }

    # Detect GPU
    gpu_info = detect_gpu()
    if gpu_info:
        info.update(gpu_info)

    return info


def _detect_nvidia_smi() -> Optional[Dict[str, Any]]:
    """
    Detect NVIDIA GPU using nvidia-smi (works without torch).
    
    Returns:
        GPU info dict or None if not available
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split('\n')[0]
            parts = line.split(',')
            if len(parts) >= 2:
                name = parts[0].strip()
                memory_mb = float(parts[1].strip())
                memory_gb = memory_mb / 1024
                
                logger.info(f"Detected NVIDIA GPU via nvidia-smi: {name} ({memory_gb:.1f} GB)")
                
                return {
                    "gpu_type": "cuda",
                    "gpu_name": name,
                    "vram_total_gb": round(memory_gb, 2),
                }
    except FileNotFoundError:
        # nvidia-smi not found
        pass
    except subprocess.TimeoutExpired:
        logger.debug("nvidia-smi timed out")
    except Exception as e:
        logger.debug(f"nvidia-smi detection failed: {e}")
    
    return None


# =============================================================================
# ROCm SUPPORT FOR AMD GPUs (M1.6)
# =============================================================================

def _detect_rocm() -> Optional[Dict[str, Any]]:
    """
    Detect AMD GPU using rocm-smi (Linux only).
    
    ROCm (Radeon Open Compute) is AMD's GPU compute platform.
    Similar to nvidia-smi but for AMD GPUs.
    
    Returns:
        GPU info dict or None if not available
    """
    if platform.system() != "Linux":
        return None
    
    try:
        # Try rocm-smi for AMD GPUs
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0 and result.stdout.strip():
            output = result.stdout
            
            # Parse GPU name
            gpu_name = "AMD GPU"
            for line in output.split('\n'):
                if "Card series:" in line or "GPU" in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        gpu_name = parts[1].strip()
                        break
            
            # Parse VRAM
            vram_gb = 8.0  # Default
            for line in output.split('\n'):
                if "Total Memory" in line or "VRAM Total Memory" in line:
                    # Format: "VRAM Total Memory (B): 8589934592"
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            vram_bytes = int(parts[1].strip().split()[0])
                            vram_gb = vram_bytes / (1024**3)
                        except (ValueError, IndexError):
                            pass
                        break
            
            logger.info(f"Detected AMD GPU via rocm-smi: {gpu_name} ({vram_gb:.1f} GB)")
            
            return {
                "gpu_type": "rocm",
                "gpu_name": gpu_name,
                "vram_total_gb": round(vram_gb, 2),
            }
            
    except FileNotFoundError:
        logger.debug("rocm-smi not found")
    except subprocess.TimeoutExpired:
        logger.debug("rocm-smi timed out")
    except Exception as e:
        logger.debug(f"rocm-smi detection failed: {e}")
    
    # Alternative: Try lspci for AMD GPU detection
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "VGA" in line and ("AMD" in line or "ATI" in line or "Radeon" in line):
                    # Found AMD GPU but no rocm-smi
                    logger.info(f"AMD GPU detected via lspci (ROCm not installed)")
                    return {
                        "gpu_type": "rocm",
                        "gpu_name": line.split(':')[-1].strip() if ':' in line else "AMD GPU",
                        "vram_total_gb": 8.0,  # Conservative default
                        "rocm_installed": False,
                    }
    except Exception:
        pass
    
    return None


# =============================================================================
# METAL SUPPORT FOR APPLE SILICON (M1.5)
# =============================================================================

def _detect_metal() -> Optional[Dict[str, Any]]:
    """
    Detect Apple Silicon GPU (Metal).
    
    On Apple Silicon, GPU uses unified memory shared with CPU.
    We need to be more conservative with memory allocation.
    
    Returns:
        GPU info dict or None if not available
    """
    if platform.system() != "Darwin":
        return None
    
    # Check for Apple Silicon
    if platform.machine() != "arm64":
        # Intel Mac - Metal available but different characteristics
        return None
    
    # Try to get accurate GPU info using system_profiler
    gpu_name = "Apple Silicon"
    gpu_cores = 0
    
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            
            for display in displays:
                chipset = display.get("sppci_model", "")
                if chipset:
                    gpu_name = chipset
                
                # Get GPU core count
                cores = display.get("sppci_cores", "")
                if cores:
                    try:
                        gpu_cores = int(cores.split()[0])
                    except (ValueError, IndexError):
                        pass
                    
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"system_profiler failed: {e}")
    
    # Get total system RAM (unified memory)
    total_ram = psutil.virtual_memory().total
    total_ram_gb = total_ram / (1024**3)
    
    # For Apple Silicon, llama.cpp can use a portion of unified memory
    # Conservative estimate: 60-70% of total RAM can be used for model
    # M1: 8GB/16GB, M1 Pro/Max: 16-64GB, M2: 8-24GB, M2 Pro/Max: 16-96GB
    
    # Estimate available GPU memory based on chip variant
    # This is the memory that can realistically be used for model inference
    if total_ram_gb >= 64:
        # M1 Max/M2 Max - can use ~75% for model
        available_for_gpu = total_ram_gb * 0.75
    elif total_ram_gb >= 32:
        # M1 Pro/M2 Pro - can use ~70% for model
        available_for_gpu = total_ram_gb * 0.70
    elif total_ram_gb >= 16:
        # M1/M2 with 16GB - can use ~65% for model
        available_for_gpu = total_ram_gb * 0.65
    else:
        # 8GB base model - very constrained
        available_for_gpu = total_ram_gb * 0.50
    
    # Try PyTorch MPS detection for compatibility check
    mps_available = False
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        pass
    
    logger.info(f"Detected {gpu_name} (Metal) - {total_ram_gb:.0f}GB unified memory, ~{available_for_gpu:.0f}GB available for GPU")
    
    return {
        "gpu_type": "metal",
        "gpu_name": gpu_name,
        "vram_total_gb": round(available_for_gpu, 2),  # Use available, not total
        "unified_memory_gb": round(total_ram_gb, 2),
        "gpu_cores": gpu_cores,
        "mps_available": mps_available,
    }


def detect_gpu() -> Optional[Dict[str, Any]]:
    """
    Detect available GPU and VRAM.
    Tries multiple methods in order: PyTorch CUDA, nvidia-smi, ROCm, Metal.

    Returns:
        GPU info dict or None if no GPU detected
        
    Dict keys:
        - gpu_type: "cuda", "rocm", "metal", or "cpu"
        - gpu_name: Human-readable GPU name
        - vram_total_gb: Available VRAM in GB
    """
    # Method 1: Try PyTorch CUDA (if installed)
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            logger.info(f"Detected NVIDIA GPU via PyTorch: {device.name}")
            return {
                "gpu_type": "cuda",
                "gpu_name": device.name,
                "vram_total_gb": round(device.total_memory / (1024**3), 2),
            }
    except ImportError:
        logger.debug("PyTorch not installed, trying nvidia-smi")

    # Method 2: Try nvidia-smi (Windows/Linux without torch)
    nvidia_gpu = _detect_nvidia_smi()
    if nvidia_gpu:
        return nvidia_gpu
    
    # Method 3: Try ROCm for AMD GPUs (Linux)
    rocm_gpu = _detect_rocm()
    if rocm_gpu:
        return rocm_gpu

    # Method 4: Try Metal for Apple Silicon
    metal_gpu = _detect_metal()
    if metal_gpu:
        return metal_gpu

    # No GPU detected
    logger.info("No GPU detected, will use CPU inference")
    return None


def get_recommended_threads() -> int:
    """
    Recommend optimal thread count for CPU inference.

    Returns:
        Number of threads to use
    """
    physical_cores = psutil.cpu_count(logical=False)

    # Use physical cores, not hyperthreads
    # Reserve 1-2 cores for system
    if physical_cores > 4:
        return physical_cores - 2
    elif physical_cores > 2:
        return physical_cores - 1
    else:
        return physical_cores


# =============================================================================
# GPU LAYER CALCULATION (Updated with Metal/ROCm support)
# =============================================================================

def calculate_gpu_layers(
    vram_gb: float, 
    model_size_gb: float, 
    reserve_vram_gb: float = 0.5,
    gpu_type: str = "cuda"
) -> int:
    """
    Calculate optimal number of GPU layers based on available VRAM.
    
    Uses percentage-based approach for better accuracy across different model sizes.
    Target: Use up to 90-95% of VRAM for maximum GPU acceleration.
    
    Args:
        vram_gb: Total GPU VRAM in GB
        model_size_gb: Model file size in GB
        reserve_vram_gb: VRAM to reserve for KV cache and CUDA overhead
        gpu_type: "cuda", "rocm", or "metal"
        
    Returns:
        Recommended number of GPU layers
    """
    # Adjust reserve based on GPU type
    if gpu_type == "metal":
        # Metal needs more headroom due to unified memory
        reserve_vram_gb = max(reserve_vram_gb, 1.0)
    elif gpu_type == "rocm":
        # ROCm is less mature, be more conservative
        reserve_vram_gb = max(reserve_vram_gb, 0.75)
    
    # Reserve VRAM for KV cache and GPU context
    available_vram = vram_gb - reserve_vram_gb
    
    if available_vram <= 0 or model_size_gb <= 0:
        return 0
    
    # Estimate total layers based on model size
    # This is architecture-dependent, these are typical values:
    # - Qwen/Llama 7B: 28-32 layers
    # - Llama 13B: 40 layers  
    # - Llama 70B: 80 layers
    if model_size_gb < 2:
        estimated_total_layers = 24  # Small models (1-3B)
    elif model_size_gb < 5:
        estimated_total_layers = 32  # 7B Q4/Q5 quantized (Qwen 7B has 28)
    elif model_size_gb < 8:
        estimated_total_layers = 36  # 7B higher quant or small 13B
    elif model_size_gb < 15:
        estimated_total_layers = 45  # 13B models
    elif model_size_gb < 40:
        estimated_total_layers = 60  # 33B-34B models
    else:
        estimated_total_layers = 80  # Large models (70B+)
    
    # Calculate how much of the model fits in VRAM
    # Model weights take ~80-90% of file size when loaded (rest is metadata)
    effective_model_size = model_size_gb * 0.85
    
    # Calculate percentage of model that fits
    vram_ratio = available_vram / effective_model_size
    
    # Adjust target utilization based on GPU type
    if gpu_type == "metal":
        # Metal is more sensitive to memory pressure
        target_util = 0.85
    elif gpu_type == "rocm":
        target_util = 0.90
    else:
        target_util = 0.95
    
    # Convert to layers
    gpu_layers = int(estimated_total_layers * min(vram_ratio * target_util, 1.0))
    
    # Ensure reasonable bounds
    gpu_layers = max(0, min(gpu_layers, estimated_total_layers))
    
    logger.debug(
        f"GPU layer calc [{gpu_type}]: VRAM={vram_gb:.1f}GB, model={model_size_gb:.1f}GB, "
        f"available={available_vram:.1f}GB, ratio={vram_ratio:.2f}, layers={gpu_layers}/{estimated_total_layers}"
    )
    
    return gpu_layers


def calculate_gpu_layers_from_info(
    gpu_info: Optional[Dict[str, Any]], 
    model_size_gb: float
) -> int:
    """
    Calculate GPU layers using detected GPU info.
    
    Convenience wrapper that extracts VRAM and type from gpu_info dict.
    
    Args:
        gpu_info: Dict from detect_gpu() or None
        model_size_gb: Model file size in GB
        
    Returns:
        Recommended number of GPU layers (0 if no GPU)
    """
    if gpu_info is None:
        return 0
    
    vram_gb = gpu_info.get("vram_total_gb", 0)
    gpu_type = gpu_info.get("gpu_type", "cuda")
    
    return calculate_gpu_layers(vram_gb, model_size_gb, gpu_type=gpu_type)


def get_system_info() -> Dict[str, Any]:
    """
    Get detailed system information for CLI display.
    
    Returns:
        Dict with OS, Python, CPU, RAM, and GPU details.
    """
    base_info = get_hardware_info()
    
    cpu_brand = platform.processor() 
    if not cpu_brand:
        cpu_brand = base_info.get('arch', 'Unknown CPU')
        
    return {
        'os': base_info['os'],
        'os_release': platform.release(),
        'python_version': sys.version.split()[0],
        'cpu_brand': cpu_brand,
        'cpu_cores': base_info.get('cpu_count', 0),
        'ram_total_gb': base_info.get('ram_total_gb', 0.0),
        'gpu_count': 1 if 'gpu_name' in base_info else 0,
        'gpu_name': base_info.get('gpu_name'),
        'vram_total_gb': base_info.get('vram_total_gb', 0.0),
    }
