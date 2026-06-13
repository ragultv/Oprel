"""
Hardware profiler (M1.21)

One-time hardware profiling with caching for fast startup.
Saves profile to ~/.oprel/hardware_profile.json
"""

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import psutil

from oprel.telemetry.hardware import detect_gpu, get_recommended_threads
from oprel.telemetry.memory import get_total_ram, get_available_ram
from oprel.utils.logging import get_logger

logger = get_logger(__name__)

# Profile cache location
PROFILE_CACHE_DIR = Path.home() / ".oprel"
PROFILE_CACHE_FILE = PROFILE_CACHE_DIR / "hardware_profile.json"

# Profile is valid for 7 days (hardware changes are rare)
PROFILE_VALIDITY_DAYS = 7


@dataclass
class HardwareProfile:
    """Complete hardware profile for backend selection and optimization"""
    
    # System info
    os_type: str  # "Windows", "Linux", "Darwin"
    os_version: str
    python_version: str
    
    # CPU info
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_features: List[str]  # ["AVX2", "AVX512", "SSE4.2", etc.]
    recommended_threads: int
    
    # RAM info
    ram_total_gb: float
    ram_available_gb: float
    max_model_size_gb: float
    
    # GPU info (optional)
    has_gpu: bool
    gpu_type: Optional[str]  # "cuda", "rocm", "metal", None
    gpu_name: Optional[str]
    vram_total_gb: Optional[float]
    cuda_version: Optional[str]
    
    # Disk info
    disk_total_gb: float
    disk_free_gb: float
    
    # Profile metadata
    profile_version: str = "1.0"
    profiled_at: float = 0.0  # Unix timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "os_type": self.os_type,
            "os_version": self.os_version,
            "python_version": self.python_version,
            "cpu_model": self.cpu_model,
            "cpu_cores_physical": self.cpu_cores_physical,
            "cpu_cores_logical": self.cpu_cores_logical,
            "cpu_features": self.cpu_features,
            "recommended_threads": self.recommended_threads,
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "max_model_size_gb": self.max_model_size_gb,
            "has_gpu": self.has_gpu,
            "gpu_type": self.gpu_type,
            "gpu_name": self.gpu_name,
            "vram_total_gb": self.vram_total_gb,
            "cuda_version": self.cuda_version,
            "disk_total_gb": self.disk_total_gb,
            "disk_free_gb": self.disk_free_gb,
            "profile_version": self.profile_version,
            "profiled_at": self.profiled_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "HardwareProfile":
        """Create from dictionary"""
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if profile is older than validity period"""
        age_days = (time.time() - self.profiled_at) / (60 * 60 * 24)
        return age_days > PROFILE_VALIDITY_DAYS
    
    def get_recommended_backend(self) -> str:
        """
        Get recommended backend based on hardware profile.
        
        Returns:
            "pytorch" or "llama.cpp"
        """
        if not self.has_gpu:
            return "llama.cpp"
        
        # PyTorch for mid-range and high-end GPUs (6GB+ VRAM)
        if self.vram_total_gb and self.vram_total_gb >= 6:
            return "pytorch"
        
        # llama.cpp for low-end GPUs and CPUs
        return "llama.cpp"


def _detect_cpu_features() -> List[str]:
    """
    Detect CPU features like AVX2, AVX512, SSE, etc.
    
    Returns:
        List of supported CPU features
    """
    features = []
    
    try:
        if platform.system() == "Windows":
            # Use wmic on Windows
            import subprocess
            result = subprocess.run(
                ["wmic", "cpu", "get", "caption"],
                capture_output=True,
                text=True,
                timeout=5
            )
            cpu_info = result.stdout
            
            # Basic detection based on CPU model
            # Modern CPUs (2015+) almost always have AVX2
            if "Intel" in cpu_info or "AMD" in cpu_info:
                features.append("SSE4.2")
                features.append("AVX")
                features.append("AVX2")
                
                # AVX512 is on newer Intel CPUs (Ice Lake+, 2019+)
                if "i9" in cpu_info or "i7-1" in cpu_info:
                    features.append("AVX512")
                    
        elif platform.system() == "Linux":
            # Read /proc/cpuinfo on Linux
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read().lower()
                    
                if "avx512" in cpuinfo:
                    features.append("AVX512")
                if "avx2" in cpuinfo:
                    features.append("AVX2")
                if "avx" in cpuinfo:
                    features.append("AVX")
                if "sse4_2" in cpuinfo:
                    features.append("SSE4.2")
                if "f16c" in cpuinfo:
                    features.append("F16C")
                    
            except Exception as e:
                logger.warning(f"Could not read /proc/cpuinfo: {e}")
                
        elif platform.system() == "Darwin":
            # Use sysctl on macOS
            import subprocess
            try:
                result = subprocess.run(
                    ["sysctl", "-a"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                sysctl_output = result.stdout.lower()
                
                if "avx512" in sysctl_output:
                    features.append("AVX512")
                if "avx2" in sysctl_output:
                    features.append("AVX2")
                if "avx" in sysctl_output:
                    features.append("AVX")
                    
                # Apple Silicon has NEON (ARM SIMD)
                if "arm" in platform.machine().lower():
                    features.append("NEON")
                    
            except Exception as e:
                logger.warning(f"Could not run sysctl: {e}")
        
        logger.debug(f"Detected CPU features: {features}")
        
    except Exception as e:
        logger.warning(f"Failed to detect CPU features: {e}")
        # Return safe defaults
        features = ["SSE4.2", "AVX"]
    
    return features if features else ["SSE4.2"]


def _get_cpu_model() -> str:
    """
    Get CPU model name.
    
    Returns:
        CPU model string
    """
    try:
        if platform.system() == "Windows":
            import subprocess
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                return lines[1].strip()
                
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
                        
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
            
    except Exception as e:
        logger.warning(f"Could not detect CPU model: {e}")
    
    return "Unknown CPU"


def _get_disk_info() -> tuple[float, float]:
    """
    Get disk space information.
    
    Returns:
        Tuple of (total_gb, free_gb)
    """
    try:
        # Get disk info for home directory (where models are cached)
        usage = psutil.disk_usage(str(Path.home()))
        total_gb = usage.total / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        
        logger.debug(f"Disk: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        return total_gb, free_gb
        
    except Exception as e:
        logger.warning(f"Could not get disk info: {e}")
        return 500.0, 100.0  # Default estimates


def profile_hardware(force: bool = False) -> HardwareProfile:
    """
    Profile system hardware capabilities.
    
    Args:
        force: Force re-profiling even if cache exists
        
    Returns:
        HardwareProfile with complete system information
    """
    # Check cache first (unless force profiling)
    if not force:
        cached_profile = _load_cached_profile()
        if cached_profile and not cached_profile.is_expired():
            logger.info("Using cached hardware profile")
            return cached_profile
    
    logger.info("Profiling hardware...")
    start_time = time.time()
    
    # Detect GPU
    gpu_info = detect_gpu()
    has_gpu = gpu_info is not None
    
    # Get RAM info
    ram_total = get_total_ram()
    ram_available = get_available_ram()
    
    # Estimate max model size (accounts for OS overhead)
    from oprel.telemetry.memory import estimate_max_model_size
    max_model_size = estimate_max_model_size()
    
    # Get disk info
    disk_total, disk_free = _get_disk_info()
    
    # Create profile
    profile = HardwareProfile(
        # System
        os_type=platform.system(),
        os_version=platform.version(),
        python_version=platform.python_version(),
        
        # CPU
        cpu_model=_get_cpu_model(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        cpu_features=_detect_cpu_features(),
        recommended_threads=get_recommended_threads(),
        
        # RAM
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        max_model_size_gb=max_model_size,
        
        # GPU
        has_gpu=has_gpu,
        gpu_type=gpu_info.get("gpu_type") if has_gpu else None,
        gpu_name=gpu_info.get("gpu_name") if has_gpu else None,
        vram_total_gb=gpu_info.get("vram_total_gb") if has_gpu else None,
        cuda_version=gpu_info.get("cuda_version") if has_gpu else None,
        
        # Disk
        disk_total_gb=disk_total,
        disk_free_gb=disk_free,
        
        # Metadata
        profiled_at=time.time(),
    )
    
    # Save to cache
    _save_profile(profile)
    
    elapsed = time.time() - start_time
    logger.info(f"Hardware profiling completed in {elapsed:.2f}s")
    
    return profile


def _load_cached_profile() -> Optional[HardwareProfile]:
    """
    Load hardware profile from cache.
    
    Returns:
        Cached HardwareProfile or None if not found/invalid
    """
    try:
        if not PROFILE_CACHE_FILE.exists():
            logger.debug("No cached profile found")
            return None
        
        with open(PROFILE_CACHE_FILE, "r") as f:
            data = json.load(f)
        
        profile = HardwareProfile.from_dict(data)
        logger.debug(f"Loaded profile from {PROFILE_CACHE_FILE}")
        
        return profile
        
    except Exception as e:
        logger.warning(f"Failed to load cached profile: {e}")
        return None


def _save_profile(profile: HardwareProfile) -> None:
    """
    Save hardware profile to cache.
    
    Args:
        profile: HardwareProfile to save
    """
    try:
        # Create cache directory if it doesn't exist
        PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(PROFILE_CACHE_FILE, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        
        logger.debug(f"Saved profile to {PROFILE_CACHE_FILE}")
        
    except Exception as e:
        logger.warning(f"Failed to save profile: {e}")


def print_profile_report(profile: Optional[HardwareProfile] = None) -> None:
    """
    Print detailed hardware profile report.
    
    Args:
        profile: HardwareProfile to print, or None to load/create one
    """
    if profile is None:
        profile = profile_hardware()
    
    print("\n" + "="*70)
    print("HARDWARE PROFILE")
    print("="*70)
    
    # System
    print(f"\n📋 System:")
    print(f"   OS:              {profile.os_type} {profile.os_version}")
    print(f"   Python:          {profile.python_version}")
    
    # CPU
    print(f"\n🔧 CPU:")
    print(f"   Model:           {profile.cpu_model}")
    print(f"   Cores:           {profile.cpu_cores_physical} physical, {profile.cpu_cores_logical} logical")
    print(f"   Features:        {', '.join(profile.cpu_features)}")
    print(f"   Threads:         {profile.recommended_threads} (recommended)")
    
    # RAM
    print(f"\n💾 Memory:")
    print(f"   Total RAM:       {profile.ram_total_gb:.1f} GB")
    print(f"   Available RAM:   {profile.ram_available_gb:.1f} GB")
    print(f"   Max Model Size:  {profile.max_model_size_gb:.1f} GB")
    
    # GPU
    if profile.has_gpu:
        print(f"\n🎮 GPU:")
        print(f"   Type:            {profile.gpu_type.upper()}")
        print(f"   Name:            {profile.gpu_name}")
        print(f"   VRAM:            {profile.vram_total_gb:.1f} GB")
        if profile.cuda_version:
            print(f"   CUDA Version:    {profile.cuda_version}")
    else:
        print(f"\n🎮 GPU:            None detected (CPU-only mode)")
    
    # Disk
    print(f"\n💿 Storage:")
    print(f"   Disk Space:      {profile.disk_free_gb:.1f} GB free / {profile.disk_total_gb:.1f} GB total")
    
    # Recommendation
    backend = profile.get_recommended_backend()
    print(f"\n✨ Recommended Backend: {backend}")
    
    # Profile age
    age_hours = (time.time() - profile.profiled_at) / 3600
    if age_hours < 1:
        age_str = f"{int(age_hours * 60)} minutes ago"
    elif age_hours < 24:
        age_str = f"{int(age_hours)} hours ago"
    else:
        age_str = f"{int(age_hours / 24)} days ago"
    
    print(f"\n📅 Profile Age:     {age_str}")
    print(f"   Cache Location:  {PROFILE_CACHE_FILE}")
    
    print("="*70 + "\n")


def clear_profile_cache() -> bool:
    """
    Delete cached hardware profile.
    
    Returns:
        True if deleted, False if didn't exist
    """
    try:
        if PROFILE_CACHE_FILE.exists():
            PROFILE_CACHE_FILE.unlink()
            logger.info(f"Deleted profile cache: {PROFILE_CACHE_FILE}")
            return True
        else:
            logger.info("No profile cache to delete")
            return False
            
    except Exception as e:
        logger.error(f"Failed to delete profile cache: {e}")
        return False
