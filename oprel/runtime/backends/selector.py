"""Backend selector for the llama.cpp-only runtime."""

from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from oprel.telemetry.profiler import HardwareProfile, profile_hardware
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class BackendType(Enum):
    """Available backend types"""
    LLAMA_CPP = "llama.cpp"


class BackendSelector:
    """
    Backend selection logic.
    
    The runtime currently exposes a single supported backend: ``llama.cpp``.
    """
    
    def __init__(self, profile: Optional[HardwareProfile] = None):
        self.profile = profile or profile_hardware()
        logger.debug(f"Initialized BackendSelector with profile: {self.profile.gpu_type}")
    
    def select_backend(
        self,
        model_size_gb: Optional[float] = None,
        preferred_backend: Optional[str] = None,
    ) -> str:
        """Always returns llama.cpp"""
        return BackendType.LLAMA_CPP.value
    
    def _auto_select_backend(self, model_size_gb: Optional[float] = None) -> str:
        return BackendType.LLAMA_CPP.value
    
    def _validate_backend(self, backend: str) -> str:
        return BackendType.LLAMA_CPP.value
    
    def _get_selection_reason(self, backend: str, model_size_gb: Optional[float] = None) -> str:
        return "Forced llama.cpp for stability"
    
    def get_fallback_chain(self, preferred_backend: str) -> list[str]:
        return [BackendType.LLAMA_CPP.value]
    
    def recommend_backend_for_model(self, model_path: Path) -> Dict[str, any]:
        return {
            "backend": BackendType.LLAMA_CPP.value,
            "reason": "Forced llama.cpp",
            "alternatives": [],
            "warnings": [],
        }


def select_backend(model_path, vram_gb, model_format='gguf'):
    """Always use llama.cpp for GGUF. It's stable and proven."""
    return 'llama.cpp'


def get_backend_info(backend: str) -> Dict[str, str]:
    """
    Get information about a backend.
    """
    info = {
        BackendType.LLAMA_CPP.value: {
            "name": "llama.cpp",
            "description": "C++ inference engine with broad compatibility",
            "advantages": [
                "Works on all hardware (CPU, low-end GPU, high-end GPU)",
                "Excellent CPU performance with quantization",
                "Hybrid GPU/CPU inference for large models",
                "Low memory overhead",
            ],
            "best_for": "CPU-only systems, low VRAM GPUs (<6GB), very large models",
        },
    }
    
    return info.get(backend, {
        "name": backend,
        "description": "Unknown backend",
        "advantages": [],
        "best_for": "Unknown",
    })
