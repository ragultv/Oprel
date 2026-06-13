"""Quantization selector for llama.cpp GGUF models."""

from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QuantizationRecommendation:
    """Recommended quantization with rationale"""
    quantization: str  # Q4_K_M, Q8_0, F16, etc.
    reason: str  # Why this quant was chosen
    expected_vram_gb: float  # Expected VRAM usage
    quality_level: str  # "high", "medium", "low"
    speed_estimate: str  # "fast", "medium", "slow"


class QuantizationSelector:
    """
    Auto-select optimal quantization based on VRAM and model size.
    Always chooses the highest-quality GGUF quantization that fits in
    the available VRAM budget.
    """
    
    # Quantization quality ranking (best to worst)
    QUANT_QUALITY = {
        "F16": 1.0,      # Full precision (baseline)
        "F32": 1.0,      # Full precision
        "Q8_0": 0.95,    # Virtually indistinguishable from F16
        "Q6_K": 0.90,    # Excellent quality
        "Q5_K_M": 0.85,  # Very good
        "Q5_K_S": 0.83,
        "Q4_K_M": 0.75,  # Good (Ollama default)
        "Q4_K_S": 0.72,
        "Q4_0": 0.70,
        "Q3_K_M": 0.60,  # Acceptable
        "Q3_K_S": 0.55,
        "Q2_K": 0.45,    # Low quality, only for extreme constraints
    }
    
    # Memory multipliers (relative to base model size)
    QUANT_SIZE_MULTIPLIERS = {
        "F32": 1.00,
        "F16": 0.50,
        "Q8_0": 0.25,
        "Q6_K": 0.19,
        "Q5_K_M": 0.16,
        "Q5_K_S": 0.15,
        "Q4_K_M": 0.13,
        "Q4_K_S": 0.12,
        "Q4_0": 0.12,
        "Q3_K_M": 0.10,
        "Q3_K_S": 0.09,
        "Q2_K": 0.07,
    }
    
    # Speed estimates (relative to F16)
    QUANT_SPEED = {
        "F16": 1.0,
        "Q8_0": 1.2,
        "Q6_K": 1.4,
        "Q5_K_M": 1.5,
        "Q4_K_M": 1.7,  # Sweet spot: good speed + quality
        "Q4_0": 1.8,
        "Q3_K_M": 2.0,
        "Q2_K": 2.2,
    }
    
    def __init__(self, backend: str = "llama.cpp"):
        """
        Initialize quantization selector.
        
        Args:
            backend: Only ``llama.cpp`` is supported.
        """
        self.backend = backend
    
    def recommend(
        self,
        model_size_gb: float,
        available_vram_gb: float,
        context_length: int = 2048,
        prefer_speed: bool = False
    ) -> QuantizationRecommendation:
        """
        Recommend optimal quantization for given hardware.
        
        Args:
            model_size_gb: Base model size in GB (F32)
            available_vram_gb: Available VRAM
            context_length: Requested context length
            prefer_speed: Prioritize speed over quality
            
        Returns:
            QuantizationRecommendation with best quant for hardware
        """
        # Reserve 20% VRAM for KV cache and overhead
        usable_vram = available_vram_gb * 0.80
        
        # Estimate KV cache size (rough approximation)
        kv_cache_gb = self._estimate_kv_cache(model_size_gb, context_length)
        
        # Available VRAM for model weights
        vram_for_weights = usable_vram - kv_cache_gb
        
        logger.debug(
            f"VRAM budget: {available_vram_gb:.1f}GB total, "
            f"{vram_for_weights:.1f}GB for weights, "
            f"{kv_cache_gb:.1f}GB for KV cache"
        )
        
        if self.backend != "llama.cpp":
            logger.warning("Unsupported backend '%s'; falling back to llama.cpp rules", self.backend)

        return self._recommend_gguf(model_size_gb, vram_for_weights, prefer_speed)
    
    def _recommend_gguf(
        self,
        base_size_gb: float,
        available_vram_gb: float,
        prefer_speed: bool
    ) -> QuantizationRecommendation:
        """Recommend GGUF quantization for llama.cpp."""
        
        # Try quantizations from best to worst quality
        quants_by_quality = sorted(
            self.QUANT_QUALITY.items(),
            key=lambda x: x[1], 
            reverse=True
        )
        
        for quant, quality in quants_by_quality:
            if quant not in self.QUANT_SIZE_MULTIPLIERS:
                continue
            
            multiplier = self.QUANT_SIZE_MULTIPLIERS[quant]
            quant_size = base_size_gb * multiplier
            
            if quant_size <= available_vram_gb:
                # Found a fit!
                speed_mult = self.QUANT_SPEED.get(quant, 1.0)
                
                # If prefer_speed, skip to Q4_K_M or lower
                if prefer_speed and quality > 0.75:
                    continue
                
                # Determine quality level
                if quality >= 0.85:
                    quality_level = "high"
                elif quality >= 0.70:
                    quality_level = "medium"
                else:
                    quality_level = "low"
                
                # Speed estimate
                if speed_mult >= 1.6:
                    speed_est = "fast"
                elif speed_mult >= 1.2:
                    speed_est = "medium"
                else:
                    speed_est = "slow"
                
                reason = self._get_reason(quant, available_vram_gb, quant_size)
                
                return QuantizationRecommendation(
                    quantization=quant,
                    reason=reason,
                    expected_vram_gb=quant_size,
                    quality_level=quality_level,
                    speed_estimate=speed_est
                )
        
        # Nothing fits - recommend Q2_K with warning
        return QuantizationRecommendation(
            quantization="Q2_K",
            reason="Model too large for VRAM. Consider hybrid GPU/CPU offloading.",
            expected_vram_gb=base_size_gb * 0.07,
            quality_level="low",
            speed_estimate="fast"
        )
    
    def _estimate_kv_cache(self, model_size_gb: float, context_length: int) -> float:
        """
        Rough KV cache size estimation.
        More accurate calculation in kv_cache.py.
        """
        # Rule of thumb: KV cache ≈ 0.5-1GB per 1B params for 2048 context
        # Assume ~1B params per 2GB model size (Q4)
        params_b = model_size_gb / 2.0
        
        # Scale by context length
        context_scale = context_length / 2048.0
        
        # KV cache size
        kv_cache_gb = params_b * 0.7 * context_scale
        
        return kv_cache_gb
    
    def _get_reason(self, quant: str, vram_gb: float, size_gb: float) -> str:
        """Generate human-readable reason for recommendation"""
        fit_pct = (size_gb / vram_gb) * 100
        
        if quant in ["F16", "F32"]:
            return f"Full precision - best quality ({fit_pct:.0f}% VRAM used)"
        elif quant == "Q8_0":
            return f"Q8 - near FP16 quality, faster ({fit_pct:.0f}% VRAM)"
        elif quant in ["Q6_K", "Q5_K_M"]:
            return f"High quality quantization ({fit_pct:.0f}% VRAM)"
        elif quant in ["Q4_K_M", "Q4_K_S"]:
            return f"Balanced quality/size ({fit_pct:.0f}% VRAM)"
        else:
            return f"Optimized for VRAM constraints ({fit_pct:.0f}% VRAM)"


def auto_select_quantization(
    model_size_gb: float,
    available_vram_gb: float,
    backend: str = "llama.cpp",
    context_length: int = 2048,
    prefer_speed: bool = False
) -> str:
    """
    Convenience function: Auto-select quantization, return quant string.
    
    Example:
        >>> quant = auto_select_quantization(7.0, 12.0)
        >>> print(quant)  # "Q8_0" (best quality that fits)
    """
    selector = QuantizationSelector(backend)
    rec = selector.recommend(model_size_gb, available_vram_gb, context_length, prefer_speed)
    
    logger.info(f"Auto-selected {rec.quantization}: {rec.reason}")
    return rec.quantization
