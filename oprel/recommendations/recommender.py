"""
Intelligent model recommendation engine.
Analyzes hardware capabilities and suggests optimal models.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from oprel.telemetry.hardware import get_hardware_info
from oprel.downloader.aliases import OFFICIAL_REPOS, CATEGORY_INFO
from oprel.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ModelScore:
    alias: str
    category: str
    params_b: float
    estimated_size_gb: float
    score: float
    reason: str
    fit_status: str  # "perfect", "good", "tight", "too_large"
    device: str      # "GPU", "CPU", "Mixed"

class ModelRecommender:
    def __init__(self):
        self.hw_info = get_hardware_info()
        self.gpu_info = self.hw_info.get("gpu_name")
        self.vram_gb = self.hw_info.get("vram_total_gb", 0)
        self.ram_gb = self.hw_info.get("ram_total_gb", 0)
        self.cpu_cores = self.hw_info.get("cpu_count", 0)
        
        # Heuristic constants
        self.system_reserved_ram = 4.0  # GB
        self.system_reserved_vram = 1.0 # GB
        self.q4_bytes_per_param = 0.75  # GB per Billion params (approx for Q4_K_M + context)

    def _estimate_params(self, alias: str) -> float:
        """Extract parameter count from alias name (e.g., 'qwen3-8b' -> 8.0)"""
        match = re.search(r'[-_](\d+\.?\d*)b', alias.lower())
        if match:
            return float(match.group(1))
        
        # Fallback for known models without explicit size in name
        if "sdxl" in alias: return 6.6
        if "sd-1.5" in alias: return 1.0
        if "flux" in alias: return 12.0
        
        return 0.0

    def score_model(self, alias: str, category: str) -> Optional[ModelScore]:
        params = self._estimate_params(alias)
        if params == 0:
            return None

        # Estimate memory usage (Gb)
        # Assuming Q4 quantization for text models, 
        # and standard fp16/bf16 or optimized for image models
        if category == "text-to-image" or category == "text-to-video":
             # Image models often run in fp16 or int8, sizes vary
             # Use rough estimates based on params
             estimated_size = params * 0.8  # slightly higher per param for unquantized weights often used
        else:
             # GGUF Q4_K_M approximation
             estimated_size = params * self.q4_bytes_per_param + 0.5 # +500MB for context/kv-cache

        # Check fit
        fit_status = "too_large"
        device = "CPU"
        score = 0.0
        reason = []

        # memory available
        avail_ram = self.ram_gb - self.system_reserved_ram
        avail_vram = self.vram_gb - self.system_reserved_vram

        # check GPU fit
        if self.vram_gb > 0:
            if estimated_size <= avail_vram:
                fit_status = "perfect"
                device = "GPU"
                score += 100
                reason.append("Fits entirely in VRAM")
            elif estimated_size <= avail_vram * 1.5:
                # Partial GPU offloading
                fit_status = "good"
                device = "Mixed"
                score += 70
                reason.append("Partial GPU offloading")
            else:
                device = "CPU" # fallback to CPU logic check
        
        # Check CPU fit if not fully on GPU
        if device != "GPU":
            if estimated_size <= avail_ram:
                if fit_status == "too_large": 
                    fit_status = "good" if self.cpu_cores >= 8 else "tight"
                score += 40 # Base score for CPU
                
                # Penalize large models on few cores
                if self.cpu_cores < 4 and params > 3:
                     score -= 20
                     reason.append("Slow on dual-core CPU")
                elif self.cpu_cores >= 8:
                     score += 10
                     reason.append("Multi-core CPU boost")
            else:
                if fit_status == "too_large":
                    fit_status = "too_large"
                    score = 0
                    reason.append("Exceeds available RAM")

        # Prefer newer/better models logic (simple heuristics based on name)
        if "qwen3" in alias or "llama3" in alias or "flux" in alias:
            score += 15
            reason.append("New architecture")
        
        # Penalize very old models
        if "llama-2" in alias:
            score -= 10
        
        # Size preference: Bigger is better IF it fits
        if fit_status in ["perfect", "good"]:
            score += params * 2  # Reward size
        
        return ModelScore(
            alias=alias,
            category=category,
            params_b=params,
            estimated_size_gb=estimated_size,
            score=score,
            reason=", ".join(reason),
            fit_status=fit_status,
            device=device
        )

    def get_recommendations(self, limit: int = 5) -> Dict[str, List[ModelScore]]:
        rec_by_cat = {}
        
        for category, models in OFFICIAL_REPOS.items():
            scores = []
            for alias in models.keys():
                ms = self.score_model(alias, category)
                if ms and ms.fit_status != "too_large":
                    scores.append(ms)
            
            # Sort by score descending
            scores.sort(key=lambda x: x.score, reverse=True)
            rec_by_cat[category] = scores[:limit]
            
        return rec_by_cat

    def print_report(self):
        print(f"System Analysis")
        print(f"---------------")
        print(f"CPU: {self.cpu_cores} Cores")
        print(f"RAM: {self.ram_gb:.1f} GB")
        if self.gpu_info:
            print(f"GPU: {self.gpu_info} ({self.vram_gb:.1f} GB VRAM)")
        else:
            print(f"GPU: None detected (CPU-only mode)")
        print()

        recs = self.get_recommendations(limit=3)
        
        # Prioritize key categories
        priority_cats = ["text-generation", "coding", "vision", "text-to-image"]
        
        print("Recommended Models")
        print("==================")
        
        found_any = False
        for cat in priority_cats:
            if cat in recs and recs[cat]:
                found_any = True
                cat_info = CATEGORY_INFO.get(cat, {})
                print(f"\n{cat_info.get('icon', '')} {cat_info.get('name', cat).upper()}")
                print(f"{'Model Alias':<25} {'Size':<10} {'Device':<8} {'Score':<6} {'Reason'}")
                print("-" * 80)
                
                for m in recs[cat]:
                    star = "★" if m.score >= 90 else ""
                    print(f"{m.alias:<25} {m.estimated_size_gb:.1f}GB     {m.device:<8} {int(m.score):<6} {star} {m.reason}")
        
        if not found_any:
            print("\nNo models found that fit your hardware comfortably.")
            print("Try upgrading RAM or using smaller distilled models.")
            
        print("\n\nTo run a model:")
        print("  oprel run <model_alias>")

def show_recommendations():
    recommender = ModelRecommender()
    recommender.print_report()
