"""
Model information utilities for extracting parameters and quantizations
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi, list_repo_files
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


# Quantization size multipliers (relative to FP16)
QUANT_SIZE_MULTIPLIERS = {
    "Q2_K": 0.125,      # 2-bit
    "Q2_K_S": 0.125,
    "Q3_K_S": 0.1875,   # 3-bit
    "Q3_K_M": 0.1875,
    "Q3_K_L": 0.1875,
    "Q4_0": 0.25,       # 4-bit
    "Q4_1": 0.25,
    "Q4_K_S": 0.25,
    "Q4_K_M": 0.25,
    "Q5_0": 0.3125,     # 5-bit
    "Q5_1": 0.3125,
    "Q5_K_S": 0.3125,
    "Q5_K_M": 0.3125,
    "Q6_K": 0.375,      # 6-bit
    "Q8_0": 0.5,        # 8-bit
    "F16": 1.0,         # Full precision
    "F32": 2.0,         # 32-bit float
}


def extract_model_parameters(model_id: str) -> Optional[str]:
    """
    Extract parameter count from model ID or name.
    
    Examples:
        "Qwen/Qwen3-235B-GGUF" -> "235B"
        "unsloth/Llama-3.3-70B-Instruct-GGUF" -> "70B"
        "microsoft/Phi-4-GGUF" -> "14B" (known mapping)
    
    Args:
        model_id: HuggingFace model ID
        
    Returns:
        Parameter count string (e.g., "7B", "14B") or None
    """
    # Known model parameter mappings
    known_params = {
        "phi-4": "14B",
        "phi-3.5-mini": "3.8B",
        "deepseek-coder-v2-16b": "16B",
        "hermes-3-8b": "8B",
        "gpt-oss-20b": "20B",
        "gpt-oss-120b": "120B",
        "devstral-2-24b": "24B",
    }
    
    # Check known mappings first
    model_lower = model_id.lower()
    for key, params in known_params.items():
        if key in model_lower:
            return params
    
    # Extract from model ID using regex
    # Patterns: 235B, 70B, 7B, 1.5B, 0.5B, 300M, etc.
    patterns = [
        r'[-_](\d+\.?\d*[BM])[-_]',  # Matches -235B-, -7B-, -1.5B-, -300M-
        r'[-_](\d+\.?\d*)[BM]',       # Matches -235B, -7B, -1.5B
        r'(\d+\.?\d*[BM])',           # Matches 235B, 7B, 1.5B, 300M anywhere
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_id, re.IGNORECASE)
        if match:
            param_str = match.group(1).upper()
            # Normalize: ensure it ends with B or M
            if not param_str.endswith(('B', 'M')):
                param_str += 'B'
            return param_str
    
    return None


def parameters_to_float(param_str: str) -> float:
    """
    Convert parameter string to float (in billions).
    
    Examples:
        "235B" -> 235.0
        "7B" -> 7.0
        "1.5B" -> 1.5
        "300M" -> 0.3
    
    Args:
        param_str: Parameter string (e.g., "7B", "300M")
        
    Returns:
        Parameter count in billions
    """
    param_str = param_str.upper().strip()
    
    if param_str.endswith('B'):
        return float(param_str[:-1])
    elif param_str.endswith('M'):
        return float(param_str[:-1]) / 1000.0
    else:
        # Try to parse as number
        try:
            return float(param_str)
        except ValueError:
            return 0.0


def calculate_model_size(parameters: str, quantization: str) -> float:
    """
    Calculate approximate model size in GB based on parameters and quantization.
    
    Formula: size_gb = params_billions * 2.0 * quant_multiplier
    (2.0 GB per billion parameters at FP16)
    
    Args:
        parameters: Parameter count (e.g., "7B", "235B")
        quantization: Quantization type (e.g., "Q4_K_M", "Q5_K_M")
        
    Returns:
        Estimated size in GB
    """
    params_b = parameters_to_float(parameters)
    multiplier = QUANT_SIZE_MULTIPLIERS.get(quantization.upper(), 0.25)
    
    # Base formula: 2 GB per billion params at FP16
    size_gb = params_b * 2.0 * multiplier
    
    return round(size_gb, 2)


def get_gguf_quantizations(repo_id: str) -> List[str]:
    """
    Fetch available GGUF quantizations from HuggingFace repo.
    
    Args:
        repo_id: HuggingFace repository ID
        
    Returns:
        List of quantization types (e.g., ["Q4_K_M", "Q5_K_M", "Q8_0"])
    """
    try:
        api = HfApi()
        model_info = api.model_info(repo_id, files_metadata=True)
        gguf_files = [
            sibling.rfilename
            for sibling in getattr(model_info, "siblings", [])
            if getattr(sibling, "rfilename", "").endswith(".gguf")
        ]

        if not gguf_files:
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
        
        quantizations = set()
        for file in gguf_files:
            # Extract quantization from filename
            # Common patterns:
            # - model-Q4_K_M.gguf
            # - model.Q5_K_M.gguf
            # - Qwen3-7B-Q4_K_M.gguf
            
            # Remove .gguf extension
            name = file.replace(".gguf", "")
            
            # Try to find quantization pattern
            for quant in QUANT_SIZE_MULTIPLIERS.keys():
                if quant in name.upper():
                    quantizations.add(quant)
                    break
        
        # Sort by quality (higher bits first)
        quant_order = ["F32", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
                       "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_L", "Q3_K_M", "Q3_K_S",
                       "Q2_K_S", "Q2_K"]
        
        sorted_quants = []
        for q in quant_order:
            if q in quantizations:
                sorted_quants.append(q)
        
        return sorted_quants
        
    except Exception as e:
        logger.error(f"Failed to fetch quantizations for {repo_id}: {e}")
        return []


def get_local_quantizations(model_id: str, cache_dir: Path) -> list[str]:
    """
    Get locally available quantizations for a model.
    
    Args:
        model_id: HuggingFace model ID
        cache_dir: Cache directory path
        
    Returns:
        List of locally available quantization types
    """
    # Convert repo_id to cache directory format: models--Author--Name
    cache_name = "models--" + model_id.replace("/", "--")
    model_cache_dir = cache_dir / cache_name
    
    local_quants = []
    
    if model_cache_dir.exists():
        # Check snapshots directory
        snapshots_dir = model_cache_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    # Find GGUF files
                    gguf_files = list(snapshot.glob("*.gguf"))
                    for file in gguf_files:
                        # Extract quantization from filename
                        name_upper = file.name.upper()
                        for quant in ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]:
                            if quant in name_upper:
                                if quant not in local_quants:
                                    local_quants.append(quant)
                                break
    
    return local_quants


def get_model_info(repo_id: str, alias: str = None) -> Dict:
    """
    Get comprehensive model information including parameters and quantizations.
    
    Args:
        repo_id: HuggingFace repository ID
        alias: Optional model alias
        
    Returns:
        Dictionary with model info
    """
    parameters = extract_model_parameters(repo_id)
    quantizations = get_gguf_quantizations(repo_id)
    
    # Calculate sizes for each quantization
    sizes = {}
    if parameters:
        for quant in quantizations:
            sizes[quant] = calculate_model_size(parameters, quant)
    
    return {
        "repo_id": repo_id,
        "alias": alias,
        "parameters": parameters or "Unknown",
        "quantizations": quantizations,
        "sizes": sizes,
        "default_quantization": quantizations[0] if quantizations else None
    }
