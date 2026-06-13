"""
Multimodal backend support for vision models (VLMs).
Handles image input for models like LLaVA, Qwen-VL, MiniCPM-V.
"""

import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def encode_image_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def is_vision_model(model_id: str) -> bool:
    """
    Check if a model is a vision model (VLM).
    
    Args:
        model_id: Model identifier
        
    Returns:
        True if vision model
    """
    model_lower = model_id.lower()
    vision_keywords = [
        'llava', 'vl', 'minicpm-v', 
        'moondream', 'internvl', 'cogvlm', 'vision'
    ]
    return any(keyword in model_lower for keyword in vision_keywords)


def detect_mmproj_file(model_dir: Path) -> Optional[Path]:
    """
    Detect the vision projection file (mmproj) for multimodal models.
    
    Vision models need two files:
    1. Main GGUF file (text decoder)
    2. MMProj GGUF file (vision encoder)
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Path to mmproj file if found
    """
    if not model_dir.exists():
        return None
    
    # Look for mmproj files
    mmproj_patterns = [
        '*mmproj*.gguf',
        '*vision*.gguf',
        '*clip*.gguf',
    ]
    
    # Search in current directory
    for pattern in mmproj_patterns:
        matches = list(model_dir.glob(pattern))
        if matches:
            logger.info(f"Found vision projection file: {matches[0].name}")
            return matches[0]
    
    # Search in parent directory (HF cache structure)
    if model_dir.parent and model_dir.parent.exists():
        for pattern in mmproj_patterns:
            matches = list(model_dir.parent.glob(pattern))
            if matches:
                logger.info(f"Found vision projection file in parent dir: {matches[0].name}")
                return matches[0]
    
    # Search recursively in cache (HF snapshots structure)
    cache_root = model_dir
    while cache_root.parent != cache_root:
        if "snapshots" in str(cache_root):
            # We're in a snapshot dir, search sibling snapshots too
            snapshots_dir = cache_root.parent
            for pattern in mmproj_patterns:
                matches = list(snapshots_dir.rglob(pattern))
                if matches:
                    logger.info(f"Found vision projection file in cache: {matches[0].name}")
                    return matches[0]
            break
        cache_root = cache_root.parent
    
    return None


def format_vision_prompt(
    text_prompt: str,
    image_paths: List[str],
    model_architecture: str = "llava"
) -> Dict[str, Any]:
    """
    Format a vision prompt for multimodal inference.
    
    Different vision models use different prompt formats:
    - LLaVA: "USER: <image>\n{prompt}\nASSISTANT:"
    - Qwen-VL: "<img>{image}</img>{prompt}"
    - MiniCPM-V: Uses special tokens
    
    Args:
        text_prompt: Text question/prompt
        image_paths: List of image file paths
        model_architecture: Model architecture (llava, qwen-vl, minicpm-v)
        
    Returns:
        Dict with 'prompt' and 'images' keys
    """
    # Encode images to base64
    encoded_images = []
    for img_path in image_paths:
        if Path(img_path).exists():
            encoded_images.append(encode_image_base64(img_path))
        else:
            logger.warning(f"Image not found: {img_path}")
    
    if not encoded_images:
        raise ValueError("No valid images provided")
    
    # Format based on architecture
    # Since we use the OpenAI chat completions api (/v1/chat/completions) with an array of
    # text and image_url objects, we do NOT need to insert manual placeholders like [IMAGE] or <img>
    # The server (llama.cpp or vLLM) will use the model's native chat template to insert them properly.
    prompt = text_prompt
    
    return {
        "prompt": prompt,
        "images": encoded_images,
        "num_images": len(encoded_images)
    }


def get_vision_model_config(model_id: str) -> Dict[str, Any]:
    """
    Get configuration for a specific vision model.
    
    Returns model-specific settings for optimal inference.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Configuration dict
    """
    model_lower = model_id.lower()
    
    # LLaVA models
    if 'llava' in model_lower:
        return {
            'architecture': 'llava',
            'requires_mmproj': True,
            'image_token': '<image>',
            'max_images': 1,
            'recommended_ctx': 8192,
        }
    
    # Qwen-VL models
    elif 'qwen' in model_lower and 'vl' in model_lower:
        return {
            'architecture': 'qwen-vl',
            'requires_mmproj': True,
            'image_token': '<img>',
            'max_images': 8,  # Qwen-VL supports multiple images
            'recommended_ctx': 8192,
        }
    
    # MiniCPM-V models
    elif 'minicpm' in model_lower:
        return {
            'architecture': 'minicpm-v',
            'requires_mmproj': True,
            'image_token': '<image>',
            'max_images': 1,
            'recommended_ctx': 8192,
        }
    
    # Moondream
    elif 'moondream' in model_lower:
        return {
            'architecture': 'moondream',
            'requires_mmproj': True,
            'image_token': '<image>',
            'max_images': 1,
            'recommended_ctx': 2048,
        }
    
    # Default config
    else:
        return {
            'architecture': 'unknown',
            'requires_mmproj': True,
            'image_token': '<image>',
            'max_images': 1,
            'recommended_ctx': 8192,
        }


class VisionModelValidator:
    """
    Validates that vision models have the required files and setup.
    """
    
    @staticmethod
    def validate_vision_model(model_dir: Path, model_id: str) -> Dict[str, Any]:
        """
        Validate that a vision model has all required components.
        
        Args:
            model_dir: Model directory
            model_id: Model identifier
            
        Returns:
            Validation result dict with 'valid', 'mmproj_path', 'errors'
        """
        result = {
            'valid': False,
            'mmproj_path': None,
            'errors': []
        }
        
        # Check if directory exists
        if not model_dir.exists():
            result['errors'].append(f"Model directory not found: {model_dir}")
            return result
        
        # Check for main GGUF file
        main_gguf = list(model_dir.glob('*.gguf'))
        if not main_gguf:
            result['errors'].append("No GGUF files found in model directory")
            return result
        
        # Check for mmproj file
        config = get_vision_model_config(model_id)
        if config['requires_mmproj']:
            mmproj = detect_mmproj_file(model_dir)
            if not mmproj:
                result['errors'].append(
                    f"Vision projection file (mmproj) not found. "
                    f"Vision models require both the main model and vision encoder. "
                    f"Download the complete model package."
                )
                return result
            result['mmproj_path'] = mmproj
        
        # All checks passed
        result['valid'] = True
        return result
