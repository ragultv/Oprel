"""
Model type detection and validation.
Determines if a model is text-generation, image-generation, video-generation, or vision.
"""

from typing import Optional, Literal
from oprel.downloader.aliases import get_model_category

ModelType = Literal["text-generation", "vision", "text-to-image", "text-to-video", "embeddings", "unknown"]


def detect_model_type(model_id: str) -> ModelType:
    """
    Detect the type of model based on its ID/alias.
    
    Args:
        model_id: Model ID or alias
        
    Returns:
        ModelType enum value
    """
    # Try to get category from aliases
    category = get_model_category(model_id)
    
    if category:
        # Direct category match
        if category in ["text-generation", "coding", "reasoning"]:
            return "text-generation"
        elif category in {"vision", "Text + Vision"}:
            return "vision"
        elif category == "text-to-image":
            return "text-to-image"
        elif category == "text-to-video":
            return "text-to-video"
        elif category == "embeddings":
            return "embeddings"
    
    # Fallback: detect from model name
    model_lower = model_id.lower()
    
    # Vision models
    if any(x in model_lower for x in ['vl', 'vision', 'llava', 'minicpm-v', 'moondream', 'internvl']):
        return "vision"
    
    # Image generation
    if any(x in model_lower for x in ['flux', 'sana', 'sdxl', 'stable-diff', 'pixart', 'auraflow', 'playground']):
        return "text-to-image"
    
    # Video generation
    if any(x in model_lower for x in ['wan', 'mochi', 'video', 'cogvideo', 'animatediff', 'ltx-video']):
        return "text-to-video"
    
    # Embeddings
    if 'embed' in model_lower:
        return "embeddings"
    
    # Default to text generation (most LLMs)
    return "text-generation"


def is_supported_model_type(model_type: ModelType) -> bool:
    """
    Check if a model type is currently supported by the Oprel runtime.
    
    Currently supported:
    - text-generation (llama.cpp backend)
    - vision (llama.cpp with multimodal support)
    - text-to-image (stable-diffusion.cpp backend)
    - embeddings (llama.cpp with pooling mode)
    
    Not yet supported:
    - text-to-video (requires a video diffusion backend)
    
    Args:
        model_type: Model type to check
        
    Returns:
        True if supported, False otherwise
    """
    supported_types = {"text-generation", "vision", "text-to-image", "embeddings"}
    return model_type in supported_types


def get_unsupported_message(model_type: ModelType) -> str:
    """
    Get a helpful error message for unsupported model types.
    
    Args:
        model_type: The unsupported model type
        
    Returns:
        Error message with guidance
    """
    messages = {
        "text-to-image": (
            "Image generation models are supported through the stable-diffusion.cpp backend.\n"
            "\n"
            "Use `oprel gen-image <model> \"your prompt\"` to generate images, or\n"
            "`oprel setup image` to download the backend binary for this machine.\n"
            "\n"
            "Browse available aliases with:\n"
            "See: oprel list-models --category text-to-image"
        ),
        "text-to-video": (
            "Video generation models are not yet supported in this version.\n"
            "\n"
            "Video generation models (Wan, Mochi, CogVideoX) require a video diffusion\n"
            "backend, which is planned for a future release.\n"
            "\n"
            "For now, please use text-generation models.\n"
            "See: oprel list-models --category text-generation"
        ),
        "embeddings": (
            "Embedding models are supported in this version.\n"
            "\n"
            "Use `oprel embed <model> \"text\"` to generate vectors.\n"
            "\n"
            "Browse available aliases with:\n"
            "See: oprel list-models --category embeddings"
        ),
    }
    
    return messages.get(model_type, f"Model type '{model_type}' is not supported")
