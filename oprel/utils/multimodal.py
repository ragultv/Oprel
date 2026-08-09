"""
Multimodal utilities for vision, image generation, and video generation.
Handles image/video I/O and format conversions for different model types.
"""

from pathlib import Path
from typing import Optional, List, Union
import base64
from io import BytesIO
import time
from PIL import Image

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def preprocess_image_to_bytes(
    image_input: Union[str, bytes, Path, Image.Image],
    max_dim: int = 1024,
    max_pixels: Optional[int] = None,
    quality: int = 80,
) -> bytes:
    """
    Preprocess an image for vision model input.

    Two operating modes (max_pixels takes priority when provided):

    * **Pixel-budget mode** (``max_pixels`` is set):
      Downscales the image so that ``width × height <= max_pixels`` while
      preserving the aspect ratio exactly.  This is the correct mode for
      vision inference — it produces consistently-sized inputs regardless of
      the image's original aspect ratio.  Never upscales.

    * **Legacy max-dim mode** (only ``max_dim`` provided):
      Clamps the longest axis to ``max_dim``.  Kept for backward compatibility
      with existing callers that don't pass ``max_pixels``.

    Both modes:
    - Return original bytes immediately if no resize is needed (passthrough).
    - Convert palette / RGBA images to RGB (JPEG compatibility).
    - Compress to JPEG once with the specified quality.

    Args:
        image_input: File path (str/Path), raw bytes, or PIL Image.
        max_dim:     Longest-axis cap — used only when ``max_pixels`` is None.
        max_pixels:  Pixel budget for pixel-budget mode.  When set, ``max_dim``
                     is ignored.  Pass ``config.vision_max_pixels`` here.
        quality:     JPEG output quality (1–100).

    Returns:
        bytes: Compressed image bytes ready for base64 encoding.
    """
    if max_pixels is not None:
        # ── Pixel-budget mode: delegate to VisionImageProcessor ──────────
        from oprel.utils.vision_processor import VisionImageProcessor

        # Normalise input to bytes so VisionImageProcessor can inspect the
        # format header without re-reading from disk.
        if isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as fh:
                raw_bytes: bytes = fh.read()
        elif isinstance(image_input, bytes):
            raw_bytes = image_input
        elif isinstance(image_input, Image.Image):
            buf = BytesIO()
            fmt = image_input.format or "JPEG"
            image_input.save(buf, format=fmt)
            raw_bytes = buf.getvalue()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        processor = VisionImageProcessor(max_pixels=max_pixels, quality=quality)
        result = processor.process(raw_bytes)
        return result.image_bytes

    # ── Legacy max-dim mode (backward compat) ────────────────────────────
    start_time = time.perf_counter()
    img = None

    try:
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Convert RGBA / P modes to RGB for JPEG compatibility
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if any dimension exceeds max_dim
        width, height = img.size
        if width > max_dim or height > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))

            try:
                resample_filter = Image.Resampling.BILINEAR
            except AttributeError:
                resample_filter = Image.BILINEAR  # type: ignore[attr-defined]

            img = img.resize((new_width, new_height), resample_filter)

        out_io = BytesIO()
        img.save(out_io, format="JPEG", quality=quality)
        compressed_bytes = out_io.getvalue()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Image preprocessed (resized to {img.size[0]}x{img.size[1]}, "
            f"compressed to JPEG) in {elapsed_ms:.1f}ms"
        )
        return compressed_bytes

    except Exception as exc:
        logger.error(f"Image preprocessing failed: {exc}", exc_info=True)
        if isinstance(image_input, bytes):
            return image_input
        elif isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as f:
                return f.read()
        raise ValueError(f"Failed to preprocess image: {exc}") from exc


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string for vision model input.
    Used by VLM models (qwen-vl, llava, moondream, etc.)
    
    Args:
        image_path: Path to image file (jpg, png, webp, etc.)
        
    Returns:
        Base64-encoded string suitable for API requests
    """
    image_file = Path(image_path)
    
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Preprocess image to compressed JPEG bytes
    compressed_bytes = preprocess_image_to_bytes(image_file)
    
    encoded = base64.b64encode(compressed_bytes).decode('utf-8')
    
    # Since we preprocessed it to JPEG, the mime type is always image/jpeg
    return f"data:image/jpeg;base64,{encoded}"


def save_generated_image(
    image_data: bytes,
    output_path: str,
    format: str = "png"
) -> str:
    """
    Save generated image bytes to file.
    Used by text-to-image models (flux, sana, sdxl, etc.)
    
    Args:
        image_data: Raw image bytes
        output_path: Where to save the image
        format: Image format (png, jpg, webp)
        
    Returns:
        Absolute path to saved image
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct extension
    if not output_file.suffix:
        output_file = output_file.with_suffix(f".{format}")
    
    # Save image
    with open(output_file, 'wb') as f:
        f.write(image_data)
    
    logger.info(f"Image saved to: {output_file}")
    return str(output_file.absolute())


def save_generated_video(
    video_data: bytes,
    output_path: str,
    format: str = "mp4"
) -> str:
    """
    Save generated video bytes to file.
    Used by text-to-video models (wan, mochi, cogvideox, etc.)
    
    Args:
        video_data: Raw video bytes
        output_path: Where to save the video
        format: Video format (mp4, webm, gif)
        
    Returns:
        Absolute path to saved video
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct extension
    if not output_file.suffix:
        output_file = output_file.with_suffix(f".{format}")
    
    # Save video
    with open(output_file, 'wb') as f:
        f.write(video_data)
    
    logger.info(f"Video saved to: {output_file}")
    return str(output_file.absolute())


def format_vision_prompt(
    text_prompt: str,
    image_paths: List[str],
    model_type: str = "auto"
) -> Union[str, dict]:
    """
    Format prompt for vision models with image inputs.
    Different VLM models expect different formats.
    
    Args:
        text_prompt: Text question/instruction
        image_paths: List of image file paths
        model_type: "qwen-vl", "llava", "moondream", or "auto"
        
    Returns:
        Formatted prompt (string or dict depending on model)
    """
    # Detect model type from image count if auto
    if model_type == "auto":
        if len(image_paths) > 1:
            model_type = "qwen-vl"  # Supports multi-image
        else:
            model_type = "llava"  # Standard VLM
    
    # Encode images
    encoded_images = [encode_image_to_base64(img) for img in image_paths]
    
    # Format based on model type
    if model_type == "qwen-vl":
        # Qwen-VL format: interleaved text and images
        messages = []
        for i, img in enumerate(encoded_images):
            messages.append({
                "type": "image",
                "image": img
            })
        messages.append({
            "type": "text",
            "text": text_prompt
        })
        return {"messages": messages}
    
    elif model_type == "llava":
        # LLaVA format: image + text
        return {
            "prompt": text_prompt,
            "images": encoded_images
        }
    
    elif model_type == "moondream":
        # Moondream simple format
        return {
            "image": encoded_images[0],
            "question": text_prompt
        }
    
    else:
        # Generic format
        return {
            "prompt": text_prompt,
            "images": encoded_images
        }


def get_model_category_from_alias(alias: str) -> Optional[str]:
    """
    Determine model category from alias name.
    Helps CLI auto-detect what kind of input/output to expect.
    
    Args:
        alias: Model alias (e.g., "qwen3-vl-7b", "flux-1-dev")
        
    Returns:
        Category: "vision", "text-to-image", "text-to-video", "text-generation", etc.
    """
    from oprel.downloader.aliases import get_model_category
    
    category = get_model_category(alias)
    if category:
        return category
    
    # Fallback: detect from alias name
    alias_lower = alias.lower()
    
    if any(x in alias_lower for x in ['vl', 'vision', 'llava', 'minicpm-v', 'moondream']):
        return "vision"
    elif any(x in alias_lower for x in ['flux', 'sana', 'sdxl', 'stable-diff', 'pixart']):
        return "text-to-image"
    elif any(x in alias_lower for x in ['wan', 'mochi', 'video', 'cogvideo', 'animatediff']):
        return "text-to-video"
    elif 'embed' in alias_lower:
        return "embeddings"
    else:
        return "text-generation"


def format_multimodal_request(
    prompt: str,
    model_alias: str,
    image_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> dict:
    """
    Auto-format request based on model category.
    Simplifies CLI usage by detecting model type and formatting accordingly.
    
    Args:
        prompt: Text prompt/question
        model_alias: Model alias to use
        image_paths: Input images (for vision models)
        output_path: Where to save generated media
        **kwargs: Additional model-specific parameters
        
    Returns:
        Formatted request dict for API
    """
    category = get_model_category_from_alias(model_alias)
    
    request = {
        "prompt": prompt,
        **kwargs
    }
    
    # Vision models: add image inputs
    if category == "vision":
        if not image_paths:
            raise ValueError("Vision models require --image argument")
        
        # Format with images
        formatted = format_vision_prompt(prompt, image_paths)
        request.update(formatted)
    
    # Image/video generation: add output format
    elif category in ["text-to-image", "text-to-video"]:
        if output_path:
            request["output_path"] = output_path
        else:
            # Auto-generate output filename
            import time
            timestamp = int(time.time())
            ext = "mp4" if category == "text-to-video" else "png"
            request["output_path"] = f"generated_{timestamp}.{ext}"
    
    # Text generation: standard format
    else:
        pass  # Already correct format
    
    return request
