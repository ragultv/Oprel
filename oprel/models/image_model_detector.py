"""Helpers for detecting image model layouts."""

from pathlib import Path
from typing import Literal, Optional

from oprel.utils.logging import get_logger

logger = get_logger(__name__)

ModelFormat = Literal["diffusers", "merged_checkpoint", "gguf", "unknown"]


def detect_image_model_format(model_path: Path) -> ModelFormat:
    """
    Detect the layout used by an image model path.

    Returns:
        "gguf": GGUF image model file for stable-diffusion.cpp.
        "unknown": Unrecognized format.
    """
    if not model_path.exists():
        return "unknown"

    if model_path.is_dir():
        gguf_files = list(model_path.rglob("*.gguf"))
        if gguf_files:
            logger.info("Detected GGUF image model directory: %s", model_path)
            return "gguf"

    if model_path.is_file():
        suffix = model_path.suffix.lower()
        if suffix == ".gguf":
            logger.info("Detected GGUF image model file: %s", model_path.name)
            return "gguf"

    return "unknown"


def is_lora(model_path: Path) -> bool:
    return False


def is_sd_compatible(model_name: str) -> bool:
    """Legacy compatibility shim retained for callers that still import it."""
    return model_name.lower().endswith(".gguf")


def validate_image_model_format(model_path: Path, model_name: str) -> tuple[bool, Optional[str]]:
    """Validate that an image model path looks loadable."""
    format_type = detect_image_model_format(model_path)

    if format_type == "unknown":
        error = (
            f"Unsupported image model format: {model_name}\n\n"
            f"The stable-diffusion.cpp backend only supports GGUF image models.\n"
            f"Expected a .gguf file or a directory containing .gguf files.\n\n"
            f"Found: {model_path}"
        )
        return False, error

    return True, None


def get_comfyui_workflow_type(model_format: ModelFormat, model_name: str) -> str:
    """
    Legacy helper retained for compatibility with older callers.

    Returns:
        "unsupported" for any non-GGUF layout.
    """
    if model_format == "gguf":
        return "gguf"
    return "unsupported"
