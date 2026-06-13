"""Hugging Face integration for image-generation model assets."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files

from oprel.core.config import Config
from oprel.downloader.aliases import MODEL_ALIASES, resolve_model_id
from oprel.models.model_types import detect_model_type
from oprel.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_EXTENSIONS = (".gguf",)
EXCLUDED_KEYWORDS = (
    "lora",
    "controlnet",
    "upscaler",
    "esrgan",
    "embedding",
    "textual_inversion",
    "preview",
)

_LIKELY_LLM_ARCHITECTURES = {
    "llama",
    "mistral",
    "qwen",
    "phi",
    "gemma",
    "bert",
    "gptneox",
    "rwkv",
}


@dataclass
class ImageModelAssets:
    """Downloaded assets required for stable-diffusion.cpp inference."""

    repo_id: str | None
    model_path: Path | None = None
    diffusion_model_path: Path | None = None
    vae_path: Path | None = None
    clip_l_path: Path | None = None
    clip_g_path: Path | None = None
    t5xxl_path: Path | None = None
    llm_path: Path | None = None
    llm_vision_path: Path | None = None

    @property
    def primary_path(self) -> Path:
        if self.model_path:
            return self.model_path
        if self.diffusion_model_path:
            return self.diffusion_model_path
        raise FileNotFoundError("No image model file resolved")

    @property
    def uses_component_mode(self) -> bool:
        return self.diffusion_model_path is not None and self.model_path is None


def _is_local_model_path(model_id: str) -> bool:
    return Path(model_id).expanduser().exists()


def _download_file(repo_id: str, filename: str, cache_dir: Path, force_download: bool) -> Path:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
        resume_download=True,
        force_download=force_download,
    )
    return Path(path)


def _read_gguf_metadata_keys(file_path: Path) -> tuple[str, set[str]]:
    """Read GGUF metadata keys and the `general.architecture` value."""
    with file_path.open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 0x46554747:  # GGUF
            raise ValueError("not a GGUF file")

        version = struct.unpack("<I", f.read(4))[0]
        if version != 3:
            raise ValueError(f"unsupported GGUF version: {version}")

        _tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

        def skip_value(value_type: int) -> None:
            if value_type in (0, 1, 7):
                f.read(1)
            elif value_type in (2, 3):
                f.read(2)
            elif value_type in (4, 5, 6):
                f.read(4)
            elif value_type in (10, 11, 12):
                f.read(8)
            elif value_type == 8:  # string
                length = struct.unpack("<Q", f.read(8))[0]
                f.read(length)
            elif value_type == 9:  # array
                array_type = struct.unpack("<I", f.read(4))[0]
                array_len = struct.unpack("<Q", f.read(8))[0]
                for _ in range(array_len):
                    skip_value(array_type)
            else:
                raise ValueError(f"unknown GGUF metadata type: {value_type}")

        keys: set[str] = set()
        architecture = ""

        for _ in range(metadata_kv_count):
            key_len = struct.unpack("<Q", f.read(8))[0]
            key = f.read(key_len).decode("utf-8", errors="replace")
            value_type = struct.unpack("<I", f.read(4))[0]
            keys.add(key)

            if key == "general.architecture" and value_type == 8:
                length = struct.unpack("<Q", f.read(8))[0]
                architecture = f.read(length).decode("utf-8", errors="replace")
            else:
                skip_value(value_type)

    return architecture.lower().strip(), keys


def validate_image_gguf_compatibility(file_path: Path) -> tuple[bool, str | None]:
    """Validate whether a GGUF file is compatible with stable-diffusion.cpp."""
    if file_path.suffix.lower() != ".gguf":
        return False, "Expected a .gguf image model file."

    try:
        architecture, keys = _read_gguf_metadata_keys(file_path)
    except Exception as exc:
        return False, f"Could not parse GGUF metadata: {exc}"

    if architecture in _LIKELY_LLM_ARCHITECTURES:
        return (
            False,
            "This GGUF appears to be a text LLM model, not an image diffusion model.",
        )

    # stable-diffusion.cpp requires SD metadata for SD family GGUF models.
    if architecture.startswith("sd") and "sd.version" not in keys:
        return (
            False,
            "Missing required 'sd.version' GGUF metadata. This quant is not a stable-diffusion.cpp-compatible conversion.",
        )

    return True, None


def _select_main_model_file(repo_files: list[str]) -> str | None:
    candidates = [
        path
        for path in repo_files
        if path.lower().endswith(MODEL_EXTENSIONS)
        and not any(part.startswith(".") for part in Path(path).parts)
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda candidate: (candidate.lower().count("/"), candidate.lower()))
    return candidates[0]


def resolve_image_model_assets(
    model_id: str,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> ImageModelAssets:
    """
    Resolve a local path or Hugging Face repo into stable-diffusion.cpp assets.

    The cpp backend only accepts GGUF image models. The model can be either a local
    `.gguf` file, a directory containing GGUF files, or a repository that hosts GGUF
    image weights.
    """
    cache_dir = cache_dir or Config().cache_dir
    # Accept pasted shell/Explorer paths like "C:\\path\\model.gguf" by removing
    # surrounding quotes before path/repo resolution.
    model_id = model_id.strip().strip('"').strip("'")

    if _is_local_model_path(model_id):
        local_path = Path(model_id).expanduser().resolve()
        logger.info("Using local image model path: %s", local_path)
        if local_path.is_file():
            if local_path.suffix.lower() not in MODEL_EXTENSIONS:
                raise ValueError("Unsupported local image model format. Expected a .gguf file.")
            is_valid, reason = validate_image_gguf_compatibility(local_path)
            if not is_valid:
                raise ValueError(
                    f"Selected image model is not compatible with stable-diffusion.cpp: {reason} "
                    f"(file: {local_path.name})"
                )
            return ImageModelAssets(repo_id=None, model_path=local_path)

        if local_path.is_dir():
            gguf_files = sorted(local_path.rglob("*.gguf"))
            if not gguf_files:
                raise ValueError("Unsupported local image model directory. Expected at least one .gguf file.")
            return ImageModelAssets(repo_id=None, model_path=gguf_files[0].resolve())

        raise ValueError("Unsupported local image model path. Expected a .gguf file or directory containing one.")

    resolved_id = MODEL_ALIASES.get(model_id, resolve_model_id(model_id))
    if model_id.lower() == "vae" or resolved_id.lower().endswith("/vae"):
        raise ValueError(
            "The selected image model is a VAE component, not a full stable-diffusion.cpp generation model. "
            "Choose a GGUF diffusion model instead."
        )

    # When users pass a plain alias/repo (for example: "ideation"), prefer any
    # already-downloaded compatible GGUF for that repo before pulling another quant.
    if not force_download:
        try:
            from oprel.downloader.hub import _find_cached_model_for_repo

            cached_model = _find_cached_model_for_repo(resolved_id, cache_dir, quantization=None)
            if cached_model is not None:
                is_valid, reason = validate_image_gguf_compatibility(cached_model)
                if is_valid:
                    logger.info(
                        "Using cached image model variant for '%s': %s",
                        resolved_id,
                        cached_model.name,
                    )
                    return ImageModelAssets(repo_id=resolved_id, model_path=cached_model)

                logger.info(
                    "Cached image model variant is incompatible and will be ignored: %s (%s)",
                    cached_model.name,
                    reason,
                )
        except Exception as exc:
            logger.debug("Could not reuse cached image model for '%s': %s", resolved_id, exc)

    logger.info("Resolving image model assets for repo: %s", resolved_id)
    repo_files = list_repo_files(resolved_id)

    main_model_file = _select_main_model_file(repo_files)
    if main_model_file is None:
        raise FileNotFoundError(
            f"Could not find a supported .gguf image model file in '{resolved_id}'. "
            "The cpp backend only supports GGUF image weights."
        )

    model_path = _download_file(resolved_id, main_model_file, cache_dir, force_download)
    is_valid, reason = validate_image_gguf_compatibility(model_path)
    if not is_valid:
        raise ValueError(
            f"Downloaded GGUF is not compatible with stable-diffusion.cpp: {reason} "
            f"(file: {model_path.name}, repo: {resolved_id})"
        )
    logger.info("Using GGUF image model: %s", model_path.name)
    return ImageModelAssets(repo_id=resolved_id, model_path=model_path)
