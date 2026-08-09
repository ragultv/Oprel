"""Hugging Face integration for image-generation model assets."""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError

from oprel.core.config import Config
from oprel.downloader.aliases import MODEL_ALIASES, resolve_model_id
from oprel.downloader.hub import (
    _enable_fast_transfer,
    _socket_timeout,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    BACKOFF_FACTOR,
)
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
    _enable_fast_transfer()
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            with _socket_timeout(DEFAULT_TIMEOUT[1]):
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(cache_dir),
                    resume_download=True,
                    force_download=force_download,
                )
            return Path(path)
        except (requests.exceptions.RequestException, TimeoutError, ConnectionError, socket.timeout) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait_time = BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(
                    f"Download attempt {attempt + 1}/{MAX_RETRIES} failed for {filename}: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {MAX_RETRIES} download attempts failed for {filename}")
        except HfHubHTTPError as e:
            if getattr(e, "response", None) is not None and e.response.status_code in [401, 403, 404]:
                raise
            elif getattr(e, "response", None) is not None and e.response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                last_error = e
                wait_time = BACKOFF_FACTOR * (2 ** attempt)
                logger.warning(
                    f"Server error {e.response.status_code} for {filename}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(
        f"Failed to download {repo_id}/{filename} after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


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


@dataclass
class ComponentSpec:
    vae_repo: str
    vae_filename: str
    vae_fallback_repos: tuple[str, ...] = ()
    llm_repo: Optional[str] = None
    llm_filename: Optional[str] = None
    clip_l_repo: Optional[str] = None
    clip_l_filename: Optional[str] = None
    clip_g_repo: Optional[str] = None
    clip_g_filename: Optional[str] = None
    t5xxl_repo: Optional[str] = None
    t5xxl_filename: Optional[str] = None


_COMPONENT_SPECS = {
    "lumina2": ComponentSpec(
        vae_repo="second-state/FLUX.1-dev-GGUF",
        vae_filename="ae.safetensors",
        vae_fallback_repos=(
            "camenduru/FLUX.1-dev",
            "SicariusSicariiStuff/FLUX.1-dev",
            "ffxvs/vae-flux",
            "black-forest-labs/FLUX.1-schnell",
        ),
        llm_repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        llm_filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    ),
}

_COMPONENT_REPO_OVERRIDES: dict[str, str] = {
    "unsloth/z-image-turbo-gguf": "lumina2",
    "leejet/z-image-turbo-gguf": "lumina2",
}


def _get_component_spec(arch: str, repo_id: str | None) -> ComponentSpec | None:
    if arch and arch.lower() in _COMPONENT_SPECS:
        return _COMPONENT_SPECS[arch.lower()]
    if repo_id and repo_id.lower() in _COMPONENT_REPO_OVERRIDES:
        arch_key = _COMPONENT_REPO_OVERRIDES[repo_id.lower()]
        return _COMPONENT_SPECS.get(arch_key)
    return None


def _resolve_component_assets(
    main_path: Path,
    spec: ComponentSpec,
    cache_dir: Path,
    force_download: bool,
    repo_id: str | None,
) -> ImageModelAssets:
    vae_path: Path | None = None
    llm_path: Path | None = None
    clip_l_path: Path | None = None
    clip_g_path: Path | None = None
    t5xxl_path: Path | None = None

    if spec.vae_filename:
        candidates = [
            main_path.parent / spec.vae_filename,
            main_path.parent / "vae" / spec.vae_filename,
            main_path.parent / "ae.safetensors",
            main_path.parent / "ae.sft",
        ]
        for candidate in candidates:
            if candidate.exists():
                vae_path = candidate
                break
        if not vae_path and spec.vae_repo:
            repos_to_try = [spec.vae_repo, *spec.vae_fallback_repos]
            last_err = None
            for r in repos_to_try:
                try:
                    vae_path = _download_file(r, spec.vae_filename, cache_dir, force_download)
                    break
                except Exception as exc:
                    last_err = exc
                    logger.warning("Could not download VAE '%s' from repo '%s': %s. Trying fallback candidate...", spec.vae_filename, r, exc)
            if not vae_path and last_err:
                raise last_err

    if spec.llm_filename:
        candidates = [
            main_path.parent / spec.llm_filename,
            main_path.parent / "text_encoders" / spec.llm_filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                llm_path = candidate
                break
        if not llm_path and spec.llm_repo:
            llm_path = _download_file(spec.llm_repo, spec.llm_filename, cache_dir, force_download)

    if spec.clip_l_filename:
        candidates = [main_path.parent / spec.clip_l_filename]
        for candidate in candidates:
            if candidate.exists():
                clip_l_path = candidate
                break
        if not clip_l_path and spec.clip_l_repo:
            clip_l_path = _download_file(spec.clip_l_repo, spec.clip_l_filename, cache_dir, force_download)

    if spec.clip_g_filename:
        candidates = [main_path.parent / spec.clip_g_filename]
        for candidate in candidates:
            if candidate.exists():
                clip_g_path = candidate
                break
        if not clip_g_path and spec.clip_g_repo:
            clip_g_path = _download_file(spec.clip_g_repo, spec.clip_g_filename, cache_dir, force_download)

    if spec.t5xxl_filename:
        candidates = [main_path.parent / spec.t5xxl_filename]
        for candidate in candidates:
            if candidate.exists():
                t5xxl_path = candidate
                break
        if not t5xxl_path and spec.t5xxl_repo:
            t5xxl_path = _download_file(spec.t5xxl_repo, spec.t5xxl_filename, cache_dir, force_download)

    return ImageModelAssets(
        repo_id=repo_id,
        model_path=None,
        diffusion_model_path=main_path,
        vae_path=vae_path,
        clip_l_path=clip_l_path,
        clip_g_path=clip_g_path,
        t5xxl_path=t5xxl_path,
        llm_path=llm_path,
    )


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
            try:
                arch, _ = _read_gguf_metadata_keys(local_path)
            except Exception:
                arch = ""
            spec = _get_component_spec(arch, None)
            if spec:
                logger.info("Local image model '%s' uses component architecture '%s', resolving companion assets", local_path.name, arch)
                return _resolve_component_assets(local_path, spec, cache_dir, force_download, repo_id=None)
            return ImageModelAssets(repo_id=None, model_path=local_path)

        if local_path.is_dir():
            gguf_files = sorted(local_path.rglob("*.gguf"))
            if not gguf_files:
                raise ValueError("Unsupported local image model directory. Expected at least one .gguf file.")
            main_gguf = gguf_files[0].resolve()
            try:
                arch, _ = _read_gguf_metadata_keys(main_gguf)
            except Exception:
                arch = ""
            spec = _get_component_spec(arch, None)
            if spec:
                logger.info("Local image model directory uses component architecture '%s', resolving companion assets", arch)
                return _resolve_component_assets(main_gguf, spec, cache_dir, force_download, repo_id=None)
            return ImageModelAssets(repo_id=None, model_path=main_gguf)

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
                    try:
                        arch, _ = _read_gguf_metadata_keys(cached_model)
                    except Exception:
                        arch = ""
                    spec = _get_component_spec(arch, resolved_id)
                    if spec:
                        logger.info(
                            "Using cached component image model variant for '%s': %s (arch: %s)",
                            resolved_id,
                            cached_model.name,
                            arch or resolved_id,
                        )
                        return _resolve_component_assets(cached_model, spec, cache_dir, force_download, repo_id=resolved_id)
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
    try:
        arch, _ = _read_gguf_metadata_keys(model_path)
    except Exception:
        arch = ""
    spec = _get_component_spec(arch, resolved_id)
    if spec:
        logger.info("Downloaded GGUF '%s' uses component architecture '%s', resolving companion assets", model_path.name, arch or resolved_id)
        return _resolve_component_assets(model_path, spec, cache_dir, force_download, repo_id=resolved_id)
    logger.info("Using GGUF image model: %s", model_path.name)
    return ImageModelAssets(repo_id=resolved_id, model_path=model_path)
