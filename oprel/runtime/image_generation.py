"""stable-diffusion.cpp image generation runtime."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from oprel.core.config import Config
from oprel.downloader.image_hub import ImageModelAssets, resolve_image_model_assets
from oprel.runtime.binaries.installer import ensure_binary
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImageGenerationParams:
    model: str
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str | None = None
    output_path: Path | None = None
    force_download: bool = False


def _is_cuda_oom_error(details: str) -> bool:
    lowered = details.lower()
    return "cuda error: out of memory" in lowered or "out of memory" in lowered and "cuda" in lowered


class StableDiffusionCppRunner:
    """Thin wrapper around the stable-diffusion.cpp CLI binary."""

    def __init__(self, config: Optional[Config] = None, prefer_cpu_binary: bool = False):
        self.config = config or Config()
        self.prefer_cpu_binary = prefer_cpu_binary

    def _ensure_binary(self) -> Path:
        version = getattr(self.config, "image_binary_version", "latest")
        return ensure_binary(
            backend="stable-diffusion.cpp",
            version=version,
            binary_dir=self.config.binary_dir,
            config=self.config,
            prefer_cpu=self.prefer_cpu_binary,
        )

    def _build_command(
        self,
        binary_path: Path,
        assets: ImageModelAssets,
        params: ImageGenerationParams,
        output_path: Path,
    ) -> list[str]:
        cmd = [
            str(binary_path),
            "--output",
            str(output_path),
            "--prompt",
            params.prompt,
            "--negative-prompt",
            params.negative_prompt,
            "--width",
            str(params.width),
            "--height",
            str(params.height),
            "--steps",
            str(params.steps),
            "--cfg-scale",
            str(params.cfg_scale),
            "--seed",
            str(params.seed),
        ]

        if params.sampler:
            cmd.extend(["--sampling-method", params.sampler])

        if assets.model_path:
            cmd.extend(["--model", str(assets.model_path)])
        else:
            if not assets.diffusion_model_path:
                raise ValueError("No model assets available for stable-diffusion.cpp")
            cmd.extend(["--diffusion-model", str(assets.diffusion_model_path)])
            if assets.vae_path:
                cmd.extend(["--vae", str(assets.vae_path)])
            if assets.clip_l_path:
                cmd.extend(["--clip_l", str(assets.clip_l_path)])
            if assets.clip_g_path:
                cmd.extend(["--clip_g", str(assets.clip_g_path)])
            if assets.t5xxl_path:
                cmd.extend(["--t5xxl", str(assets.t5xxl_path)])
            if assets.llm_path:
                cmd.extend(["--llm", str(assets.llm_path)])
            if assets.llm_vision_path:
                cmd.extend(["--llm_vision", str(assets.llm_vision_path)])

        return cmd

    def generate(self, params: ImageGenerationParams) -> Path:
        return self._generate_internal(params, progress_callback=None)

    def generate_with_progress(
        self,
        params: ImageGenerationParams,
        progress_callback: Callable[[float, str], None],
    ) -> Path:
        return self._generate_internal(params, progress_callback=progress_callback)

    def _generate_once(
        self,
        params: ImageGenerationParams,
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> Path:
        binary_path = self._ensure_binary()
        assets = resolve_image_model_assets(
            params.model,
            cache_dir=self.config.cache_dir,
            force_download=params.force_download,
        )

        output_path = params.output_path
        if output_path is None:
            generated_dir = self.config.cache_dir / "generated_images"
            generated_dir.mkdir(parents=True, exist_ok=True)
            output_path = generated_dir / f"oprel_{int(time.time())}.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self._build_command(binary_path, assets, params, output_path)

        logger.info("Running stable-diffusion.cpp for image generation")
        logger.debug("stable-diffusion.cpp command: %s", " ".join(cmd))

        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        if progress_callback is None:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60 * 20,
                creationflags=creation_flags,
            )
            stderr_text = result.stderr or ""
            stdout_text = result.stdout or ""
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creation_flags,
            )
            merged_lines: list[str] = []
            progress_re = re.compile(r"\|[^|]*\|\s*(\d+)\/(\d+)")

            try:
                progress_callback(1.0, "Starting generation")
                assert process.stdout is not None
                for line in process.stdout:
                    line = line.rstrip("\n")
                    merged_lines.append(line)

                    match = progress_re.search(line)
                    if match:
                        current = int(match.group(1))
                        total = max(int(match.group(2)), 1)
                        pct = min(100.0, max(1.0, (current / total) * 100.0))
                        progress_callback(pct, f"Sampling step {current}/{total}")
                process.wait(timeout=60 * 20)
            finally:
                if process.stdout:
                    process.stdout.close()

            stderr_text = "\n".join(merged_lines)
            stdout_text = stderr_text

            class _Result:
                def __init__(self, returncode: int):
                    self.returncode = returncode

            result = _Result(process.returncode or 0)

        output_exists = output_path.exists() and output_path.stat().st_size > 0

        if result.returncode != 0:
            # Some stable-diffusion.cpp builds can return a non-zero code even after
            # successfully writing the output image. In that case, prefer the file.
            if output_exists:
                logger.warning(
                    "stable-diffusion.cpp exited with code %s but produced output image: %s",
                    result.returncode,
                    output_path,
                )
                return output_path

            stderr = stderr_text.strip()
            stdout = stdout_text.strip()
            details = stderr or stdout or "Unknown stable-diffusion.cpp error"
            if "get sd version from file failed" in details.lower():
                raise RuntimeError(
                    "stable-diffusion.cpp could not read required SD metadata from the selected GGUF. "
                    "This usually means the quant is not a compatible stable-diffusion.cpp conversion. "
                    "Choose another downloaded quant/model from the dropdown and retry. "
                    f"Backend details: {details}"
                )
            raise RuntimeError(f"stable-diffusion.cpp failed: {details}")

        if not output_exists:
            raise RuntimeError("stable-diffusion.cpp finished without producing an output image")

        return output_path

    def _generate_internal(
        self,
        params: ImageGenerationParams,
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> Path:
        try:
            return self._generate_once(params, progress_callback)
        except RuntimeError as exc:
            message = str(exc)
            if self.prefer_cpu_binary or not _is_cuda_oom_error(message):
                raise

            logger.warning(
                "CUDA image generation ran out of memory; retrying once with the CPU stable-diffusion.cpp binary"
            )
            cpu_runner = StableDiffusionCppRunner(config=self.config, prefer_cpu_binary=True)
            try:
                return cpu_runner._generate_once(params, progress_callback)
            except Exception as cpu_exc:
                raise RuntimeError(
                    f"stable-diffusion.cpp CUDA path ran out of memory and CPU fallback also failed: {cpu_exc}"
                ) from cpu_exc


def generate_image_file(params: ImageGenerationParams, config: Optional[Config] = None) -> Path:
    runner = StableDiffusionCppRunner(config=config)
    return runner.generate(params)


def generate_image_file_with_progress(
    params: ImageGenerationParams,
    progress_callback: Callable[[float, str], None],
    config: Optional[Config] = None,
) -> Path:
    runner = StableDiffusionCppRunner(config=config)
    return runner.generate_with_progress(params, progress_callback)
