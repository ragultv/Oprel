"""
Vision Image Preprocessing for Oprel.

Provides a pixel-budget-based image optimizer that runs before images are
sent to llama.cpp vision models.  This is the *single* implementation;
multimodal.py's preprocess_image_to_bytes() delegates here.

Design decisions:
- Pixel budget (not a per-axis cap) preserves aspect ratio correctly.
- Never upscales: images already within budget are returned as-is.
- Single encode pass: avoids repeated JPEG→PNG→JPEG round-trips.
- LANCZOS resampling for highest quality downscale.
- Explicit, structured [Vision] log lines for easy benchmarking.
- Exception fallback logs clearly and returns original bytes — never silently
  swallows errors and sends an oversized image to the model.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from PIL import Image

from oprel.utils.logging import get_logger

logger = get_logger(__name__)

# Pixel budgets for quick reference:
#   262_144 →  512×512 equivalent
#   409_600 →  640×640 equivalent   ← default
#   589_824 →  768×768 equivalent
# 1_048_576 → 1024×1024 equivalent
DEFAULT_MAX_PIXELS: int = 409_600
DEFAULT_QUALITY: int = 90

# Formats that llama.cpp accepts natively; if the input is already one of
# these we avoid re-encoding unless a resize is needed.
_PASSTHROUGH_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass
class VisionProcessingResult:
    """Carries the processed image bytes and metrics for benchmarking."""

    image_bytes: bytes
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    original_pixels: int
    output_pixels: int
    scale_factor: float
    was_resized: bool
    preprocessing_ms: float


class VisionImageProcessor:
    """
    Reusable image preprocessor for vision model inputs.

    Usage::

        processor = VisionImageProcessor(max_pixels=409_600, quality=90)
        result = processor.process(image_bytes)
        # result.image_bytes is ready to be base64-encoded and sent to llama.cpp

    The processor is model-independent: it knows nothing about Qwen3-VL,
    LLaVA, or any other architecture.
    """

    def __init__(
        self,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        quality: int = DEFAULT_QUALITY,
    ) -> None:
        if max_pixels < 1:
            raise ValueError(f"max_pixels must be >= 1, got {max_pixels}")
        if not (1 <= quality <= 100):
            raise ValueError(f"quality must be 1–100, got {quality}")
        self.max_pixels = max_pixels
        self.quality = quality

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image_input: bytes | str) -> VisionProcessingResult:
        """
        Inspect, (optionally) resize, and re-encode an image.

        Args:
            image_input: Raw image bytes or a file path string.

        Returns:
            VisionProcessingResult with processed bytes and benchmarking data.

        Raises:
            ValueError: If the input cannot be decoded as an image.
        """
        t0 = time.perf_counter()

        img = self._load(image_input)
        orig_w, orig_h = img.size
        orig_pixels = orig_w * orig_h

        logger.info(
            f"[Vision] Original image: {orig_w}x{orig_h} "
            f"({orig_pixels / 1_000_000:.2f} MP)"
        )
        logger.debug(f"[Vision] Target pixel budget: {self.max_pixels}")

        if not self.should_resize(orig_w, orig_h):
            logger.info("[Vision] Within pixel budget. No resize required.")
            # Still need to ensure format compatibility; encode only if needed.
            out_bytes = self._encode_passthrough(img, image_input)
            elapsed = (time.perf_counter() - t0) * 1_000
            logger.debug(f"[Vision] Preprocessing (passthrough): {elapsed:.1f} ms")
            return VisionProcessingResult(
                image_bytes=out_bytes,
                original_width=orig_w,
                original_height=orig_h,
                output_width=orig_w,
                output_height=orig_h,
                original_pixels=orig_pixels,
                output_pixels=orig_pixels,
                scale_factor=1.0,
                was_resized=False,
                preprocessing_ms=elapsed,
            )

        new_w, new_h = self.calculate_dimensions(orig_w, orig_h)
        scale = math.sqrt(self.max_pixels / orig_pixels)

        logger.info(
            f"[Vision] Resized image: {new_w}x{new_h} "
            f"({new_w * new_h / 1_000_000:.2f} MP)"
        )
        logger.info(f"[Vision] Scale factor: {scale:.3f}")

        # Ensure RGB for JPEG compatibility (RGBA/P → composite on white)
        img = self._to_rgb(img)

        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:  # Pillow < 9.1
            resample = Image.LANCZOS  # type: ignore[attr-defined]

        img = img.resize((new_w, new_h), resample)

        out_io = BytesIO()
        img.save(out_io, format="JPEG", quality=self.quality, optimize=True)
        out_bytes = out_io.getvalue()

        elapsed = (time.perf_counter() - t0) * 1_000
        logger.info(f"[Vision] Preprocessing time: {elapsed:.1f} ms")

        return VisionProcessingResult(
            image_bytes=out_bytes,
            original_width=orig_w,
            original_height=orig_h,
            output_width=new_w,
            output_height=new_h,
            original_pixels=orig_pixels,
            output_pixels=new_w * new_h,
            scale_factor=scale,
            was_resized=True,
            preprocessing_ms=elapsed,
        )

    def should_resize(self, width: int, height: int) -> bool:
        """Return True if width × height exceeds the configured pixel budget."""
        return (width * height) > self.max_pixels

    def calculate_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """
        Compute target (width, height) preserving aspect ratio within the budget.

        Never upscales: if the image is already within budget, returns original
        dimensions (callers should check should_resize() first).
        """
        pixels = width * height
        if pixels <= self.max_pixels:
            return width, height
        scale = math.sqrt(self.max_pixels / pixels)
        return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(image_input: bytes | str) -> Image.Image:
        if isinstance(image_input, bytes):
            return Image.open(BytesIO(image_input))
        return Image.open(image_input)

    @staticmethod
    def _to_rgb(img: Image.Image) -> Image.Image:
        """Convert palette/RGBA images to plain RGB, compositing on white."""
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])  # alpha channel as mask
            return bg
        if img.mode in ("LA", "P"):
            converted = img.convert("RGBA")
            bg = Image.new("RGB", converted.size, (255, 255, 255))
            bg.paste(converted, mask=converted.split()[3])
            return bg
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    def _encode_passthrough(
        self, img: Image.Image, original_input: bytes | str
    ) -> bytes:
        """
        Return original bytes if the format is already JPEG/PNG/WebP (and
        no conversion is needed).  Otherwise encode to JPEG once.
        """
        fmt = img.format or ""
        if fmt.upper() in _PASSTHROUGH_FORMATS:
            # Already in a supported format — return original bytes untouched.
            if isinstance(original_input, bytes):
                return original_input
            with open(original_input, "rb") as fh:
                return fh.read()

        # Unsupported source format (e.g. BMP, TIFF) — convert to JPEG once.
        img = self._to_rgb(img)
        out_io = BytesIO()
        img.save(out_io, format="JPEG", quality=self.quality, optimize=True)
        return out_io.getvalue()
