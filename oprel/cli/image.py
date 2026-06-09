"""Image generation commands for Oprel."""

from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

from oprel.client_api import generate_image as client_generate_image
from oprel.core.config import Config
from oprel.runtime.binaries.installer import ensure_binary
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def cmd_gen_image(args: argparse.Namespace) -> int:
    """Generate an image using stable-diffusion.cpp."""
    try:
        output_path = Path(args.output).expanduser().resolve() if args.output else None
        print(f"Generating with stable-diffusion.cpp GGUF model: {args.model}")
        print(f"Prompt: {args.prompt}")
        start = time.time()
        response = client_generate_image(
            args.model,
            args.prompt,
            size=f"{args.width}x{args.height}",
            negative_prompt=args.negative or "",
            steps=args.steps,
            cfg_scale=args.guidance or 7.0,
            seed=getattr(args, "seed", -1),
            sampler=getattr(args, "sampler", None),
        )

        image_entry = response.data[0] if response.data else {}
        image_payload = image_entry.get("b64_json") or image_entry.get("url") or ""
        if image_payload.startswith("data:image"):
            image_payload = image_payload.split(",", 1)[1]

        if output_path is None:
            output_path = Path.cwd() / f"oprel_{int(time.time())}.png"

        output_path.write_bytes(base64.b64decode(image_payload))
        generated_path = output_path
        elapsed = time.time() - start

        print(f"Saved to: {generated_path}")
        print(f"Completed in {elapsed:.1f}s")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled")
        return 1
    except Exception as exc:
        logger.error("Image generation failed: %s", exc, exc_info=True)
        print(f"Error: {exc}")
        return 1


def cmd_setup_image(args: argparse.Namespace) -> int:
    """Download the stable-diffusion.cpp binary for this machine."""
    try:
        config = Config()
        binary_path = ensure_binary(
            backend="stable-diffusion.cpp",
            version=config.image_binary_version,
            binary_dir=config.binary_dir,
            config=config,
        )
        print(f"stable-diffusion.cpp is ready: {binary_path}")
        return 0
    except Exception as exc:
        logger.error("Image setup failed: %s", exc, exc_info=True)
        print(f"Error: {exc}")
        return 1


def cmd_setup_runtimes(args: argparse.Namespace) -> int:
    """Download llama.cpp and stable-diffusion.cpp binaries for this machine."""
    try:
        config = Config()
        targets = [
            ("llama.cpp", config.binary_version),
            ("stable-diffusion.cpp", config.image_binary_version),
        ]

        for backend, version in targets:
            print(f"Preparing {backend} ({version})...")
            binary_path = ensure_binary(
                backend=backend,
                version=version,
                binary_dir=config.binary_dir,
                config=config,
            )
            print(f"{backend} is ready: {binary_path}")
        return 0
    except Exception as exc:
        logger.error("Runtime setup failed: %s", exc, exc_info=True)
        print(f"Error: {exc}")
        return 1
