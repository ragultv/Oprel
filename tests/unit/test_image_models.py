"""Unit tests for stable-diffusion.cpp model compatibility and architecture routing."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from oprel.downloader.aliases import OFFICIAL_REPOS
from oprel.downloader.image_hub import validate_image_gguf_compatibility
from oprel.runtime.image_generation import StableDiffusionCppRunner, ImageGenerationParams, ImageModelAssets
from oprel.core.config import Config


def test_mlue_removed_from_image_models():
    """Ensure Mlue is completely removed from text-to-image models."""
    image_models = OFFICIAL_REPOS.get("text-to-image", {})
    assert "Mlue-4b" not in image_models
    assert "mlue-4b" not in image_models
    assert not any("mlue" in k.lower() for k in image_models.keys())
    assert not any("bennydaball" in v.lower() for v in image_models.values())


def test_llm_architecture_rejected_by_image_validator(tmp_path):
    """Test that text LLM architectures (like qwen3, llama) are rejected."""
    fake_gguf = tmp_path / "fake_model.gguf"
    fake_gguf.write_bytes(b"dummy")

    with patch("oprel.downloader.image_hub._read_gguf_metadata_keys") as mock_read:
        # Test qwen3
        mock_read.return_value = ("qwen3", {"general.architecture", "general.name"})
        is_valid, reason = validate_image_gguf_compatibility(fake_gguf)
        assert is_valid is False
        assert "text LLM" in reason

        # Test llama
        mock_read.return_value = ("llama", {"general.architecture"})
        is_valid, reason = validate_image_gguf_compatibility(fake_gguf)
        assert is_valid is False
        assert "text LLM" in reason

        # Test valid flux architecture
        mock_read.return_value = ("flux", {"general.architecture"})
        is_valid, reason = validate_image_gguf_compatibility(fake_gguf)
        assert is_valid is True
        assert reason is None


def test_build_command_flux_routes_to_diffusion_model(tmp_path):
    """Test that a FLUX model is automatically invoked with --diffusion-model, not --model."""
    runner = StableDiffusionCppRunner(config=Config())
    fake_flux = tmp_path / "flux-test.gguf"
    fake_flux.write_bytes(b"dummy")

    assets = ImageModelAssets(
        repo_id="test/flux",
        model_path=fake_flux,
    )
    params = ImageGenerationParams(
        model=str(fake_flux),
        prompt="a scenic mountain",
    )

    with patch("oprel.downloader.image_hub._read_gguf_metadata_keys", return_value=("flux", {"general.architecture"})):
        cmd = runner._build_command(
            binary_path=Path("sd-cli.exe"),
            assets=assets,
            params=params,
            output_path=tmp_path / "out.png",
        )

        assert "--diffusion-model" in cmd
        assert "--model" not in cmd
        assert str(fake_flux) in cmd


def test_build_command_rejects_llm(tmp_path):
    """Test that attempting to build command with an LLM raises a clear ValueError."""
    runner = StableDiffusionCppRunner(config=Config())
    fake_llm = tmp_path / "qwen-text.gguf"
    fake_llm.write_bytes(b"dummy")

    assets = ImageModelAssets(
        repo_id="test/llm",
        model_path=fake_llm,
    )
    params = ImageGenerationParams(
        model=str(fake_llm),
        prompt="a scenic mountain",
    )

    with patch("oprel.downloader.image_hub._read_gguf_metadata_keys", return_value=("qwen3", {"general.architecture"})):
        with pytest.raises(ValueError, match="is a text LLM"):
            runner._build_command(
                binary_path=Path("sd-cli.exe"),
                assets=assets,
                params=params,
                output_path=tmp_path / "out.png",
            )

