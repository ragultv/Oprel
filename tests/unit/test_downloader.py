"""
Unit tests for model download progress tracking and speech model removal.
"""

import time
from unittest.mock import MagicMock, patch
from oprel.downloader.hub import RobustDownloadProgress, _create_progress_class, download_model, download_model_with_progress
from oprel.server.services.speech import get_speech_models_status
from oprel.models.model_types import detect_model_type


def test_robust_download_progress_callback():
    """Verify RobustDownloadProgress emits progress to callbacks without getting auto-disabled."""
    events = []

    def callback(downloaded: int, total: int):
        events.append((downloaded, total))

    progress_cls = _create_progress_class(callback=callback, show_cli_bar=False)
    progress_bar = progress_cls(total=1000, initial=0, desc="test-model.gguf")

    with progress_bar as p:
        p.update(250)
        time.sleep(0.22)
        p.update(250)
        time.sleep(0.22)
        p.update(500)

    # Initial event (0, 1000) from __enter__, intermediate updates, and final (1000, 1000) from __exit__
    assert len(events) >= 3
    assert events[0] == (0, 1000)
    assert events[-1] == (1000, 1000)
    assert any(ev[0] >= 500 for ev in events)


def test_robust_download_progress_cli_mode():
    """Verify RobustDownloadProgress works in CLI mode without errors."""
    cli_cls = _create_progress_class(callback=None, show_cli_bar=True)
    progress_bar = cli_cls(total=5000, initial=0, desc="cli-model.gguf")

    with progress_bar as p:
        p.update(2000)
        assert p.n == 2000
        p.update(3000)
        assert p.n == 5000


def test_download_model_with_progress_delegation():
    """Verify download_model_with_progress delegates to download_model with show_cli_progress=False."""
    with patch("oprel.downloader.hub.download_model") as mock_dm:
        mock_dm.return_value = "/fake/path/model.gguf"
        cb = lambda d, t: None

        res = download_model_with_progress(
            model_id="test/repo",
            quantization="Q4_K_M",
            progress_callback=cb,
        )

        assert res == "/fake/path/model.gguf"
        mock_dm.assert_called_once_with(
            model_id="test/repo",
            quantization="Q4_K_M",
            cache_dir=None,
            force_download=False,
            progress_callback=cb,
            show_cli_progress=False,
        )


def test_fish_audio_removed_from_speech_models():
    """Verify Fish Audio is completely removed from available speech models."""
    with patch("oprel.server.services.speech.check_whisper_installed", return_value=True):
        models = get_speech_models_status(include_all=True)
        model_ids = [m["id"] for m in models]

        assert "fish-s1-mini" not in model_ids
        assert "kokoro-82m" in model_ids
        assert "whisper-base" in model_ids


def test_detect_model_type_speech():
    """Verify speech model type detection works for kokoro and whisper."""
    assert detect_model_type("kokoro-82m") == "speech"
    assert detect_model_type("whisper-base") == "speech"


def test_storage_resolution_3_tier(tmp_path):
    """Verify 3-tier storage resolution: env var, persistent config, and fallback."""
    from oprel.downloader.cache import get_storage_root, get_storage_info
    import os

    # Tier 1: Environment variable
    env_target = tmp_path / "custom_env_cache"
    with patch.dict(os.environ, {"OPREL_CACHE_DIR": str(env_target)}):
        assert get_storage_root() == env_target.resolve()

    # Tier 2: Persistent config
    cfg_target = tmp_path / "configured_cache"
    cfg_target.mkdir(parents=True, exist_ok=True)
    with patch.dict(os.environ, {}, clear=True):
        with patch("oprel.downloader.cache.Path.home", return_value=tmp_path):
            cfg_file = tmp_path / ".cache" / "oprel_storage.json"
            cfg_file.parent.mkdir(parents=True, exist_ok=True)
            import json
            cfg_file.write_text(json.dumps({"storage_path": str(cfg_target)}), encoding="utf-8")
            assert get_storage_root() == cfg_target.resolve()

    # Storage info structure
    info = get_storage_info()
    assert "storage_path" in info
    assert "free_space_gb" in info
    assert "total_space_gb" in info
    assert info["free_space_gb"] >= 0


def test_updated_binary_versions_and_update_command():
    """Verify latest binary versions are configured and oprel update routes properly."""
    import argparse
    from oprel.core.config import Config
    from oprel.runtime.binaries.registry import resolve_version, get_binary_info
    from oprel.cli.image import cmd_update

    cfg = Config()
    assert cfg.binary_version == "b12189"
    assert resolve_version("llama.cpp", "latest") == "b12189"
    assert resolve_version("stable-diffusion.cpp", "latest") == "master-1056-9234ad8"

    # Verify Windows CUDA URLs exist in registry
    llama_info = get_binary_info("llama.cpp", "b12189", "Windows-AMD64-cuda")
    assert llama_info is not None
    assert "b12189" in llama_info["url"]

    sd_info = get_binary_info("stable-diffusion.cpp", "master-1056-9234ad8", "Windows-AMD64-cuda")
    assert sd_info is not None
    assert "master-1056-9234ad8" in sd_info["url"]

    # Verify cmd_update invokes ensure_binary for targets
    with patch("oprel.cli.image.ensure_binary", return_value="/mock/path") as mock_ensure:
        args = argparse.Namespace(force=True, target="all")
        code = cmd_update(args)
        assert code == 0
        assert mock_ensure.call_count == 2
