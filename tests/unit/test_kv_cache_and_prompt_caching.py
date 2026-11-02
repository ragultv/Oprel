"""
Unit tests for TurboQuant KV cache and Prompt Caching features.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oprel.core.config import Config
from oprel.runtime.backends.llama_cpp import LlamaCppBackend, _resolve_kv_cache_type
from oprel.runtime.kv_cache import (
    KV_CACHE_BYTES,
    estimate_kv_cache_size,
    recommend_max_context,
)


def test_kv_cache_bytes_mapping():
    """Verify that turboquant and iq4_nl are mapped to extreme 4-bit (0.5 bytes/element)."""
    assert "turboquant" in KV_CACHE_BYTES
    assert KV_CACHE_BYTES["turboquant"] == 0.5
    assert "iq4_nl" in KV_CACHE_BYTES
    assert KV_CACHE_BYTES["iq4_nl"] == 0.5


def test_estimate_kv_cache_size_turboquant():
    """Verify 4x memory savings with TurboQuant compared to FP16."""
    batch_size = 1
    context_length = 8192
    hidden_dim = 4096
    num_layers = 32

    fp16_est = estimate_kv_cache_size(
        batch_size=batch_size,
        context_length=context_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        kv_cache_type="f16",
    )

    turbo_est = estimate_kv_cache_size(
        batch_size=batch_size,
        context_length=context_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        kv_cache_type="turboquant",
    )

    # 0.5 bytes vs 2.0 bytes -> Exactly 4x reduction
    assert turbo_est.size_bytes * 4 == fp16_est.size_bytes
    assert turbo_est.size_gb == pytest.approx(fp16_est.size_gb / 4.0, rel=1e-3)


def test_resolve_kv_cache_type_turboquant():
    """Verify that _resolve_kv_cache_type correctly translates turboquant aliases."""
    cfg_turbo = Config(kv_cache_type="turboquant")
    assert _resolve_kv_cache_type(cfg_turbo, "Q4_K_M", is_vision=False) == "q4_0"

    cfg_extreme = Config(kv_cache_type="extreme")
    assert _resolve_kv_cache_type(cfg_extreme, "Q8_0", is_vision=False) == "q4_0"

    cfg_auto = Config(kv_cache_type="auto")
    # auto never picks the lossy q4_0 KV cache; q4_0 is opt-in only.
    assert _resolve_kv_cache_type(cfg_auto, "Q4_K_M", is_vision=False) == "q8_0"
    assert _resolve_kv_cache_type(cfg_auto, "Q8_0", is_vision=False) == "q8_0"
    assert _resolve_kv_cache_type(cfg_auto, "Q4_K_M", is_vision=True) == "f16"


def test_llama_cpp_build_command_turboquant_and_prompt_cache(tmp_path):
    """Verify that llama-server CLI flags include TurboQuant KV cache types and prompt cache settings."""
    prompt_dir = tmp_path / "prompt_cache"
    config = Config(
        kv_cache_type="turboquant",
        prompt_cache_enabled=True,
        prompt_cache_dir=prompt_dir,
        cache_reuse_tokens=256,
        flash_attention=True,
    )

    dummy_model = tmp_path / "dummy_model.gguf"
    dummy_model.write_bytes(b"dummy")

    backend = LlamaCppBackend(
        binary_path=Path("llama-server"),
        model_path=dummy_model,
        config=config,
    )

    with patch("oprel.runtime.backends.llama_cpp.detect_gpu", return_value=None):
        cmd = backend.build_command(port=8080)

    # KV Cache flags
    assert "--cache-type-k" in cmd
    k_idx = cmd.index("--cache-type-k")
    assert cmd[k_idx + 1] == "q4_0"

    assert "--cache-type-v" in cmd
    v_idx = cmd.index("--cache-type-v")
    assert cmd[v_idx + 1] == "q4_0"

    # Prompt Cache flags
    assert "--cache-reuse" in cmd
    reuse_idx = cmd.index("--cache-reuse")
    assert cmd[reuse_idx + 1] == "256"

    assert "--slot-save-path" in cmd
    slot_idx = cmd.index("--slot-save-path")
    assert cmd[slot_idx + 1] == str(prompt_dir)
    assert prompt_dir.exists()


def test_http_client_cache_prompt_payload():
    """Verify HTTP client sends cache_prompt: True by default."""
    from oprel.client.http import HTTPClient

    client = HTTPClient(port=8080)

    with patch.object(client.session, "post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"text": "Hello"}]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client.generate(prompt="Test prompt", stream=False)

        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload.get("cache_prompt") is True
