"""Regression: a model larger than free VRAM must not be launched with a 512-token context."""

from oprel.runtime.backends.llama_cpp import vram_context_cap
from oprel.runtime.kv_cache import recommend_max_context

DIMS = dict(hidden_dim=3072, num_layers=32, num_kv_heads=8, num_attention_heads=24, kv_cache_type="f16")


def test_no_gpu_means_no_vram_cap():
    assert vram_context_cap(vram_gb=0, model_size_gb=3.0, **DIMS) is None


def test_model_larger_than_vram_budget_is_not_capped_to_the_512_floor():
    # 3.2 GB of weights on a GPU with 3.0 GB free: recommend_max_context alone returns its 512 floor
    assert recommend_max_context(available_vram_gb=3.0, model_size_gb=3.2, **DIMS) == 512
    assert vram_context_cap(vram_gb=3.0, model_size_gb=3.2, **DIMS) is None


def test_weights_that_barely_fit_leave_no_room_for_kv_cache_so_they_offload():
    # 3.0 GB of weights on 3.4 GB free is above the 85% budget (2.89 GB)
    assert vram_context_cap(vram_gb=3.4, model_size_gb=3.0, **DIMS) is None


def test_model_that_fits_still_gets_a_kv_based_cap():
    cap = vram_context_cap(vram_gb=12.0, model_size_gb=3.0, **DIMS)
    assert cap is not None and cap >= 4096


def test_ctx_size_env_var_counts_as_an_explicit_user_choice(monkeypatch):
    from oprel.core.config import Config

    monkeypatch.setenv("OPREL_CTX_SIZE", "12288")
    cfg = Config()
    assert cfg.ctx_size == 12288 and "ctx_size" in cfg.model_fields_set  # disables the automatic tier default and VRAM cap
    assert Config(ctx_size=4096).ctx_size == 4096  # an explicit argument still wins over the environment


def test_invalid_ctx_size_env_var_is_ignored(monkeypatch):
    from oprel.core.config import Config

    for bad in ("abc", "100", "999999999", ""):
        monkeypatch.setenv("OPREL_CTX_SIZE", bad)
        cfg = Config()
        assert cfg.ctx_size == 8192 and "ctx_size" not in cfg.model_fields_set
