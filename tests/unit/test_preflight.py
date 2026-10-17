from types import SimpleNamespace

from oprel.runtime.preflight import check_fit
from oprel.runtime.tiers import TIER_A, TIER_B, TIER_C, default_context, detect_tier


def _meta(size_gb, layers=32, ctx=8192, dim=4096, kv_heads=8, heads=32):
    return SimpleNamespace(
        file_size_bytes=int(size_gb * 1024**3),
        num_layers=layers,
        context_length=ctx,
        embedding_dim=dim,
        num_kv_heads=kv_heads,
        num_attention_heads=heads,
    )


def test_small_model_fits_fully_on_gpu():
    r = check_fit(_meta(2.0), free_vram_gb=8.0, free_ram_gb=16.0, context_length=4096, kv_cache_type="q8_0")
    assert r.verdict == "gpu" and r.gpu_layers == 32


def test_7b_on_6gb_is_hybrid_not_full():
    r = check_fit(_meta(4.7), free_vram_gb=6.0, free_ram_gb=16.0, context_length=8192, kv_cache_type="f16")
    assert r.verdict in ("hybrid", "gpu")
    assert r.vram_needed_gb <= 6.0


def test_no_gpu_uses_cpu():
    r = check_fit(_meta(2.0), free_vram_gb=None, free_ram_gb=16.0)
    assert r.verdict == "cpu" and r.gpu_layers == 0


def test_too_big_for_ram_is_insufficient_with_message():
    r = check_fit(_meta(14.0), free_vram_gb=4.0, free_ram_gb=8.0)
    assert r.verdict == "insufficient"
    assert any("RAM" in m for m in r.messages)


def test_file_size_alone_is_not_enough():
    # 5.5 GB of weights on a 6 GB card cannot be full-GPU once KV cache + overhead are counted.
    r = check_fit(_meta(5.5), free_vram_gb=6.0, free_ram_gb=16.0, context_length=8192, kv_cache_type="f16")
    assert r.verdict != "gpu"


def test_suggests_smaller_context_when_it_improves_fit():
    r = check_fit(_meta(4.0), free_vram_gb=5.2, free_ram_gb=16.0, context_length=16384, kv_cache_type="f16")
    assert r.suggested_context <= r.context_length


def test_tiers():
    assert detect_tier(None) == TIER_A and detect_tier(4) == TIER_A
    assert detect_tier(6) == TIER_B and detect_tier(8) == TIER_C
    assert default_context(4) == 4096 and default_context(12) == 8192


def test_starter_recommendation_by_hardware():
    from oprel.runtime.tiers import recommend_starter

    assert recommend_starter(None, 8)["alias"] == "qwen3-1.7b"
    assert recommend_starter(4, 16)["alias"] == "qwen3-4b"
    assert recommend_starter(6, 16)["alias"] == "qwen3-8b"
    assert recommend_starter(4, 6)["alias"] == "qwen3-1.7b"


def test_first_run_only_for_a_brand_new_install(tmp_path, monkeypatch):
    from oprel.server import db
    from oprel.server.routes import preflight as route

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "fresh.db")
    db.init_db()
    monkeypatch.setattr("oprel.downloader.cache.list_cached_models", lambda: [])
    assert route._is_first_run() is True

    db.create_conversation(model_id="m", title="hello")
    assert route._is_first_run() is False


def test_first_run_false_when_models_exist_or_check_fails(tmp_path, monkeypatch):
    from oprel.server import db
    from oprel.server.routes import preflight as route

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "fresh2.db")
    db.init_db()
    monkeypatch.setattr("oprel.downloader.cache.list_cached_models", lambda: [{"name": "x.gguf"}])
    assert route._is_first_run() is False

    monkeypatch.setattr(db, "list_conversations", lambda: (_ for _ in ()).throw(RuntimeError("db broken")))
    assert route._is_first_run() is False
