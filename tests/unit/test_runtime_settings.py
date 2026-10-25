"""Model runtime settings (context window, KV cache type): file, environment and API."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from oprel.core.config import Config, read_runtime_settings, runtime_settings_path, write_runtime_settings


def test_defaults_are_automatic():
    cfg = Config()
    assert cfg.ctx_size == 8192 and "ctx_size" not in cfg.model_fields_set
    assert cfg.kv_cache_type == "auto" and "kv_cache_type" not in cfg.model_fields_set


def test_saved_settings_apply_to_every_new_config_without_a_restart():
    write_runtime_settings(8192, "q4")
    cfg = Config()  # what Model() builds on each model load
    assert cfg.kv_cache_type == "q4_0" and cfg.ctx_size == 8192
    assert {"ctx_size", "kv_cache_type"} <= cfg.model_fields_set  # explicit => no automatic tier default / VRAM cap


def test_priority_is_argument_then_environment_then_file(monkeypatch):
    write_runtime_settings(8192, "q8_0")
    assert Config().ctx_size == 8192
    monkeypatch.setenv("OPREL_CTX_SIZE", "12288")
    monkeypatch.setenv("OPREL_KV_CACHE_TYPE", "q4")
    cfg = Config()
    assert cfg.ctx_size == 12288 and cfg.kv_cache_type == "q4_0"
    assert Config(ctx_size=4096, kv_cache_type="f16").ctx_size == 4096


def test_auto_clears_a_saved_choice():
    write_runtime_settings(8192, "q4_0")
    write_runtime_settings(None, "auto")
    assert read_runtime_settings() == {} and Config().kv_cache_type == "auto"


@pytest.mark.parametrize("ctx,kv", [(100, None), (999_999_999, None), (None, "1bit"), (None, "q2")])
def test_invalid_values_are_rejected_and_nothing_is_written(ctx, kv):
    with pytest.raises(ValueError):
        write_runtime_settings(ctx, kv)
    assert not runtime_settings_path().exists()


def test_a_corrupt_settings_file_is_ignored():
    runtime_settings_path().write_text("{not json", encoding="utf-8")
    assert read_runtime_settings() == {} and Config().kv_cache_type == "auto"
    runtime_settings_path().write_text(json.dumps({"ctx_size": "abc", "kv_cache_type": "bogus"}), encoding="utf-8")
    assert read_runtime_settings() == {}


@pytest.fixture
def client():
    from oprel.server.app import app
    from oprel.utils.auth import get_api_token

    return TestClient(app, base_url="http://127.0.0.1:11435", headers={"Authorization": f"Bearer {get_api_token()}"})


def test_api_roundtrip_and_validation(client):
    r = client.get("/v1/system/runtime-settings")
    assert r.status_code == 200 and r.json()["kv_cache_type"] == "auto" and r.json()["ctx_size"] is None
    assert "q4_0" in r.json()["kv_cache_types"] and "q1" not in r.json()["kv_cache_types"]

    r = client.put("/v1/system/runtime-settings", json={"ctx_size": 8192, "kv_cache_type": "q4_0"})
    assert r.status_code == 200 and r.json() == {"ctx_size": 8192, "kv_cache_type": "q4_0", "unloaded": False}
    assert client.get("/v1/system/runtime-settings").json()["kv_cache_type"] == "q4_0"

    assert client.put("/v1/system/runtime-settings", json={"ctx_size": 10}).status_code == 400
    assert client.put("/v1/system/runtime-settings", json={"kv_cache_type": "1bit"}).status_code == 400
    assert client.get("/v1/system/runtime-settings").json()["ctx_size"] == 8192  # a rejected request changes nothing


def test_api_unloads_the_model_so_it_reloads_with_the_new_settings(client, monkeypatch):
    unloaded = []
    monkeypatch.setattr("oprel.server.services.models.unload_model", lambda mid: unloaded.append(mid) or {"success": True})
    r = client.put("/v1/system/runtime-settings", json={"ctx_size": 8192, "kv_cache_type": "q4_0", "reload_model_id": "some/model"})
    assert r.json()["unloaded"] is True and unloaded == ["some/model"]

    def not_loaded(mid):
        raise KeyError(mid)

    monkeypatch.setattr("oprel.server.services.models.unload_model", not_loaded)
    assert client.put("/v1/system/runtime-settings", json={"ctx_size": 8192, "reload_model_id": "x"}).json()["unloaded"] is False
