"""One-shot internal model calls (Video Studio) must not create chats or long-term memories."""

from __future__ import annotations

import asyncio

import pytest

from oprel.server.schemas.providers import ProviderChatRequest
from oprel.server.services import providers
from oprel.server.services.generation import GenerateParams, generate_text

PROVIDER = {"id": "p", "name": "P", "type": "openai", "api_key": "k", "base_url": "http://x", "enabled": True, "enabled_model_ids": ["m"]}


class Spy:
    def __init__(self, monkeypatch):
        self.created, self.messages, self.memories = [], [], []
        monkeypatch.setattr(providers.db, "get_provider", lambda pid: PROVIDER)
        monkeypatch.setattr(providers.db, "create_conversation", lambda **kw: self.created.append(kw) or "chat_new")
        monkeypatch.setattr(providers.db, "add_message", lambda *a, **k: self.messages.append(a))
        monkeypatch.setattr(providers.db, "add_inference_log", lambda **k: None)

        async def memory(conv_id, user, assistant):
            self.memories.append(conv_id)

        async def fake_call(*a, **k):
            return "a reply"

        monkeypatch.setattr(providers, "_record_turn_supermemory", memory)
        monkeypatch.setattr(providers, "_call_openai", fake_call)


def request(**kw):
    return ProviderChatRequest(model="m", messages=[{"role": "user", "content": "Plan a video of about 30 seconds"}], stream=False, **kw)


def test_control_normal_provider_chat_is_saved_and_remembered(monkeypatch):
    spy = Spy(monkeypatch)
    result = asyncio.run(providers.provider_chat_proxy("p", request()))
    assert result.conversation_id == "chat_new" and spy.created and spy.messages and spy.memories


def test_ephemeral_provider_chat_creates_no_conversation_messages_or_memories(monkeypatch):
    spy = Spy(monkeypatch)
    result = asyncio.run(providers.provider_chat_proxy("p", request(ephemeral=True)))
    assert result.text == "a reply" and result.conversation_id.startswith("ephemeral_")
    assert spy.created == [] and spy.messages == [] and spy.memories == []


def test_generate_text_forwards_the_ephemeral_flag_to_providers(monkeypatch):
    seen = {}

    async def fake_proxy(pid, body):
        seen["ephemeral"] = body.ephemeral
        return type("R", (), {"text": "x", "model_id": "m", "conversation_id": "ephemeral_1", "message_count": 1})()

    monkeypatch.setattr("oprel.server.services.generation.db.get_provider", lambda pid: PROVIDER)
    monkeypatch.setattr(providers, "provider_chat_proxy", fake_proxy)
    params = GenerateParams(model_id="p::m", prompt="hi", max_tokens=10, temperature=0.1, top_p=1, top_k=1, repeat_penalty=1, stream=False,
                            images=None, conversation_id=None, system_prompt=None, reset_conversation=True, thinking=False, rag=False, ephemeral=True)
    asyncio.run(generate_text(params))
    assert seen["ephemeral"] is True


def test_default_generate_params_are_not_ephemeral():
    params = GenerateParams(model_id="m", prompt="hi", max_tokens=1, temperature=0, top_p=1, top_k=1, repeat_penalty=1, stream=False,
                            images=None, conversation_id=None, system_prompt=None, reset_conversation=False, thinking=False, rag=False)
    assert params.ephemeral is False


def test_video_studio_model_calls_are_ephemeral(monkeypatch):
    from oprel.video import ai

    seen = {}

    async def fake_generate(params):
        seen["ephemeral"] = params.ephemeral
        from oprel.server.services.generation import GenerateResult
        return GenerateResult(text="ok", model_id="m", conversation_id="ephemeral_x", message_count=1)

    monkeypatch.setattr("oprel.server.services.generation.generate_text", fake_generate)
    assert asyncio.run(ai._ask_model("m", "sys", "user", 100)) == "ok"
    assert seen["ephemeral"] is True


@pytest.mark.parametrize("raw,expected", [("q4", "q4_0"), ("Q4_0", "q4_0"), ("q8_0", "q8_0"), ("iq4_nl", "iq4_nl")])
def test_kv_cache_type_env_var(monkeypatch, raw, expected):
    from oprel.core.config import Config

    monkeypatch.setenv("OPREL_KV_CACHE_TYPE", raw)
    cfg = Config()
    assert cfg.kv_cache_type == expected and "kv_cache_type" in cfg.model_fields_set


@pytest.mark.parametrize("raw", ["q1", "1bit", "int2", "garbage", ""])
def test_unsupported_kv_cache_types_are_ignored_not_passed_to_llama_cpp(monkeypatch, raw):
    from oprel.core.config import Config

    monkeypatch.setenv("OPREL_KV_CACHE_TYPE", raw)
    assert Config().kv_cache_type == "auto"
