"""
Local llama.cpp context budgeting in the tool-calling adapter: small contexts skip thinking and
cap max_tokens; being cut off raises a clear error instead of returning an empty reply.
"""

import json

import httpx
import pytest

from oprel.server.tools.adapters import ModelAdapterError, OpenAICompatibleAdapter, SamplingParams

TOOLS = [{"type": "function", "function": {"name": "scene_add_objects", "description": "d", "parameters": {}}}]
MESSAGES = [{"role": "user", "content": "draw a table"}]


def _server(n_ctx, prompt_tokens, reply=None, props_status=200):
    """Fake llama-server. Records the chat body it receives."""
    seen = {}

    def handler(req: httpx.Request):
        if req.url.path == "/props":
            if props_status != 200:
                return httpx.Response(props_status)
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": n_ctx}})
        if req.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(prompt_tokens))})
        if req.url.path == "/v1/chat/completions":
            seen["body"] = json.loads(req.content)
            return httpx.Response(200, json=reply or {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
        return httpx.Response(404)

    return httpx.MockTransport(handler), seen


def _adapter(transport, local=True):
    return OpenAICompatibleAdapter(label="Local model qwen", base_url="http://llama/v1", model="qwen",
                                   transport=transport, llama_server_root="http://llama" if local else None)


async def _complete(adapter, max_tokens=4096):
    return await adapter.complete(MESSAGES, TOOLS, SamplingParams(max_tokens=max_tokens))


@pytest.mark.anyio
async def test_small_context_skips_thinking_and_caps_output():
    transport, seen = _server(n_ctx=4096, prompt_tokens=2000)
    turn = await _complete(_adapter(transport))
    assert turn.content == "ok"
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["body"]["max_tokens"] == 2096


@pytest.mark.anyio
async def test_roomy_context_keeps_thinking():
    transport, seen = _server(n_ctx=32768, prompt_tokens=2000)
    await _complete(_adapter(transport), max_tokens=4096)
    assert "chat_template_kwargs" not in seen["body"]
    assert seen["body"]["max_tokens"] == 4096


@pytest.mark.anyio
async def test_prompt_that_fills_the_context_fails_fast_with_advice():
    transport, seen = _server(n_ctx=4096, prompt_tokens=3990)
    with pytest.raises(ModelAdapterError) as info:
        await _complete(_adapter(transport))
    assert "4096-token context" in str(info.value) and "OPREL_CTX_SIZE" in str(info.value)
    assert "body" not in seen  # the model was never asked


@pytest.mark.anyio
async def test_cut_off_while_thinking_is_an_error_not_an_empty_reply():
    reply = {"choices": [{"message": {"content": "", "reasoning_content": "Let me think about the legs..."},
                          "finish_reason": "length"}]}
    transport, _ = _server(n_ctx=8192, prompt_tokens=2000, reply=reply)
    with pytest.raises(ModelAdapterError) as info:
        await _complete(_adapter(transport))
    assert "ran out of room" in str(info.value) and "thinking" in str(info.value)
    assert "8192-token context" in str(info.value)


@pytest.mark.anyio
async def test_tool_call_cut_off_after_the_call_is_still_used():
    reply = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "scene_add_objects", "arguments": "{\"objects\": []}"}},
    ]}, "finish_reason": "length"}]}
    transport, _ = _server(n_ctx=4096, prompt_tokens=2000, reply=reply)
    turn = await _complete(_adapter(transport))
    assert [c.name for c in turn.tool_calls] == ["scene_add_objects"]


@pytest.mark.anyio
async def test_servers_without_props_work_as_before():
    transport, seen = _server(n_ctx=None, prompt_tokens=0, props_status=404)
    await _complete(_adapter(transport), max_tokens=1234)
    assert seen["body"]["max_tokens"] == 1234 and "chat_template_kwargs" not in seen["body"]


@pytest.mark.anyio
async def test_cloud_providers_are_not_budgeted():
    transport, seen = _server(n_ctx=4096, prompt_tokens=3990)
    await _complete(_adapter(transport, local=False), max_tokens=1234)
    assert seen["body"]["max_tokens"] == 1234 and "chat_template_kwargs" not in seen["body"]


@pytest.fixture
def anyio_backend():
    return "asyncio"
