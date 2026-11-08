"""
Model adapters for tool calling.

Every backend Oprel talks to exposes an OpenAI-compatible /chat/completions that
accepts `tools`:
- local llama.cpp (llama-server with Jinja templates, on by default since b4000+;
  Oprel ships b7822) parses each model's native tool-call format for us
- OpenAI, Groq, OpenRouter, NVIDIA NIM
- Gemini via its OpenAI-compatibility endpoint

So one adapter covers all of them; only base URL / auth differ.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from oprel.server import db
from oprel.tools.base import ToolCall
from oprel.utils.redaction import redact

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_TOOL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

PROVIDER_DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


class ModelAdapterError(RuntimeError):
    pass


class ContextFullError(ModelAdapterError):
    """The model's context had no room left to answer (or it got cut off before answering)."""


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""

    def as_message(self) -> dict[str, Any]:
        """The assistant message to append to history (OpenAI shape)."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)},
                }
                for c in self.tool_calls
            ]
        return msg


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str, str | None]:
    if isinstance(raw, dict):
        return raw, json.dumps(raw), None
    text = raw if isinstance(raw, str) else ""
    if not text.strip():
        return {}, "{}", None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, text, f"{exc.msg} at position {exc.pos}"
    if not isinstance(value, dict):
        return {}, text, "arguments must be a JSON object"
    return value, text, None


def parse_tool_calls(message: dict[str, Any], tool_names: set[str]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") or {}
        name = fn.get("name") or ""
        args, raw_args, err = _parse_arguments(fn.get("arguments"))
        calls.append(ToolCall(id=raw.get("id") or _new_call_id(), name=name, arguments=args,
                              raw_arguments=raw_args, parse_error=err))
    if calls:
        return calls
    return _fallback_parse(message.get("content") or "", tool_names)


def _fallback_parse(content: str, tool_names: set[str]) -> list[ToolCall]:
    """
    Some GGUF templates lack tool support, so llama.cpp returns the call as text.
    Accept the two unambiguous shapes: Hermes/Qwen `<tool_call>{...}</tool_call>`, or a
    reply that is *only* a JSON object naming a known tool.
    """
    text = _THINK_RE.sub("", content).strip()
    candidates = _TOOL_TAG_RE.findall(text)
    if not candidates:
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates = [stripped]
    calls = []
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or (obj.get("function") or {}).get("name")
        if name not in tool_names:
            continue
        args, raw_args, err = _parse_arguments(obj.get("arguments", obj.get("parameters", {})))
        calls.append(ToolCall(id=_new_call_id(), name=name, arguments=args, raw_arguments=raw_args, parse_error=err))
    return calls


@dataclass
class SamplingParams:
    max_tokens: int | None = 2048
    temperature: float | None = 0.7
    top_p: float | None = None
    thinking: bool = True  # False: local thinking models skip reasoning (see Integration.allow_thinking)


# Local llama.cpp models often run with a small context (4k on 4 GB GPUs). Tool definitions
# take a big share of it, and "thinking" models (Qwen3, DeepSeek-R1 distills) can spend the
# rest reasoning and get cut off before they call a tool or answer.
MIN_ROOM_TO_ANSWER = 256      # fewer tokens left than this: fail fast with a clear message
# Fewer than this: ask the chat template to skip thinking. Qwen3 often thinks 1.5-3k tokens
# about tool definitions, and a multi-object tool call needs several hundred more.
MIN_ROOM_TO_THINK = 4096
_CONTEXT_HELP = (
    "Use a cloud model, or give local models more context: set the environment variable "
    "OPREL_CTX_SIZE (e.g. 8192) and restart Oprel."
)


class OpenAICompatibleAdapter:
    def __init__(
        self,
        *,
        label: str,
        base_url: str,
        model: str,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 300.0,
        llama_server_root: str | None = None,
    ):
        self.label = label
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._headers = headers or {}
        self._transport = transport
        self._timeout = timeout
        # Set for local llama.cpp backends: enables context budgeting via /props and /tokenize.
        self._llama_root = llama_server_root.rstrip("/") if llama_server_root else None

    async def _context_window(self, client: httpx.AsyncClient) -> int | None:
        # Not cached: a reloaded backend can reuse the port with a different --ctx-size.
        try:
            props = (await client.get(f"{self._llama_root}/props")).json()
            n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
            return int(n_ctx) if n_ctx else None
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return None

    async def _prompt_tokens(self, client: httpx.AsyncClient, messages: list[dict[str, Any]],
                             tools: list[dict[str, Any]]) -> int | None:
        """Close upper bound: the JSON text tokenizes to slightly more than the rendered template."""
        try:
            resp = await client.post(f"{self._llama_root}/tokenize",
                                     json={"content": json.dumps(messages) + json.dumps(tools)})
            return len(resp.json()["tokens"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return None

    async def _fit_to_context(self, client: httpx.AsyncClient, body: dict[str, Any],
                              messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int | None:
        """Cap max_tokens to the room left and skip thinking when it can't fit. Returns n_ctx."""
        n_ctx = await self._context_window(client)
        prompt = await self._prompt_tokens(client, messages, tools) if n_ctx else None
        if not n_ctx or prompt is None:
            return n_ctx
        room = n_ctx - prompt
        if room < MIN_ROOM_TO_ANSWER:
            raise ContextFullError(
                f"{self.label} has a {n_ctx}-token context and this conversation plus the tool definitions "
                f"already take about {prompt} tokens, so there is no room to answer. "
                f"Start a new chat or turn off tools you don't need. {_CONTEXT_HELP}"
            )
        if room < MIN_ROOM_TO_THINK:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body["max_tokens"] = min(body.get("max_tokens") or room, room)
        return n_ctx

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], params: SamplingParams
    ) -> AssistantTurn:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if params.max_tokens:
            body["max_tokens"] = params.max_tokens
        if params.temperature is not None:
            body["temperature"] = params.temperature
        if params.top_p is not None:
            body["top_p"] = params.top_p

        if self._llama_root and not params.thinking:
            # Read by thinking templates (Qwen3, ...); ignored by every other template. Only sent to
            # local llama.cpp: cloud APIs may reject unknown fields. Seen at 8k context on a 4 GB GPU:
            # Qwen3-8B thought for 1,700+ tokens (~5 minutes) about a 3-object drawing and timed out.
            body["chat_template_kwargs"] = {"enable_thinking": False}

        n_ctx = None
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
                if self._llama_root:
                    n_ctx = await self._fit_to_context(client, body, messages, tools)
                resp = await client.post(f"{self.base_url}/chat/completions", json=body, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise ModelAdapterError(
                f"{self.label} did not answer within {self._timeout / 60:.0f} minutes. On a slow GPU a thinking model "
                f"can spend that long reasoning; try a smaller context (OPREL_CTX_SIZE), a smaller model, or a cloud model."
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelAdapterError(f"{self.label}: could not reach model ({type(exc).__name__})") from exc

        if resp.status_code != 200:
            detail = resp.text[:500]
            if tools and resp.status_code in (400, 422, 500) and "tool" in detail.lower():
                raise ModelAdapterError(
                    f"{self.label} rejected the tool definitions. This model may not support tool calling. "
                    f"Details: {redact(detail)}"
                )
            raise ModelAdapterError(f"{self.label} error (HTTP {resp.status_code}): {redact(detail)}")

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_names = {t["function"]["name"] for t in tools}
        calls = parse_tool_calls(message, tool_names)
        content = message.get("content") or ""
        if calls and not message.get("tool_calls"):
            content = ""  # the text *was* the tool call (fallback parse)
        content = _THINK_RE.sub("", content).strip()
        reasoning = message.get("reasoning_content") or ""
        if not calls and not content and choice.get("finish_reason") == "length":
            # Cut off before producing anything usable: say so instead of returning an empty reply.
            spent = "thinking" if reasoning or "<think>" in (message.get("content") or "") else "its reply"
            window = f"its {n_ctx}-token context" if n_ctx else "its output limit"
            raise ContextFullError(
                f"{self.label} ran out of room before answering: it used up {window} on {spent}. {_CONTEXT_HELP}"
            )
        return AssistantTurn(content=content, tool_calls=calls, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Resolution: request.model -> adapter
# ---------------------------------------------------------------------------

def _split_provider_model(model: str) -> tuple[str, str | None]:
    if "::" in model:
        p, m = model.split("::", 1)
        return p, m
    if ":" in model:
        p, m = model.split(":", 1)
        return p, m
    return model, None


def _provider_adapter(provider: dict[str, Any], model_name: str | None) -> OpenAICompatibleAdapter:
    p_type = provider.get("type", "openai")
    base_url = (provider.get("base_url") or "").rstrip("/") or PROVIDER_DEFAULT_URLS.get(p_type, "")
    if p_type == "groq" and not base_url.endswith("/openai/v1"):
        base_url = f"{base_url}/openai/v1"
    if p_type == "gemini" and "generativelanguage.googleapis.com" in base_url and not base_url.endswith("/openai"):
        base_url = PROVIDER_DEFAULT_URLS["gemini"]
    if not base_url:
        raise ModelAdapterError(f"Provider '{provider.get('name')}' has no base URL configured")

    if not model_name:
        enabled = provider.get("enabled_model_ids") or []
        model_name = enabled[0] if enabled else None
    if not model_name:
        raise ModelAdapterError(f"Choose a model for provider '{provider.get('name')}'")

    headers = {"Authorization": f"Bearer {provider.get('api_key', '')}"}
    if p_type == "openrouter":
        headers.update({"HTTP-Referer": "https://oprel.ai", "X-Title": "OPREL"})
    return OpenAICompatibleAdapter(label=provider.get("name") or p_type, base_url=base_url, model=model_name, headers=headers)


def _local_adapter(model_id: str) -> OpenAICompatibleAdapter:
    from oprel.server.domain.state import get_state
    from oprel.server.services.generation import _resolve_model_id
    from oprel.server.services.model_state import mark_model_used
    from oprel.server.services.models import load_model

    state = get_state()
    resolved = _resolve_model_id(model_id)
    model = state.models.get(resolved)
    process = getattr(model, "_process", None) if model else None
    if model is None or (process is not None and not process.is_running()):
        state.models.pop(resolved, None)
        state.model_configs.pop(resolved, None)
        load_model(resolved)
        model = state.models[resolved]
    mark_model_used(resolved)

    client = getattr(model, "_client", None)
    base = getattr(client, "base_url", None)
    transport = None
    if not base and getattr(client, "socket_path", None):
        transport = httpx.AsyncHTTPTransport(uds=str(client.socket_path))
        base = "http://localhost"
    if not base:
        raise ModelAdapterError("Local model backend is not reachable")
    return OpenAICompatibleAdapter(label=f"Local model {model_id}", base_url=f"{base}/v1", model=resolved,
                                   transport=transport, llama_server_root=base)


def resolve_adapter(model: str) -> OpenAICompatibleAdapter:
    provider_id, model_name = _split_provider_model(model)
    provider = db.get_provider(provider_id)
    if provider:
        return _provider_adapter(provider, model_name)
    # Local ids from the web UI look like "Qwen/Qwen3-8B-GGUF::Q5_K_M". The suffix is the
    # quantization, which the model registry already tracks, so load by repo id alone
    # (the plain chat path does the same).
    return _local_adapter(model.split("::", 1)[0])
