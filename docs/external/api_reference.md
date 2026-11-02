# REST API & WebSocket Reference

The Oprel daemon server exposes a high-performance HTTP server built on FastAPI. The architecture is designed to natively emulate the OpenAI and Ollama API schemas, allowing drop-in replacement for existing tools.

## Base URLs
- **Ollama Compat:** `http://localhost:11435/api/`
- **OpenAI Compat:** `http://localhost:11435/v1/`
- **Oprel Native:** `http://localhost:11435/`

---

## Ollama Compatibility Layer

The daemon intercepts requests to `/api/*` and translates them to internal runtime calls.

### `POST /api/generate`
- **Payload:** `{"model": "llama3", "prompt": "Hi", "stream": true, "options": {"temperature": 0.7}}`
- **Response:** JSON stream of token objects.

### `POST /api/chat`
- **Payload:** `{"model": "llama3", "messages": [{"role": "user", "content": "Hi"}]}`
- **Response:** JSON stream of message delta objects.

### `GET /api/tags`
- **Response:** List of cached models.

---

## OpenAI Compatibility Layer

The daemon supports standard OpenAI SDKs pointing to `/v1`.

### `POST /v1/chat/completions`
Standard endpoint for LLM generation.
- Supports `tools` / `functions`.
- Supports `response_format: {"type": "json_object"}`.

### `GET /v1/models`
Returns OpenAI-formatted model metadata.

### `POST /v1/images/generations`
Compatible with standard DALL-E SDK clients.
- Intercepts requests and routes them to `stable-diffusion.cpp`.
- Supports `b64_json` or `url` return formats.

---

## Native Oprel Endpoints

### `POST /embedding`
Generates vectors using BERT/Nomic models.
- **Payload:** `{"model": "nomic", "input": ["text 1", "text 2"]}`

### `POST /v1/ocr`
Dedicated endpoint for high-speed Optical Character Recognition.
- Accepts `multipart/form-data` with an image file.
- **Response:** `{"text": "extracted text string", "confidence": 0.98}`

### `POST /shutdown`
Gracefully unloads all models from VRAM and kills the FastAPI server.

### `GET /health`
Returns system status, VRAM usage, and active models.

---

## Authentication (local API token)

The daemon only accepts requests that are either from the local web UI (a loopback `Origin`/same-site browser request) or that carry the per-install token:

```
Authorization: Bearer <token>
```

The token is created on first use and stored in `~/.oprel/api_token` (override with `OPREL_API_TOKEN`). The Python SDK and CLI send it automatically. For OpenAI-compatible clients, pass it as the API key:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key=open(os.path.expanduser("~/.oprel/api_token")).read().strip())
```

Requests with a foreign `Origin`, a non-loopback `Host`, or `Sec-Fetch-Site: cross-site` are rejected with 403. `/health` is open. Set `OPREL_API_AUTH=0` to disable the token check (not recommended); set `OPREL_ALLOWED_HOSTS=name1,name2` if you deliberately serve the daemon under another host name.

## Pre-load fit check and onboarding

- `GET /system/preflight?model_id=...&quantization=Q4_K_M&context_length=4096` returns `verdict` (`gpu`, `hybrid`, `cpu`, `insufficient`), estimated VRAM/RAM, GPU layers and a suggested context. Loading a model that cannot fit in RAM now fails early with a plain-language message.
- `GET /system/onboarding` returns detected hardware, a tier (A/B/C) and one recommended first model.
