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
