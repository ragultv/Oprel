# Server Architecture

The Oprel daemon server acts as the central hub for model management, execution, and client communication. It is built using **FastAPI** (`oprel/server/app.py`).

## 1. Routing Layer (`oprel/server/routes/`)
The routing layer is responsible for defining the REST and WebSocket endpoints. It serves as an adapter, translating incoming HTTP requests into internal Python data structures.

- **`openai_compat.py` & `ollama_compat.py`**: These files intercept requests formatted for other popular ecosystems (e.g., `/v1/chat/completions` or `/api/generate`) and coerce them into Oprel's internal `GenerateParams` objects.
- **`generation.py`, `images.py`, `ocr.py`**: Native route handlers that deal specifically with parsing multipart forms, handling SSE streams, and basic request validation.

## 2. Services Layer (`oprel/server/services/`)
The services layer contains the actual business logic. Routes call into services; services rarely call into routes.

- **`generation.py` (Service)**: This is the core orchestrator for LLMs. It handles:
  1. Resolving the model alias to a downloaded file.
  2. Loading the model into VRAM if it isn't already active.
  3. Prepending System Prompts and formatting the chat history using template strings.
  4. Executing Retrieval-Augmented Generation (RAG) lookups if the `--rag` flag is passed.
  5. Yielding async chunks back to the FastAPI router.
- **`model_state.py`**: Maintains a global dictionary (`get_state()`) mapping active model IDs to their loaded `Model` instances. It tracks process PIDs and last-accessed timestamps to enable auto-eviction.

## 3. Database Layer (`oprel/server/db.py`)
Oprel uses a local SQLite database (typically stored in `~/.oprel/oprel.db`) to persist state across daemon restarts.
- **Conversations**: Stores ephemeral chat histories, allowing users to resume chats by referencing a `conversation_id`.
- **Telemetry Logs**: Logs every inference request (tokens per second, latency, model used).
- **Settings**: Stores user preferences and API keys for Cloud Providers.

## 4. Background Daemon
When `oprel serve` is executed, the server detaches into a long-running background process. This means that multi-gigabyte models remain pinned in VRAM between CLI invocations, dropping "Time to First Token" (TTFT) from 5 seconds to 0.1 seconds.
