# Text Generation

Oprel provides a robust text generation engine built on top of `llama.cpp`. It supports standard completions, chat templates, and complex KV-cache management to maintain high speed.

## Under the Hood
When a generation request is dispatched to the Oprel daemon:
1. **Model Resolution**: Oprel resolves aliases (e.g., `llama3-8b`) to local cached files.
2. **Backend Spin-up**: If the model is not loaded, a dedicated `llama.cpp` subprocess is spawned.
3. **Prompt Formatting**: Oprel uses internal chat templates (matching ChatML, Llama-3, etc.) to format multi-turn conversations into the exact string required by the model.
4. **Streaming**: Tokens are streamed back via Server-Sent Events (SSE) for low latency.

## CLI Usage

### Generate (Single-Shot)
```bash
oprel generate "Write a poem about space" --model llama3-8b
```
**Key Flags:**
- `--max-tokens`: Hard limit on the response length.
- `--temperature`: Increase for creativity, decrease for deterministic facts.
- `--no-server`: Force the generation to run in the current terminal process rather than the background daemon (useful for debugging).

### Chat (Interactive)
```bash
oprel chat llama3-8b --system "You are a pirate."
```

## Advanced Features

### Deep Thinking / Reasoning
For models trained with reasoning tokens (like DeepSeek-R1), you can enable thinking mode. This parses the `<think>` blocks out of the output and presents them as separate reasoning steps in the UI.
```bash
oprel chat deepseek-r1-7b --thinking
```

### Context Caching
The daemon persists the KV cache of a conversation between turns. If you ask a follow-up question, the model does not need to re-evaluate the entire prompt history, resulting in instantaneous "Time to First Token" (TTFT).

### Ephemeral vs Persistent Conversations
- **Ephemeral**: Requests without a `conversation_id` are ephemeral. The server remembers the last 40 turns in memory just in case you send another request on the same ID, but it is not saved to the DB.
- **Persistent**: If a `conversation_id` starting with `chat_` is passed, the exact user and assistant messages are logged to the local SQLite database.
