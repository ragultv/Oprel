# Python Client API Reference

The `oprel.Client` provides a programmatic interface to the Oprel daemon. It is designed to be fully compatible with the Ollama Python SDK syntax.

## Initialization

```python
from oprel import Client

# Connects to default daemon at http://localhost:11435
client = Client()

# Or specify a custom host
client = Client(host="http://192.168.1.100:11435")
```

---

## Core Methods

### `client.chat(model, messages, stream=False, options=None)`
Conduct a multi-turn conversation. The daemon handles prompt templating automatically.

**Arguments:**
- `model` (str): Model name or alias.
- `messages` (list): List of dicts `{"role": "user|assistant|system", "content": "..."}`.
- `stream` (bool): If True, returns a generator yielding chunks.
- `options` (dict): Dictionary of parameters (e.g., `{"temperature": 0.8, "num_predict": 1024}`).

**Returns:** 
A `ChatResponse` object (or iterator).

```python
response = client.chat(
    model='llama3-8b',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response.message.content)
```

### `client.generate(model, prompt, system=None, stream=False)`
Single-shot generation.

**Arguments:**
- `model` (str): Model name.
- `prompt` (str): Input text.
- `system` (str): Optional system prompt overriding the default.

### `client.generate_image(model, prompt, size="1024x1024", ...)`
Generate images via the API.

**Arguments:**
- `model` (str): Image model alias.
- `prompt` (str): Description.
- `size` (str): Dimensions format `WxH`.
- `negative_prompt` (str): What to avoid.
- `steps` (int): Sampling steps.

**Returns:**
An `ImageResponse` containing the base64 encoded image or URL.

### `client.embed(texts, model="nomic-embed-text", normalize=True)`
Generate vector embeddings.

**Arguments:**
- `texts` (str | list): A single string or a list of strings.
- `model` (str): Embedding model to use.
- `normalize` (bool): Whether to L2-normalize the output vectors.

**Returns:**
A list of floats (if single text) or a list of lists of floats.

### `client.pull(model)`
Download a model to the daemon's cache.

```python
client.pull("qwen2.5-coder")
```

### `client.list()`
Retrieve a list of available models (both loaded in memory and cached on disk).

```python
models = client.list()
for model in models.models:
    print(f"Name: {model.name}, Loaded: {model.details.get('loaded', False)}")
```

---

## Module-level Convenience Methods

You can bypass instantiating the `Client` class and use module-level functions directly:

```python
import oprel

# Automatically creates a singleton client
response = oprel.chat(model="llama3", messages=[...])
vector = oprel.embed("Test")
```
