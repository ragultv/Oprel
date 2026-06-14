# Oprel CLI Reference

The Oprel SDK provides a comprehensive Command Line Interface (CLI) for managing models, running generation tasks, and controlling the background daemon. 

## Global Options
Before any command, you can specify global flags:
- `--version`: Show the installed Oprel version.
- `--verbose`: Enable debug-level logging. Useful for tracing model loading errors.
- `--quiet`: Suppress all non-essential output.

---

## 1. Execution Commands

These commands are used to interact with models directly from the terminal.

### `oprel chat <model> [options]`
Starts an interactive, multi-turn chat session. The CLI handles prompt templating automatically.
**Arguments:**
- `<model>`: Model ID or alias (e.g., `llama3-8b`).
**Options:**
- `--system "<prompt>"`: Inject a custom system prompt.
- `--max-memory <MB>`: Hard limit on RAM usage.
- `--quantization <level>`: Force a specific quantization (e.g., `Q4_K_M`).
- `--thinking`: Enable reasoning mode for models that support it.
- `--rag`: Enable Retrieval-Augmented Generation using the local knowledge base.
- `--no-server`: Force the model to load directly in the current process rather than the daemon.

### `oprel generate <model> <prompt> [options]`
Single-shot generation.
**Arguments:**
- `<model>`: Model ID or alias.
- `<prompt>`: The text prompt.
**Options:**
- `--max-tokens <int>`: Limit the response length (default: 8192).
- `--temperature <float>`: Adjust creativity (default: 0.7).
- `--stream / --no-stream`: Toggle token streaming.

### `oprel run <model> [prompt] [options]`
A hybrid command. If a prompt is provided, it acts like `generate`. If no prompt is provided, it drops into `chat` mode.

### `oprel gen-image <model> <prompt> [options]`
Generate images using `stable-diffusion.cpp`.
**Arguments:**
- `<model>`: GGUF model path or HF repo ID containing a GGUF image model.
- `<prompt>`: Text description of the image.
**Options:**
- `--negative "<prompt>"`: What to avoid in the image.
- `--width <int>`, `--height <int>`: Dimensions (default 1024x1024).
- `--steps <int>`: Sampling steps (default 28).
- `--output / -o <path>`: Where to save the resulting PNG.

### `oprel vision <model> <prompt> --images <path1> <path2>`
Ask questions about local images using multimodal models.
**Arguments:**
- `<model>`: Vision model alias (e.g., `qwen3-vl-7b`).
- `<prompt>`: Question about the image.
- `--images`: One or more paths to local image files.

### `oprel embed <model> [prompt] [options]`
Generate vector embeddings.
**Options:**
- `--files <path>`: Extract text from files (PDF, DOCX) and embed.
- `--batch <file>`: Read multiple lines from a text file and embed each.
- `--output <path>`: Save JSON embeddings.

---

## 2. Server & UI Commands

### `oprel serve [options]`
Starts the background daemon. The daemon keeps models loaded in memory, allowing instant API responses across multiple client calls.
- `--host <ip>` (default: 127.0.0.1)
- `--port <int>` (default: 11435)

### `oprel start [options]`
Starts the daemon AND launches the React WebUI (Oprel Studio) in your default browser.

### `oprel stop`
Sends a graceful shutdown signal to the daemon. Unloads all models from VRAM and kills orphaned backend processes.

### `oprel models`
Lists all models currently loaded into memory by the server, as well as locally cached models.

---

## 3. Knowledge Base Commands

### `oprel index add <path>`
Add a directory or file to the local vector store.

### `oprel index search <query> [options]`
Search the indexed documents.
- `--top-k <int>`: Number of results to return.

### `oprel index sync`
Force a manual sync of all watched directories.

---

## 4. Model Management Commands

### `oprel pull <model>`
Download a model without loading it into memory.

### `oprel cache`
Manage the `.oprel/cache` directory.
- `oprel cache list`: Show all downloaded models.
- `oprel cache delete <model>`: Delete a specific file.
- `oprel cache clear`: Wipe the entire cache.

### `oprel recommend`
Analyzes your CPU, RAM, and VRAM, and prints a table of recommended models and their optimal quantization levels.
