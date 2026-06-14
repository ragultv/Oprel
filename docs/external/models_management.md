# Models Management

Oprel acts as a centralized model hub for your local machine. It handles downloading, verifying, caching, and serving models efficiently.

## The Cache Directory

All downloaded models are stored in a centralized cache directory:
- **Windows**: `C:\Users\<User>\.cache\oprel\models`
- **macOS/Linux**: `~/.cache/oprel/models`

## Pulling Models

You can download models directly from the Hugging Face Hub using aliases or full repository IDs.
```bash
oprel pull llama3-8b
oprel pull TheBloke/Mistral-7B-Instruct-v0.2-GGUF
```

### Auto-Quantization Selection
When you pull an alias (like `llama3-8b`), Oprel invokes the telemetry system (`recommender.py`). It profiles your system's total RAM and VRAM, and intelligently selects the best GGUF quantization level (e.g., `Q4_K_M` for 8GB systems, `Q8_0` for 32GB systems) to maximize quality without causing out-of-memory errors.

## Model Validation

When a model is downloaded (`oprel/downloader/hub.py`), Oprel calculates its SHA-256 hash to ensure the integrity of the multi-gigabyte blob. If a download is interrupted, the hub client automatically resumes from the last downloaded byte range using HTTP `Range` headers.

## CLI Cache Commands

List all models taking up space on your hard drive:
```bash
oprel cache list
```

Remove a specific model to free up space:
```bash
oprel cache delete llama3-8b
```

Wipe all models:
```bash
oprel cache clear
```

## Inspecting Models (`gguf_parser.py`)

Oprel features a built-in GGUF metadata parser. When a model is scanned, it parses the binary file headers (without loading the massive tensor data into memory) to extract the architecture type, maximum context length, and parameter count. This powers the hardware validation checks before generation begins.
