# Models Parsing Internals

Oprel must understand exactly what a model is before attempting to load it. The `oprel/models/` package handles binary parsing and heuristics.

## GGUF Parser (`gguf_parser.py`)
GGUF is an extensible binary format that stores both the mathematical weights of the model and its architectural metadata in a structured key-value header.
- **Zero-Copy Reads**: Instead of loading a 10GB file into RAM, the parser seeks directly into the first few kilobytes of the file.
- **Metadata Extraction**: It reads key-value pairs to extract:
  - `general.architecture`: (e.g., `llama`, `qwen2`, `phi3`).
  - `llama.context_length`: The absolute maximum context window the model was trained on.
  - `general.quantization_version`: The compression method used.
- This information allows the Telemetry system to allocate the exact amount of VRAM needed *before* the subprocess even starts.

## Image Model Detector (`image_model_detector.py`)
Stable Diffusion models distributed as GGUFs often lack consistent metadata headers. Oprel uses tensor heuristics to identify them:
- By scanning the tensor shape arrays, it can identify specific bottleneck dimensions that uniquely fingerprint an SD 1.5 model versus an SDXL model versus a Flux.1 architecture. 
- It uses these fingerprints to set default parameters (like scaling down from 30 steps to 4 steps if a turbo architecture is detected).
