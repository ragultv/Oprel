# Image Generation

Oprel supports local, offline image generation using the `stable-diffusion.cpp` backend. It handles GGUF versions of Stable Diffusion 1.5, SDXL, and Flux.

## Model Compatibility

Image models must be stored in the GGUF format. When downloading an image model, Oprel's `image_model_detector.py` uses heuristics on the tensor shapes inside the GGUF header to automatically determine the architecture.

## CLI Usage

Generate an image directly from the terminal:
```bash
oprel gen-image "A vast cyberpunk city, neon lights, highly detailed, 8k resolution" \
      --model sd-turbo \
      --width 1024 \
      --height 1024 \
      --steps 20 \
      --output ./my_city.png
```

### Parameters
- `--negative`: A string specifying what to explicitly avoid in the generation (e.g., "blurry, low quality, distorted anatomy").
- `--guidance`: The Classifier-Free Guidance (CFG) scale. Higher values force the model to stick closer to the prompt at the cost of creativity. The daemon auto-detects the optimal default based on the model architecture.
- `--steps`: The number of diffusion steps. Standard models require 20-30 steps. Turbo models (SD-Turbo, SDXL-Turbo) only require 4 steps.

## API Usage

Oprel exposes an OpenAI-compatible endpoint for image generation, making it a drop-in replacement for any app expecting DALL-E.

**Endpoint:** `POST /v1/images/generations`

**Payload:**
```json
{
  "model": "sd-turbo",
  "prompt": "A cute cat playing with yarn",
  "size": "512x512",
  "response_format": "b64_json"
}
```

**Response:**
Returns an array containing the `b64_json` string, which you can decode directly into a PNG image byte array.

## Memory Management
Image generation is highly VRAM intensive. If Oprel detects high memory pressure, it will dynamically offload layers to system RAM or unload inactive LLMs to make space for the diffusion process.
