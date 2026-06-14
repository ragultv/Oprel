# Image Generation Runtime

Oprel integrates `stable-diffusion.cpp` (`oprel/runtime/image_generation.py`) to execute diffusion models locally without requiring massive Python dependencies like PyTorch.

## Execution Flow

1. **Validation**: The user requests an image generation via the `/v1/images/generations` endpoint.
2. **Memory Clearance**: Image generation requires significant VRAM (often 4-8GB depending on resolution and architecture). Before spawning the SD subprocess, `memory_pressure.py` is invoked. If an LLM is currently occupying VRAM and there isn't enough space, Oprel will gracefully pause or unload the LLM.
3. **Subprocess Execution**: `stable-diffusion.cpp` is invoked directly as a binary. Unlike the LLM runtime which spins up a persistent HTTP server, the SD runtime is generally invoked in a "single-shot" mode, writing the output directly to a temporary file.
4. **Encoding**: The Python daemon reads the temporary bitmap/PNG file, base64 encodes it, and returns it over the API, immediately cleaning up the temporary artifacts.

## Prompt Formatting
The runtime automatically injects optimal default negative prompts if the user omits them, ensuring baseline structural quality for generic generations.

## Turbo Models
Oprel detects if the model name contains `turbo`. If so, it overrides the default sampling steps (usually 20-30) down to 4 steps, ensuring 1-2 second generation times on consumer hardware.
