# Image Generation Quick Reference

## Setup (First Time Only)

```bash
# Install ComfyUI + CUDA dependencies (~2GB)
oprel setup image

# This will:
# - Download ComfyUI
# - Install PyTorch with CUDA
# - Download FLUX Schnell model (~17GB)
```

---

## Python API Usage

### Basic Example

```python
from oprel.runtime.backends.comfyui import ComfyUIClient, ComfyUIImageGenerator
from pathlib import Path

# Initialize
client = ComfyUIClient()
generator = ComfyUIImageGenerator(client)

# Generate
image_bytes = generator.generate_txt2img(
    prompt="a beautiful sunset over mountains",
    width=1024,
    height=1024,
    steps=20
)

# Save
Path("output.png").write_bytes(image_bytes)
```

### All Parameters

```python
image_bytes = generator.generate_txt2img(
    prompt="your text prompt here",
    negative_prompt="what to avoid (optional)",
    width=1024,              # Image width (divisible by 8)
    height=1024,             # Image height (divisible by 8)
    steps=20,                # Sampling steps (4-50)
    cfg_scale=7.0,           # Guidance scale (1.0-15.0)
    sampler="euler",         # Sampling method
    scheduler="normal",       # Scheduler type
    seed=-1,                 # Random seed (-1 for random)
    checkpoint="model.safetensors",  # Model to use
    timeout=300              # Max wait time (seconds)
)
```

### Available Samplers

```python
# Fast samplers
"euler"              # Best for turbo models
"euler_ancestral"    # More variation

# High quality samplers
"dpmpp_2m"          # Recommended for SDXL
"dpmpp_2m_sde"      # Better quality, slower
"dpmpp_sde"         # Creative variations

# Legacy samplers
"ddim"
"heun"
"uni_pc"
```

---

## CLI Usage

### Basic

```bash
# Simple generation
oprel gen-image flux-1-schnell "a cute cat"

# With output file
oprel gen-image sdxl-turbo "landscape" --output sunset.png
```

### Advanced

```bash
# Full parameters
oprel gen-image flux-1-dev "epic fantasy castle" \
    --width 1024 \
    --height 1024 \
    --steps 20 \
    --guidance 7.5 \
    --negative "blurry, low quality" \
    --output fantasy.png
```

### Common Use Cases

```bash
# Fast generation (FLUX Schnell)
oprel gen-image flux-1-schnell "prompt" --steps 4

# High quality (SDXL)
oprel gen-image sdxl "prompt" --steps 30 --guidance 7.5

# Specific resolution
oprel gen-image sd-1.5 "prompt" --width 512 --height 768

# Landscape format
oprel gen-image flux "prompt" --width 1920 --height 1080
```

---

## Model-Specific Settings

### FLUX Schnell (Fast)
```python
steps=4
cfg_scale=1.0
sampler="euler"
width=1024
height=1024
# ~10 seconds per image
```

### FLUX Dev (Quality)
```python
steps=20
cfg_scale=3.5
sampler="euler"
width=1024
height=1024
# ~30 seconds per image
```

### SDXL Turbo (Fast)
```python
steps=4
cfg_scale=1.0
sampler="euler"
width=1024
height=1024
# ~8 seconds per image
```

### SDXL (Best Quality)
```python
steps=30
cfg_scale=7.5
sampler="dpmpp_2m"
width=1024
height=1024
# ~60 seconds per image
```

### SD 1.5 (Balanced)
```python
steps=20
cfg_scale=7.0
sampler="euler"
width=512
height=512
# ~15 seconds per image
```

---

## Common Patterns

### 1. List Available Models

```python
from oprel.runtime.backends.comfyui import ComfyUIClient

client = ComfyUIClient()
models = client.get_models("checkpoints")
print("Available models:", models)
```

### 2. Generate Multiple Images

```python
prompts = ["a cat", "a dog", "a bird"]

for i, prompt in enumerate(prompts):
    image_bytes = generator.generate_txt2img(
        prompt=prompt,
        steps=15
    )
    Path(f"image_{i}.png").write_bytes(image_bytes)
```

### 3. Auto-Detect Model Type

```python
checkpoint = "FLUX.1-schnell-fp8.safetensors"

# Auto-configure
is_turbo = "turbo" in checkpoint.lower()
steps = 4 if is_turbo else 20
cfg = 1.0 if is_turbo else 7.0

image_bytes = generator.generate_txt2img(
    prompt="test",
    checkpoint=checkpoint,
    steps=steps,
    cfg_scale=cfg
)
```

### 4. Error Handling

```python
try:
    image_bytes = generator.generate_txt2img(
        prompt="test prompt",
        timeout=180
    )
    Path("output.png").write_bytes(image_bytes)
except TimeoutError:
    print("Generation took too long!")
except RuntimeError as e:
    print(f"Generation failed: {e}")
```

---

## Resolution Guidelines

### Common Resolutions (all divisible by 8)

```python
# Square
512x512   # SD 1.5 default (fast)
768x768   # SD 1.5 high quality
1024x1024 # SDXL/FLUX default

# Landscape (16:9)
1024x576  # HD landscape
1920x1080 # Full HD landscape

# Portrait (9:16)
576x1024  # HD portrait
1080x1920 # Full HD portrait

# Widescreen (21:9)
1344x576  # Ultrawide

# Social Media
1080x1080 # Instagram square
1080x1350 # Instagram portrait
```

---

## Prompt Engineering Tips

### Good Prompts

```python
# Be specific
"a realistic photograph of a golden retriever puppy playing in a meadow, 
 soft sunlight, professional photography, bokeh background"

# Include style
"oil painting of a castle on a hill, impressionist style, 
 warm colors, detailed brushstrokes"

# Add quality tags
"epic fantasy landscape, highly detailed, 8k, professional, 
 dramatic lighting, cinematic composition"
```

### Negative Prompts

```python
# Common negatives
"blurry, low quality, distorted, ugly, bad anatomy, watermark, 
 text, signature, amateur"

# For realistic photos
"cartoon, anime, illustration, painting, drawing, 
 unrealistic, fake"

# For art
"photograph, realistic, photo-realistic, low resolution"
```

---

## Troubleshooting

### ComfyUI Not Available

```bash
# Check status
oprel gen-image flux "test" 
# If error: "ComfyUI not running"

# Solution: Run setup again
oprel setup image
```

### Generation Timeout

```python
# Increase timeout for large images
image_bytes = generator.generate_txt2img(
    prompt="complex scene",
    width=1920,
    height=1080,
    steps=30,
    timeout=600  # 10 minutes
)
```

### Out of Memory

```python
# Reduce resolution
width=512   # Instead of 1024
height=512

# Or reduce steps
steps=15    # Instead of 30

# Or use turbo model
checkpoint="sdxl-turbo.safetensors"
steps=4
cfg_scale=1.0
```

---

## Performance Comparison

| Model | Resolution | Steps | GPU Time | Quality |
|-------|-----------|-------|----------|---------|
| FLUX Schnell | 1024x1024 | 4 | ~10s | Good |
| FLUX Dev | 1024x1024 | 20 | ~30s | Excellent |
| SDXL Turbo | 1024x1024 | 4 | ~8s | Good |
| SDXL | 1024x1024 | 30 | ~60s | Excellent |
| SD 1.5 | 512x512 | 20 | ~15s | Good |

*Times are approximate for RTX 3060 (12GB VRAM)*

---

## Advanced: Custom Workflows

For advanced users who want to customize ComfyUI workflows:

```python
# Create custom workflow
workflow = {
    "1": {
        "inputs": {"ckpt_name": "model.safetensors"},
        "class_type": "CheckpointLoaderSimple"
    },
    # ... more nodes
}

# Queue it
prompt_id = client.queue_prompt(workflow)
history = client.wait_for_completion(prompt_id)

# Extract image
for node_output in history["outputs"].values():
    if "images" in node_output:
        image_info = node_output["images"][0]
        image_bytes = client.get_image(
            filename=image_info["filename"]
        )
```

---

## Next Steps

- See `examples/image_generation_usage.py` for complete code examples
- Check ComfyUI documentation for advanced workflows
- Experiment with different models and parameters
- Join Oprel community for tips and tricks
