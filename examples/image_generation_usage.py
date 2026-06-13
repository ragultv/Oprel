"""
Image Generation Usage Examples

This demonstrates how to use image generation models with Oprel SDK
through both Python API and programmatic code.
"""

# ============================================================================
# Method 1: Using Python API (Recommended)
# ============================================================================

from oprel.runtime.backends.comfyui import ComfyUIClient, ComfyUIImageGenerator
from pathlib import Path

# Initialize client
client = ComfyUIClient(base_url="http://127.0.0.1:8188")

# Check if ComfyUI is available
if not client.is_available():
    print("❌ ComfyUI not running!")
    print("Please start ComfyUI or run: oprel setup image")
    exit(1)

# Initialize generator
generator = ComfyUIImageGenerator(client)

# Generate image
print("Generating image...")
image_bytes = generator.generate_txt2img(
    prompt="a beautiful sunset over mountains, highly detailed",
    negative_prompt="blurry, low quality, distorted",
    width=1024,
    height=1024,
    steps=20,
    cfg_scale=7.0,
    sampler="euler",
    seed=-1  # -1 for random
)

# Save image
output_path = Path("generated_sunset.png")
output_path.write_bytes(image_bytes)
print(f"✓ Saved to {output_path}")


# ============================================================================
# Method 2: Using Specific Model (Checkpoint)
# ============================================================================

# List available models
models = client.get_models("checkpoints")
print(f"Available models: {models}")

# Generate with specific checkpoint
if "FLUX.1-schnell-fp8.safetensors" in models:
    image_bytes = generator.generate_txt2img(
        prompt="a futuristic city at night",
        checkpoint="FLUX.1-schnell-fp8.safetensors",
        width=1024,
        height=1024,
        steps=4,  # FLUX Schnell is fast - only needs 4 steps
        cfg_scale=1.0  # Turbo models need cfg=1.0
    )
    Path("flux_city.png").write_bytes(image_bytes)


# ============================================================================
# Method 3: Advanced Parameters (SDXL)
# ============================================================================

image_bytes = generator.generate_txt2img(
    prompt="epic fantasy landscape with dragons flying, ultra detailed, 8k",
    negative_prompt="cartoon, anime, low quality, watermark",
    width=1024,
    height=1024,
    steps=30,
    cfg_scale=7.5,
    sampler="dpmpp_2m",
    scheduler="karras",
    seed=42,  # Fixed seed for reproducibility
    checkpoint="sd_xl_turbo_1.0_fp16.safetensors",
    timeout=300  # Max 5 minutes
)

Path("fantasy_landscape.png").write_bytes(image_bytes)
print("✓ Generated fantasy landscape")


# ============================================================================
# Method 4: Batch Generation
# ============================================================================

prompts = [
    "a cute cat sleeping",
    "a majestic lion roaring",
    "a playful dog running"
]

for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
    
    image_bytes = generator.generate_txt2img(
        prompt=prompt,
        negative_prompt="blurry, distorted",
        width=512,
        height=512,
        steps=15,
        cfg_scale=7.0
    )
    
    output_path = Path(f"animal_{i+1}.png")
    output_path.write_bytes(image_bytes)
    print(f"  ✓ Saved to {output_path}")


# ============================================================================
# Method 5: Error Handling
# ============================================================================

def generate_image_safe(prompt, output_path):
    """Generate image with proper error handling"""
    try:
        client = ComfyUIClient()
        
        if not client.is_available():
            print("Error: ComfyUI not available")
            print("Start ComfyUI or run: oprel setup image")
            return False
        
        generator = ComfyUIImageGenerator(client)
        
        image_bytes = generator.generate_txt2img(
            prompt=prompt,
            width=512,
            height=512,
            steps=20
        )
        
        Path(output_path).write_bytes(image_bytes)
        print(f"✓ Generated: {output_path}")
        return True
        
    except TimeoutError as e:
        print(f"Error: Generation timed out - {e}")
        return False
    except RuntimeError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# Use it
generate_image_safe("a serene beach at sunset", "beach.png")


# ============================================================================
# Method 6: Model-Specific Optimizations
# ============================================================================

def generate_with_auto_params(prompt, model_name=None):
    """Auto-detect model and use optimal parameters"""
    client = ComfyUIClient()
    generator = ComfyUIImageGenerator(client)
    
    # Get available models
    models = client.get_models("checkpoints")
    
    # Select model
    if model_name and model_name in models:
        checkpoint = model_name
    else:
        checkpoint = models[0] if models else None
    
    if not checkpoint:
        raise RuntimeError("No models available")
    
    # Auto-configure based on model type
    is_turbo = "turbo" in checkpoint.lower()
    is_sdxl = "xl" in checkpoint.lower() or "sdxl" in checkpoint.lower()
    is_flux = "flux" in checkpoint.lower()
    
    if is_turbo or is_flux:
        # Turbo models: fast, low steps, cfg=1.0
        steps = 4
        cfg = 1.0
        sampler = "euler"
    elif is_sdxl:
        # SDXL: higher quality, more steps
        steps = 30
        cfg = 7.5
        sampler = "dpmpp_2m"
    else:
        # SD 1.5: balanced
        steps = 20
        cfg = 7.0
        sampler = "euler"
    
    print(f"Using: {checkpoint}")
    print(f"Params: steps={steps}, cfg={cfg}, sampler={sampler}")
    
    return generator.generate_txt2img(
        prompt=prompt,
        checkpoint=checkpoint,
        width=1024,
        height=1024,
        steps=steps,
        cfg_scale=cfg,
        sampler=sampler
    )

# Use auto-detection
image_bytes = generate_with_auto_params("a magical forest with glowing mushrooms")
Path("auto_generated.png").write_bytes(image_bytes)


# ============================================================================
# Method 7: Integration with Text Generation (Image Description)
# ============================================================================

from oprel import generate

# Step 1: Generate description using LLM
description_prompt = """Generate a detailed image description for:
Topic: fantasy castle
Include: architecture details, lighting, mood, style"""

description = generate("qwen3-1.6b", description_prompt, max_tokens=200)
print(f"Generated description: {description}")

# Step 2: Use description to generate image
client = ComfyUIClient()
generator = ComfyUIImageGenerator(client)

image_bytes = generator.generate_txt2img(
    prompt=description,
    negative_prompt="blurry, low quality",
    width=1024,
    height=1024,
    steps=25,
    cfg_scale=7.5
)

Path("llm_to_image.png").write_bytes(image_bytes)
print("✓ Generated image from LLM description")


# ============================================================================
# Method 8: OpenAI API Compatible (Future)
# ============================================================================

"""
# Coming soon - OpenAI Images API compatibility
from oprel import Client

client = Client()

# OpenAI-style API
response = client.images.generate(
    model="flux-1-schnell",
    prompt="a white siamese cat",
    n=1,
    size="1024x1024"
)

image_url = response.data[0].url
# or
image_bytes = response.data[0].b64_json
"""


# ============================================================================
# CLI Usage Examples (for reference)
# ============================================================================

"""
# Basic usage
oprel gen-image flux-1-schnell "a beautiful landscape"

# With parameters
oprel gen-image sdxl-turbo "a cute cat" \\
    --width 1024 \\
    --height 1024 \\
    --steps 4 \\
    --output cat.png

# With negative prompt
oprel gen-image sd-1.5 "portrait of a person" \\
    --negative "blurry, distorted, low quality" \\
    --steps 20 \\
    --guidance 7.5

# Multiple parameters
oprel gen-image flux-1-dev "futuristic city" \\
    --width 1920 \\
    --height 1080 \\
    --steps 10 \\
    --guidance 7.0 \\
    --output cyberpunk_city.png
"""


# ============================================================================
# Best Practices
# ============================================================================

"""
1. **Model Selection**:
   - FLUX Schnell: Fast (4 steps), good quality
   - SDXL Turbo: Fast (4 steps), decent quality
   - SDXL: Best quality (20-30 steps), slower
   - SD 1.5: Balanced (15-20 steps)

2. **Parameters**:
   - Turbo models: steps=4, cfg=1.0
   - Standard models: steps=20-30, cfg=7.0-8.0
   - Resolution: multiples of 8 (512, 768, 1024, etc.)

3. **Prompts**:
   - Be specific and detailed
   - Include style keywords: "photorealistic", "artistic", "detailed"
   - Use negative prompts to avoid unwanted elements

4. **Performance**:
   - Lower resolution = faster (512x512 vs 1024x1024)
   - Fewer steps = faster (but lower quality)
   - Use turbo models for quick iterations

5. **Error Handling**:
   - Always check if ComfyUI is available
   - Handle timeouts for large images
   - Validate model names before use
"""
