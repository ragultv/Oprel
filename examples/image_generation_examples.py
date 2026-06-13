"""
Image Generation with Oprel SDK - Python Examples
==================================================

This file demonstrates how to generate images programmatically using the Oprel SDK.
"""

from oprel import ImageModel
from pathlib import Path

# Example 1: Basic Image Generation
def basic_generation():
    """Generate a single image with default settings"""
    
    # Initialize the model
    model = ImageModel("sd-1.5")  # or "sdxl-turbo", "flux-1-schnell", etc.
    
    # Generate image
    image_bytes = model.generate(
        prompt="a cute cat sitting on a windowsill, golden hour lighting",
        width=512,
        height=512,
        steps=28
    )
    
    # Save to file
    output_path = Path("generated_cat.png")
    output_path.write_bytes(image_bytes)
    print(f"✓ Saved to {output_path}")


# Example 2: Advanced Generation with Negative Prompts
def advanced_generation():
    """Generate with more control over the output"""
    
    model = ImageModel("sdxl-turbo")
    
    image_bytes = model.generate(
        prompt="professional portrait photo of a software engineer, studio lighting, sharp focus",
        negative_prompt="blurry, low quality, distorted, cartoon, anime",
        width=1024,
        height=1024,
        steps=4,  # SDXL-Turbo works best with 4 steps
        cfg_scale=7.5,  # Guidance scale
    )
    
    Path("portrait.png").write_bytes(image_bytes)
    print("✓ Portrait saved")


# Example 3: Batch Generation
def batch_generation():
    """Generate multiple images with different prompts"""
    
    model = ImageModel("sd-1.5")
    
    prompts = [
        "a serene mountain landscape at sunset",
        "a futuristic city with flying cars",
        "an underwater coral reef with tropical fish",
        "a cozy coffee shop interior, warm lighting"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        image_bytes = model.generate(
            prompt=prompt,
            width=512,
            height=512,
            steps=20
        )
        
        output_path = Path(f"batch_{i+1}.png")
        output_path.write_bytes(image_bytes)
        print(f"  ✓ Saved to {output_path}")


# Example 4: Using Context Manager (Recommended)
def context_manager_example():
    """Use context manager for automatic cleanup"""
    
    with ImageModel("flux-1-schnell") as model:
        image_bytes = model.generate(
            prompt="a magical forest with glowing mushrooms, fantasy art",
            width=1024,
            height=1024,
            steps=4  # FLUX Schnell is optimized for 4 steps
        )
        
        Path("magical_forest.png").write_bytes(image_bytes)
        print("✓ Generated with FLUX")


# Example 5: Different Model Sizes
def model_comparison():
    """Compare different model sizes and speeds"""
    
    prompt = "a red sports car on a mountain road"
    
    models = [
        ("sd-1.5", 512, 512, 20),        # Lightweight: ~4GB, fast
        ("sdxl-turbo", 1024, 1024, 4),   # Fast: ~7GB, 4 steps
        ("sana-2.7b", 1024, 1024, 18),   # Efficient: ~3GB, good quality
        ("pixart-sigma", 1024, 1024, 20), # Balanced: ~2.5GB
    ]
    
    for model_name, width, height, steps in models:
        print(f"\nGenerating with {model_name}...")
        
        try:
            with ImageModel(model_name) as model:
                image_bytes = model.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    steps=steps
                )
                
                output_path = Path(f"comparison_{model_name}.png")
                output_path.write_bytes(image_bytes)
                print(f"  ✓ Saved to {output_path}")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")


# Example 6: Error Handling
def error_handling_example():
    """Proper error handling for image generation"""
    
    try:
        model = ImageModel("sd-1.5")
        
        image_bytes = model.generate(
            prompt="a beautiful sunset over the ocean",
            width=512,
            height=512,
            steps=25
        )
        
        output_path = Path("sunset.png")
        output_path.write_bytes(image_bytes)
        print(f"✓ Success: {output_path}")
        
    except ValueError as e:
        print(f"❌ Invalid parameters: {e}")
    except RuntimeError as e:
        print(f"❌ Generation failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


# Example 7: Custom Output Directory
def organized_output():
    """Organize generated images in folders"""
    
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)
    
    with ImageModel("sdxl-turbo") as model:
        categories = {
            "landscapes": "beautiful mountain landscape with lake",
            "portraits": "professional headshot, studio lighting",
            "abstract": "abstract geometric patterns, vibrant colors",
        }
        
        for category, prompt in categories.items():
            category_dir = output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            image_bytes = model.generate(
                prompt=prompt,
                width=1024,
                height=1024,
                steps=4
            )
            
            output_path = category_dir / "sample.png"
            output_path.write_bytes(image_bytes)
            print(f"✓ Saved {category} to {output_path}")


if __name__ == "__main__":
    print("Oprel Image Generation Examples\n")
    print("Choose an example to run:")
    print("1. Basic Generation")
    print("2. Advanced Generation (negative prompts)")
    print("3. Batch Generation")
    print("4. Context Manager")
    print("5. Model Comparison")
    print("6. Error Handling")
    print("7. Organized Output")
    
    choice = input("\nEnter number (1-7): ").strip()
    
    examples = {
        "1": basic_generation,
        "2": advanced_generation,
        "3": batch_generation,
        "4": context_manager_example,
        "5": model_comparison,
        "6": error_handling_example,
        "7": organized_output,
    }
    
    if choice in examples:
        print(f"\nRunning example {choice}...\n")
        examples[choice]()
    else:
        print("Invalid choice. Running basic example...")
        basic_generation()
