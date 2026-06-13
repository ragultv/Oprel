"""
Oprel Chat Templates - Python API Examples
===========================================

This file demonstrates how to use Oprel for different types of AI tasks:
1. Text Chat (conversational AI)
2. Vision Chat (ask questions about images) 
3. Image Generation Chat (text-to-image)

Each example shows both simple one-shot usage and multi-turn conversations.
"""

# ============================================================================
# 1. TEXT CHAT - Conversational AI
# ============================================================================

def text_chat_simple():
    """Simple text generation"""
    from oprel import Model
    
    # Load a text model
    model = Model("qwen2.5-3b")
    
    # Generate a response
    response = model.generate("Explain quantum computing in simple terms")
    print(response)


def text_chat_streaming():
    """Streaming text generation (see each token as it's generated)"""
    from oprel import Model
    
    model = Model("llama3-8b")
    
    # Stream the response token by token
    for token in model.generate("Write a short poem about AI", stream=True):
        print(token, end='', flush=True)
    print()


def text_chat_conversation():
    """Multi-turn conversation with memory"""
    from oprel import Model
    
    model = Model("qwen2.5-7b", use_server=True)
    model.load()
    
    # Conversation with memory (server mode)
    conversation_id = "my_chat_session"
    
    # Turn 1
    response1 = model.generate(
        "My name is Alice",
        conversation_id=conversation_id,
        system_prompt="You are a helpful assistant."
    )
    print(f"Assistant: {response1}")
    
    # Turn 2 - model remembers Alice
    response2 = model.generate(
        "What's my name?",
        conversation_id=conversation_id
    )
    print(f"Assistant: {response2}")  # Will respond: "Your name is Alice"
    
    # Reset conversation
    response3 = model.generate(
        "What's my name?",
        conversation_id=conversation_id,
        reset_conversation=True
    )
    print(f"Assistant: {response3}")  # Won't remember Alice


# ============================================================================
# 2. VISION CHAT - Image Understanding
# ============================================================================

def vision_chat_single_image():
    """Ask questions about an image"""
    from oprel import Model
    from oprel.runtime.backends.vision import format_vision_prompt
    
    # Load vision model
    model = Model("qwen3-vl-8b", use_server=False)
    model.load()
    
    # Format vision prompt
    vision_data = format_vision_prompt(
        text_prompt="Describe what you see in this image",
        image_paths=["photo.jpg"],
        model_architecture="qwen-vl"
    )
    
    # Generate description
    response = model.generate(vision_data['prompt'])
    print(response)


def vision_chat_multiple_images():
    """Compare multiple images"""
    from oprel import Model
    from oprel.runtime.backends.vision import format_vision_prompt, get_vision_model_config
    
    model = Model("llava-v1.6-34b", use_server=False)
    model.load()
    
    config = get_vision_model_config("llava-v1.6-34b")
    
    vision_data = format_vision_prompt(
        text_prompt="What are the differences between these two images?",
        image_paths=["before.jpg", "after.jpg"],
        model_architecture=config['architecture']
    )
    
    response = model.generate(vision_data['prompt'], max_tokens=512)
    print(response)


def vision_chat_ocr():
    """Extract text from images (OCR)"""
    from oprel import Model
    from oprel.runtime.backends.vision import format_vision_prompt
    
    model = Model("qwen3-vl-8b", use_server=False)
    model.load()
    
    vision_data = format_vision_prompt(
        text_prompt="Read all the text in this image",
        image_paths=["document.png"],
        model_architecture="qwen-vl"
    )
    
    extracted_text = model.generate(vision_data['prompt'])
    print(extracted_text)


# ============================================================================
# 3. IMAGE GENERATION CHAT - Text-to-Image
# ============================================================================

def image_gen_simple():
    """Generate a single image"""
    from oprel.runtime.backends.comfyui import ComfyUIImageGenerator
    from oprel.runtime.binaries.comfyui_process import ComfyUIBackend
    
    # Start ComfyUI backend
    backend = ComfyUIBackend()
    backend.start()
    
    try:
        # Create generator
        generator = ComfyUIImageGenerator(backend.get_client())
        
        # Generate image
        image_bytes = generator.generate_txt2img(
            prompt="a beautiful sunset over mountains",
            width=512,
            height=512,
            steps=20,
            checkpoint="sd-1.5.safetensors"
        )
        
        # Save image
        with open("sunset.png", "wb") as f:
            f.write(image_bytes)
        
        print("✓ Image saved to sunset.png")
        
    finally:
        backend.stop()


def image_gen_batch():
    """Generate multiple images with different prompts"""
    from oprel.runtime.backends.comfyui import ComfyUIImageGenerator
    from oprel.runtime.binaries.comfyui_process import ComfyUIBackend
    
    backend = ComfyUIBackend()
    backend.start()
    
    try:
        generator = ComfyUIImageGenerator(backend.get_client())
        
        prompts = [
            "a futuristic city",
            "a medieval castle",
            "a tropical beach"
        ]
        
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
            
            image_bytes = generator.generate_txt2img(
                prompt=prompt,
                width=512,
                height=512,
                steps=15,
                checkpoint="sdxl-turbo.safetensors"
            )
            
            with open(f"image_{i+1}.png", "wb") as f:
                f.write(image_bytes)
        
        print("✓ All images generated")
        
    finally:
        backend.stop()


def image_gen_with_negative_prompt():
    """Control what NOT to include in images"""
    from oprel.runtime.backends.comfyui import ComfyUIImageGenerator
    from oprel.runtime.binaries.comfyui_process import ComfyUIBackend
    
    backend = ComfyUIBackend()
    backend.start()
    
    try:
        generator = ComfyUIImageGenerator(backend.get_client())
        
        image_bytes = generator.generate_txt2img(
            prompt="a portrait of a woman, professional photography, high quality",
            negative_prompt="blurry, low quality, distorted, ugly, deformed",
            width=512,
            height=768,  # Portrait aspect ratio
            steps=25,
            cfg_scale=7.5,
            checkpoint="sd-1.5.safetensors"
        )
        
        with open("portrait.png", "wb") as f:
            f.write(image_bytes)
        
        print("✓ Portrait generated")
        
    finally:
        backend.stop()


# ============================================================================
# ADVANCED: Combined Workflows
# ============================================================================

def vision_to_image_workflow():
    """Describe an image, then generate a similar one"""
    from oprel import Model
    from oprel.runtime.backends.vision import format_vision_prompt
    from oprel.runtime.backends.comfyui import ComfyUIImageGenerator
    from oprel.runtime.binaries.comfyui_process import ComfyUIBackend
    
    # Step 1: Analyze source image
    vision_model = Model("qwen3-vl-8b", use_server=False)
    vision_model.load()
    
    vision_data = format_vision_prompt(
        text_prompt="Describe this image in detail, focusing on the style and composition",
        image_paths=["reference.jpg"],
        model_architecture="qwen-vl"
    )
    
    description = vision_model.generate(vision_data['prompt'])
    print(f"Image description: {description}")
    
    # Step 2: Generate similar image
    backend = ComfyUIBackend()
    backend.start()
    
    try:
        generator = ComfyUIImageGenerator(backend.get_client())
        
        image_bytes = generator.generate_txt2img(
            prompt=f"Create an image with this description: {description}",
            width=512,
            height=512,
            steps=20,
            checkpoint="sdxl-turbo.safetensors"
        )
        
        with open("similar_image.png", "wb") as f:
            f.write(image_bytes)
        
        print("✓ Similar image generated")
        
    finally:
        backend.stop()


def chat_with_image_generation():
    """Interactive chat that generates images"""
    from oprel import Model
    from oprel.runtime.backends.comfyui import ComfyUIImageGenerator
    from oprel.runtime.binaries.comfyui_process import ComfyUIBackend
    
    # Setup
    chat_model = Model("qwen2.5-7b", use_server=True)
    chat_model.load()
    
    backend = ComfyUIBackend()
    backend.start()
    generator = ComfyUIImageGenerator(backend.get_client())
    
    try:
        conversation_id = "image_gen_chat"
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Check if user wants to generate an image
            if "generate" in user_input.lower() or "create" in user_input.lower():
                # Ask AI to extract the prompt
                response = chat_model.generate(
                    f"Extract the image generation prompt from: {user_input}. "
                    f"Return ONLY the prompt, nothing else.",
                    conversation_id=conversation_id
                )
                
                print(f"Generating image: {response}")
                
                # Generate the image
                image_bytes = generator.generate_txt2img(
                    prompt=response,
                    width=512,
                    height=512,
                    steps=15,
                    checkpoint="sdxl-turbo.safetensors"
                )
                
                # Save with timestamp
                import time
                filename = f"generated_{int(time.time())}.png"
                with open(filename, "wb") as f:
                    f.write(image_bytes)
                
                print(f"✓ Image saved to {filename}")
            else:
                # Regular chat
                response = chat_model.generate(
                    user_input,
                    conversation_id=conversation_id
                )
                print(f"AI: {response}")
    
    finally:
        backend.stop()


# ============================================================================
# RUN EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("Oprel Chat Templates - Examples\n")
    print("Uncomment the example you want to run:\n")
    
    # TEXT CHAT
    # text_chat_simple()
    # text_chat_streaming()
    # text_chat_conversation()
    
    # VISION CHAT
    # vision_chat_single_image()
    # vision_chat_multiple_images()
    # vision_chat_ocr()
    
    # IMAGE GENERATION
    # image_gen_simple()
    # image_gen_batch()
    # image_gen_with_negative_prompt()
    
    # ADVANCED
    # vision_to_image_workflow()
    # chat_with_image_generation()
    
    print("Edit this file and uncomment an example to run it!")
