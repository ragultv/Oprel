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
    from oprel import generate_image
    import base64
    
    print("Generating image...")
    # Generate image using stable-diffusion.cpp backend
    response = generate_image(
        model="sd-1.5",
        prompt="a beautiful sunset over mountains",
        size="512x512",
        steps=20
    )
    
    # Extract the base64 part and save
    img_data = response.data[0]["url"].split(",", 1)[1]
    image_bytes = base64.b64decode(img_data)
    
    with open("sunset.png", "wb") as f:
        f.write(image_bytes)
    
    print("✓ Image saved to sunset.png")


def image_gen_batch():
    """Generate multiple images with different prompts"""
    from oprel import generate_image
    import base64
    
    prompts = [
        "a futuristic city",
        "a medieval castle",
        "a tropical beach"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
        
        response = generate_image(
            model="sd-1.5",  # Using a standard lightweight GGUF model
            prompt=prompt,
            size="512x512",
            steps=15
        )
        
        img_data = response.data[0]["url"].split(",", 1)[1]
        image_bytes = base64.b64decode(img_data)
        
        with open(f"image_{i+1}.png", "wb") as f:
            f.write(image_bytes)
    
    print("✓ All images generated")


def image_gen_with_negative_prompt():
    """Control what NOT to include in images"""
    from oprel import generate_image
    import base64
    
    print("Generating image with negative prompt...")
    response = generate_image(
        model="sd-1.5",
        prompt="a portrait of a woman, professional photography, high quality",
        negative_prompt="blurry, low quality, distorted, ugly, deformed",
        size="512x512",
        steps=25,
        cfg_scale=7.5
    )
    
    img_data = response.data[0]["url"].split(",", 1)[1]
    image_bytes = base64.b64decode(img_data)
    
    with open("portrait.png", "wb") as f:
        f.write(image_bytes)
    
    print("✓ Portrait generated")


# ============================================================================
# ADVANCED: Combined Workflows
# ============================================================================

def vision_to_image_workflow():
    """Describe an image, then generate a similar one"""
    from oprel import Model, generate_image
    from oprel.runtime.backends.vision import format_vision_prompt
    import base64
    
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
    print("Generating similar image...")
    response = generate_image(
        model="sd-1.5",
        prompt=f"Create an image with this description: {description}",
        size="512x512",
        steps=20
    )
    
    img_data = response.data[0]["url"].split(",", 1)[1]
    image_bytes = base64.b64decode(img_data)
    
    with open("similar_image.png", "wb") as f:
        f.write(image_bytes)
    
    print("✓ Similar image generated")


def chat_with_image_generation():
    """Interactive chat that generates images"""
    from oprel import Model, generate_image
    import base64
    import time
    
    # Setup
    chat_model = Model("qwen2.5-7b", use_server=True)
    chat_model.load()
    
    try:
        conversation_id = "image_gen_chat"
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Check if user wants to generate an image
            if "generate" in user_input.lower() or "create" in user_input.lower():
                # Ask AI to extract the prompt
                response_text = chat_model.generate(
                    f"Extract the image generation prompt from: {user_input}. "
                    f"Return ONLY the prompt, nothing else.",
                    conversation_id=conversation_id
                )
                
                print(f"Generating image: {response_text}")
                
                # Generate the image
                response = generate_image(
                    model="sd-1.5",
                    prompt=response_text,
                    size="512x512",
                    steps=15
                )
                
                img_data = response.data[0]["url"].split(",", 1)[1]
                image_bytes = base64.b64decode(img_data)
                
                # Save with timestamp
                filename = f"generated_{int(time.time())}.png"
                with open(filename, "wb") as f:
                    f.write(image_bytes)
                
                print(f"✓ Image saved to {filename}")
            else:
                # Regular chat
                response_text = chat_model.generate(
                    user_input,
                    conversation_id=conversation_id
                )
                print(f"AI: {response_text}")
    
    finally:
        pass


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
