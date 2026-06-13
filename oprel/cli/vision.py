"""
Vision commands for Oprel CLI.
"""
import argparse
from pathlib import Path
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def cmd_vision(args: argparse.Namespace) -> int:
    """
    Vision command: Ask questions about images using VLM models.
    Supports: qwen-vl, llava, minicpm-v, moondream, etc.
    """
    from oprel.core.model import Model
    from oprel.runtime.backends.vision import format_vision_prompt, get_vision_model_config
    
    try:
        # Validate images exist
        for img_path in args.images:
            if not Path(img_path).exists():
                print(f"Error: Image not found: {img_path}")
                return 1
        
        # Get vision model config
        config = get_vision_model_config(args.model)
        logger.info(f"Vision model architecture: {config['architecture']}")
        
        # Check image count
        if len(args.images) > config['max_images']:
            print(f"Warning: {args.model} supports max {config['max_images']} images, using first {config['max_images']}")
            args.images = args.images[:config['max_images']]
        
        # Load vision model (use direct mode to avoid daemon complexity)
        print(f"Loading vision model: {args.model}")
        model = Model(args.model, use_server=False)  # Direct mode for vision
        model.load()
        
        # Format prompt with images
        vision_data = format_vision_prompt(
            text_prompt=args.prompt,
            image_paths=args.images,
            model_architecture=config['architecture']
        )
        
        print(f"\nAnalyzing {vision_data['num_images']} image(s)...\n")
        
        # Generate response with image data
        response = model.generate(
            vision_data['prompt'],
            max_tokens=args.max_tokens or 8192,
            temperature=args.temperature or 0.7,
            stream=not args.no_stream,
            images=vision_data['images'],  # Pass base64-encoded images to backend
        )
        
        if args.no_stream:
            print(response)
        else:
            for chunk in response:
                print(chunk, end='', flush=True)
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Vision command failed: {e}", exc_info=True)
        return 1
