"""
Image Generation Example for Oprel SDK.

Demonstrates how to generate an image from a prompt using the image generation API.
"""

import base64
from oprel.client import OprelClient

def main():
    client = OprelClient()
    
    prompt = "A high-tech cyberpunk city at night, neon lights, 4k resolution"
    print(f"Generating image for: '{prompt}'")
    
    # The API is compatible with OpenAI's image generation endpoint
    response = client.generate_image(
        model="sd-turbo",
        prompt=prompt,
        size="512x512",
        response_format="b64_json"
    )
    
    image_data = response['data'][0]['b64_json']
    
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(image_data))
        
    print("Image saved to output.png")

if __name__ == "__main__":
    main()
