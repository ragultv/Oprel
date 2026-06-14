"""
OCR Service Example for Oprel SDK.

Extracts text from an image using the local OCR backend.
"""

from oprel.client import OprelClient

def main():
    client = OprelClient()
    
    # Assuming 'document.png' exists in the current directory
    # You would pass the base64 encoded image or file path depending on the client method
    try:
        response = client.ocr(image_path="document.png")
        print("Extracted Text:\n", response['text'])
    except Exception as e:
        print("Error during OCR:", e)

if __name__ == "__main__":
    main()
