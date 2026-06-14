# Vision Models

Multimodal vision models allow you to "show" an image to a Large Language Model and ask questions about its contents. Oprel supports vision models packaged as GGUF (e.g., LLaVA, Qwen-VL).

## CLI Usage

You can pass local image paths directly to the CLI.

```bash
oprel vision qwen-vl "Describe the objects in this photo" --images ./photo.jpg
```

**Under the Hood:**
1. The CLI reads the local file.
2. It resizes and compresses the image to prevent overwhelming the context window (handled by `oprel.utils.multimodal.preprocess_image_to_bytes`).
3. It base64-encodes the image and sends it to the generation endpoint alongside the text prompt.

## API Usage

The API accepts base64-encoded images embedded within the prompt array, mimicking the OpenAI vision message schema.

**Endpoint:** `POST /api/chat` (Ollama compatible)

**Payload:**
```json
{
  "model": "llava-v1.6",
  "messages": [
    {
      "role": "user",
      "content": "What is in this image?",
      "images": ["<base64_encoded_string>"]
    }
  ]
}
```

## VRAM Considerations
Vision models require loading an external Vision Encoder (like CLIP) in addition to the base LLM. This significantly increases VRAM requirements. Oprel's `recommender.py` accounts for this projector/encoder overhead when suggesting hardware configurations.
