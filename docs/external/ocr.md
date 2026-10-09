# Optical Character Recognition (OCR) Service

The Oprel SDK ships with a dedicated OCR backend powered by quantized Vision-Language Models (VLMs, default `qwen2.5-vl-3b`). It extracts dense text, tabular data, and document coordinates locally with zero Python ML runtime dependencies.

## Architecture

The OCR service (`oprel/server/services/ocr_service.py`) operates through native C++ `llama-server` execution with multimodal vision projection (`--mmproj`).
- **Zero Python ML Runtime**: No `paddlepaddle`, `paddleocr`, or `torch` wheels required.
- **Model Storage**: Uses Oprel's unified GGUF cache (`~/.cache/oprel/models`) with automatic download of both the base model and its `mmproj` vision encoder.
- **Hardware Acceleration**: Automatic GPU acceleration via CUDA, Metal, ROCm, or Vulkan through Oprel's unified hardware telemetry.

## Endpoint: `POST /v1/ocr/extract`

The OCR endpoint is compatible with standard form-data uploads.

**Request:**
- `file`: The image file (multipart/form-data, supported: PNG, JPEG, WEBP, BMP, TIFF).

**Response Schema:**
```json
{
  "id": "ocr_a1b2c3d4e5f6",
  "filename": "invoice.png",
  "img_width": 1024,
  "img_height": 768,
  "results": [
    {
      "text": "Invoice #1042",
      "confidence": 0.985,
      "bbox": [[10, 10], [100, 10], [100, 20], [10, 20]],
      "bbox_norm": {
        "left": 0.009,
        "top": 0.013,
        "width": 0.087,
        "height": 0.013
      }
    }
  ],
  "full_text": "Invoice #1042",
  "word_count": 2
}
```

## Features

1. **Bounding Box Normalization**: The API returns both absolute pixel coordinates (`bbox`) and 0–1 normalized coordinates (`bbox_norm`) making it seamless to render overlay highlights on frontend applications (such as Oprel Studio's React WebUI).
2. **Context-Aware Transcriptions**: High transcription accuracy on handwritten notes, scanned receipts, complex typography, and multilingual documents.
3. **Structured Table & Line Recognition**: Preserves document reading order and table structure without requiring manual line segmentation.
