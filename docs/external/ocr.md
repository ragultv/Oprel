# Optical Character Recognition (OCR) Service

The Oprel SDK ships with a highly optimized, dedicated OCR backend powered by PaddleOCR. It is designed to extract dense text from images rapidly, without the heavy overhead of loading a multi-billion parameter multimodal LLM.

## Architecture

The OCR service (`oprel/server/services/ocr_service.py`) operates as a lazy singleton within the daemon.
- **Auto-Installation**: If PaddleOCR is not installed, the daemon triggers a background download of the `paddlepaddle` and `paddleocr` pip packages.
- **Model Downloads**: It fetches three tiny models (~7MB total) for Detection, Recognition, and Classification.
- **GPU Acceleration**: The backend automatically detects CUDA environments and routes tensor operations to the GPU. If no GPU is available, it silently falls back to CPU.

## Endpoint: `POST /v1/ocr`

The OCR endpoint is compatible with standard form-data uploads.

**Request:**
- `file`: The image file (multipart/form-data).

**Response Schema:**
```json
{
  "width": 1024,
  "height": 768,
  "results": [
    {
      "text": "Extracted line of text",
      "confidence": 0.985,
      "bbox": [[10, 10], [100, 10], [100, 20], [10, 20]],
      "bbox_norm": {
        "left": 0.009,
        "top": 0.013,
        "width": 0.087,
        "height": 0.013
      }
    }
  ]
}
```

## Features

1. **Bounding Box Normalization**: The API returns both absolute pixel coordinates (`bbox`) and 0-1 normalized coordinates (`bbox_norm`) making it incredibly easy to render overlay highlights on frontend applications (like the React WebUI's OcrView).
2. **Angle Classification**: Automatically corrects images that are upside down or rotated 90 degrees before extraction.
3. **High Density**: Unlike vision LLMs which often hallucinate or skip lines in dense documents, the PaddleOCR engine performs strict localized extraction, making it perfect for invoices, receipts, and dense PDFs.
