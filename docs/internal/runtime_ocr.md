# OCR Runtime

The Optical Character Recognition backend (`oprel/server/services/ocr_service.py`) relies on a quantized Vision-Language Model (VLM, default `qwen2.5-vl-3b` / `unsloth/Qwen2.5-VL-3B-Instruct-GGUF`) running natively via the pre-compiled `llama.cpp` runtime with multimodal vision projection (`--mmproj`).

## Architecture & Design Goals

1. **Zero Python ML Runtime**:
   - No `paddlepaddle`, `paddleocr`, or `torch` dependencies.
   - Eliminates multi-gigabyte wheel installations, compiler toolchain requirements, and subprocess `pip install` scripts.
   - All tensor evaluation runs in native C++ via `llama-server`.

2. **Native GGUF Downloader Integration**:
   - Models are downloaded and verified using the unified `oprel.downloader.hub.download_model` pipeline.
   - Both the primary GGUF model and its accompanying `mmproj` vision projection encoder are automatically cached under `~/.cache/oprel/models`.

3. **Hardware Acceleration**:
   - Hardware detection uses Oprel's unified telemetry (`oprel.telemetry.hardware.detect_gpu`).
   - `llama-server` automatically leverages CUDA, Metal, ROCm, or Vulkan based on hardware availability.

## Processing Pipeline

1. **Ingestion & Preprocessing**:
   - The API receives a multipart file upload (`image/png`, `image/jpeg`, etc.).
   - Pillow (`PILImage`) extracts the original pixel dimensions (`width`, `height`) and encodes the image into base64.
2. **Multimodal Inference**:
   - The request is dispatched to `llama-server`'s `/v1/chat/completions` endpoint with multimodal payload formatting.
   - The VLM is instructed to extract text in reading order and output bounding box coordinates.
3. **Bounding Box Normalization**:
   - The service extracts coordinates (`[ymin, xmin, ymax, xmax]`) from the VLM output.
   - Coordinates are normalized between `0.0` and `1.0` (`bbox_norm: {left, top, width, height}`) alongside pixel-space polygon boundaries (`bbox: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]`).
   - If plain text is returned, vertically sequenced reading bands are calculated as a resilient fallback.
4. **Frontend & DB Persistence**:
   - Results are persisted to SQLite database (`db.add_ocr_job`) and returned in the exact schema expected by the React WebUI (`OcrView.tsx`).
