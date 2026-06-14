# OCR Runtime

The Optical Character Recognition backend (`oprel/server/services/ocr_service.py`) relies on PaddleOCR, functioning very differently from the LLM runtime.

## Lazy Singleton Initialization
Because loading PaddleOCR requires initializing Python tensor arrays (unlike the subprocess model of `llama.cpp`), it is treated as an in-process Lazy Singleton.
- `paddlepaddle` and `paddleocr` are dynamically imported only when an OCR request hits the server.
- The engine (`_ocr_engine`) is instantiated once and kept in memory for the lifetime of the daemon.

## Model Downloader
If the tiny detection/recognition/classification models are missing, the OCR service blocks the first request and runs a synchronous download routine directly from PaddleOCR's upstream servers into `~/.cache/oprel/ocr`.

## Processing Pipeline
1. **Ingestion**: The API receives a multipart file upload. The bytes are read directly into memory.
2. **Tensor Conversion**: The raw bytes are decoded via Pillow (`PILImage`) and converted into a `numpy` array.
3. **Inference**: The `engine.ocr()` method is called. It executes:
   - *Detection*: Locates text bounding boxes.
   - *Classification*: Rotates upside-down or sideways text.
   - *Recognition*: Transcribes the text within the boxes.
4. **Normalization**: The backend calculates the relative width/height of the bounding boxes (`0.0` to `1.0`) against the original image dimensions. This math guarantees that frontends can accurately overlay highlights regardless of how the image is scaled on the user's screen.
