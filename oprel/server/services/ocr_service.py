"""
OCR Service — PaddleOCR backend

Responsibilities:
- Detect whether models are already downloaded to OCR_MODEL_DIR.
- Auto-detect GPU availability (CUDA via paddle, ROCm fallback).
- Lazy-initialize a singleton OCR engine.
- Provide a blocking download path that installs paddleocr + models on first use.
- Run extraction and return clean result dicts.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import io
import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator

from oprel.core.config import Config

logger = logging.getLogger(__name__)

CONFIG = Config()
OCR_MODEL_DIR = Path.home() / ".cache" / "oprel" / "ocr"
OCR_DET_DIR   = OCR_MODEL_DIR / "det"
OCR_REC_DIR   = OCR_MODEL_DIR / "rec"
OCR_CLS_DIR   = OCR_MODEL_DIR / "cls"

# ── Singleton engine ───────────────────────────────────────────────────────────
_ocr_engine: Any = None
_engine_lock = threading.Lock()

# ── Download state (for SSE progress) ─────────────────────────────────────────
_download_events: list[dict] = []
_download_done = threading.Event()
_download_error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

_gpu_available: bool | None = None

def detect_gpu() -> bool:
    """Return True if paddle is installed and CUDA is available (cached)."""
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available
    try:
        importlib.invalidate_caches()
        if importlib.util.find_spec("paddle") is None:
            _gpu_available = False
            return False
        import paddle  # type: ignore
        _gpu_available = bool(paddle.is_compiled_with_cuda())
        return _gpu_available
    except Exception:
        _gpu_available = False
        return False


# Warm up GPU/installation cache in a background thread on startup to prevent blocking first UI load
def _warmup_gpu_cache():
    try:
        detect_gpu()
    except Exception:
        pass

threading.Thread(target=_warmup_gpu_cache, daemon=True).start()


def is_paddleocr_installed() -> bool:
    """Check whether paddleocr is importable without loading the package."""
    try:
        importlib.invalidate_caches()
        return importlib.util.find_spec("paddleocr") is not None
    except Exception:
        return False


def models_exist() -> bool:
    """Return True when all three model subdirs exist and are non-empty."""
    for d in (OCR_DET_DIR, OCR_REC_DIR, OCR_CLS_DIR):
        if not d.exists() or not any(d.iterdir()):
            return False
    return True


def is_ocr_ready() -> bool:
    return is_paddleocr_installed() and models_exist()


def get_ocr_status() -> dict:
    size_mb = 0
    if OCR_MODEL_DIR.exists():
        size_mb = sum(f.stat().st_size for f in OCR_MODEL_DIR.rglob("*") if f.is_file()) // (1024 * 1024)
    return {
        "ready": is_ocr_ready(),
        "model_dir": str(OCR_MODEL_DIR),
        "size_mb": size_mb,
        "gpu": detect_gpu(),
        "installed": is_paddleocr_installed(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Download / setup
# ─────────────────────────────────────────────────────────────────────────────

def _emit(step: str, message: str, done: bool = False, error: str | None = None) -> None:
    """Append a progress event visible to the SSE endpoint."""
    _download_events.append({
        "step": step,
        "message": message,
        "done": done,
        "error": error,
        "ts": time.time(),
    })


def _run_download() -> None:
    """Blocking download: install packages then trigger model download."""
    global _download_error
    try:
        # ── Step 1: install paddle ────────────────────────────────────────────
        _emit("install", "Installing paddlepaddle (CPU/GPU)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "paddlepaddle", "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"paddlepaddle install failed: {result.stderr[:400]}")

        # ── Step 2: install paddleocr ─────────────────────────────────────────
        _emit("install", "Installing paddleocr...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "paddleocr", "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"paddleocr install failed: {result.stderr[:400]}")

        # ── Step 3: download models by running a dummy inference ───────────────
        _emit("models", "Downloading detection model (~7 MB)...")
        OCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        from paddleocr import PaddleOCR  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image as PILImage  # type: ignore

        use_gpu = detect_gpu()
        _emit("models", f"Initializing OCR engine (GPU={use_gpu})...")

        # Creating the engine triggers model downloads to the specified dirs
        engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=use_gpu,
            det_model_dir=str(OCR_DET_DIR),
            rec_model_dir=str(OCR_REC_DIR),
            cls_model_dir=str(OCR_CLS_DIR),
            show_log=False,
        )

        # Run on a tiny white image to ensure all models fully load
        _emit("models", "Verifying models with test inference...")
        dummy = np.ones((32, 128, 3), dtype=np.uint8) * 255
        engine.ocr(dummy, cls=True)

        _emit("done", "OCR setup complete ✓", done=True)

    except Exception as exc:
        _download_error = str(exc)
        _emit("error", str(exc), done=True, error=str(exc))
        logger.error(f"OCR download failed: {exc}")
    finally:
        _download_done.set()


def start_download_async() -> None:
    """Start the download in a background thread. Idempotent."""
    global _download_events, _download_error
    _download_events = []
    _download_error = None
    _download_done.clear()
    t = threading.Thread(target=_run_download, daemon=True)
    t.start()


def iter_download_progress() -> Generator[str, None, None]:
    """Yield SSE-formatted strings from the download event queue."""
    sent = 0
    while True:
        while sent < len(_download_events):
            evt = _download_events[sent]
            yield f"data: {json.dumps(evt)}\n\n"
            sent += 1
            if evt.get("done"):
                return
        if _download_done.is_set():
            # Drain any remaining events
            while sent < len(_download_events):
                evt = _download_events[sent]
                yield f"data: {json.dumps(evt)}\n\n"
                sent += 1
            return
        time.sleep(0.25)


# ─────────────────────────────────────────────────────────────────────────────
# OCR Engine — lazy singleton
# ─────────────────────────────────────────────────────────────────────────────

def _get_engine():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    with _engine_lock:
        if _ocr_engine is not None:
            return _ocr_engine
        if not is_ocr_ready():
            raise RuntimeError("OCR models not ready. Run setup first.")
        from paddleocr import PaddleOCR  # type: ignore
        use_gpu = detect_gpu()
        logger.info(f"Initializing PaddleOCR engine (gpu={use_gpu})")
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=use_gpu,
            det_model_dir=str(OCR_DET_DIR),
            rec_model_dir=str(OCR_REC_DIR),
            cls_model_dir=str(OCR_CLS_DIR),
            show_log=False,
        )
        return _ocr_engine


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

def run_ocr(image_bytes: bytes) -> dict:
    """
    Run OCR on raw image bytes.

    Returns:
      {
        "width": int,          # image pixel width
        "height": int,         # image pixel height
        "results": [
          {
            "text": str,
            "confidence": float,
            "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],   # absolute pixels
            "bbox_norm": { "left": f, "top": f, "width": f, "height": f }  # 0-1 normalized
          }
        ]
      }
    """
    import numpy as np  # type: ignore
    from PIL import Image as PILImage  # type: ignore

    engine = _get_engine()

    # Decode bytes → numpy array
    pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = pil_img.size
    img_array = np.array(pil_img)

    raw = engine.ocr(img_array, cls=True)

    results: list[dict] = []
    # PaddleOCR returns list[list] — outer list is "pages" (always 1 here)
    page = raw[0] if raw else []

    for line in (page or []):
        if not line:
            continue
        bbox_points, (text, conf) = line
        # bbox_points is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] as floats
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        bbox = [[int(p[0]), int(p[1])] for p in bbox_points]
        bbox_norm = {
            "left":   round(x_min / img_w, 6),
            "top":    round(y_min / img_h, 6),
            "width":  round((x_max - x_min) / img_w, 6),
            "height": round((y_max - y_min) / img_h, 6),
        }
        results.append({
            "text": str(text),
            "confidence": round(float(conf), 4),
            "bbox": bbox,
            "bbox_norm": bbox_norm,
        })

    return {"width": img_w, "height": img_h, "results": results}


def image_to_data_url(image_bytes: bytes, content_type: str = "image/png") -> str:
    """Convert raw bytes to a base64 data URL for storage in the DB."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{b64}"
