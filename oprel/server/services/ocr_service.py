"""
OCR Service — GGUF Vision-Language Model (VLM) backend

Responsibilities:
- Detect whether the vision model (default: Qwen2.5-VL-3B) and mmproj files are cached.
- Auto-detect GPU acceleration through hardware telemetry.
- Provide async model download path using native GGUF downloader (no Python ML runtime).
- Run vision-based text extraction and format bounding boxes and lines identically
  for WebUI compatibility.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Generator

from PIL import Image as PILImage

from oprel.core.config import Config
from oprel.downloader.hub import download_model
from oprel.server.services.context import CONFIG, logger

# ── Download state (for SSE progress) ─────────────────────────────────────────
_download_events: list[dict] = []
_download_done = threading.Event()
_download_error: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers & cache detection
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# RapidOCR (Production DBNet ONNX) Singleton & Helpers
# ─────────────────────────────────────────────────────────────────────────────

_rapidocr_engine = None
_rapidocr_lock = threading.Lock()
_rapidocr_available_cache: Optional[bool] = None
_gpu_detected_cache: Optional[bool] = None
_ocr_models_status_cache: Optional[list[dict[str, Any]]] = None
_ocr_models_status_time: float = 0


def invalidate_ocr_cache():
    global _ocr_models_status_cache, _ocr_models_status_time
    _ocr_models_status_cache = None
    _ocr_models_status_time = 0


def is_rapidocr_available() -> bool:
    """Return True if rapidocr_onnxruntime is installed and available (cached)."""
    global _rapidocr_available_cache
    if _rapidocr_available_cache is not None:
        return _rapidocr_available_cache
    try:
        import rapidocr_onnxruntime  # type: ignore # noqa: F401
        _rapidocr_available_cache = True
    except Exception:
        _rapidocr_available_cache = False
    return _rapidocr_available_cache


def get_rapidocr_engine():
    """Thread-safe lazy initialization for RapidOCR engine."""
    global _rapidocr_engine
    if _rapidocr_engine is not None:
        return _rapidocr_engine
    with _rapidocr_lock:
        if _rapidocr_engine is not None:
            return _rapidocr_engine
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        _rapidocr_engine = RapidOCR()
        return _rapidocr_engine


# ─────────────────────────────────────────────────────────────────────────────
# OCR Family Registry & Public helpers
# ─────────────────────────────────────────────────────────────────────────────

OCR_MODELS_INFO: dict[str, dict[str, Any]] = {
    "rapidocr": {
        "id": "rapidocr",
        "name": "RapidOCR (Production DBNet)",
        "repo_id": "rapidocr-onnxruntime",
        "architecture": "onnx",
        "params": "15M",
        "default_quant": "INT8/FP32",
        "description": "Production DBNet + SVTR pixel-grounded text detection & recognition (Default)",
    },
    "dots-ocr": {
        "id": "dots-ocr",
        "name": "dots.ocr 1.7B",
        "repo_id": "ggml-org/dots.ocr-GGUF",
        "architecture": "llama.cpp",
        "params": "1.7B",
        "default_quant": "Q8_0",
        "description": "dots multilingual document layout & visual reasoning model",
    },
    "deepseek-ocr-3b": {
        "id": "deepseek-ocr-3b",
        "name": "DeepSeek-OCR 3B",
        "repo_id": "ggml-org/DeepSeek-OCR-GGUF",
        "architecture": "llama.cpp",
        "params": "3B",
        "default_quant": "Q8_0",
        "description": "DeepSeek grounding & multilingual OCR model",
    },
    "glm-ocr": {
        "id": "glm-ocr",
        "name": "GLM-OCR",
        "repo_id": "ggml-org/GLM-OCR-GGUF",
        "architecture": "llama.cpp",
        "params": "0.9B",
        "default_quant": "Q8_0",
        "description": "Zhipu AI lightweight layout and text OCR model",
    },
    "auto": {
        "id": "auto",
        "name": "Auto (Hybrid DBNet + VLM)",
        "repo_id": "auto-hybrid",
        "architecture": "hybrid",
        "params": "Hybrid",
        "default_quant": "Optimal",
        "description": "Pixel-grounded RapidOCR detection with optional VLM layout structuring",
    },
}

_active_ocr_model_id: str | None = None


def detect_gpu() -> bool:
    """Return True if GPU hardware is detected via telemetry (cached)."""
    global _gpu_detected_cache
    if _gpu_detected_cache is not None:
        return _gpu_detected_cache
    try:
        from oprel.telemetry.hardware import detect_gpu as _detect_gpu
        gpu_info = _detect_gpu()
        _gpu_detected_cache = bool(gpu_info)
    except Exception:
        _gpu_detected_cache = False
    return _gpu_detected_cache


def get_active_ocr_model_id() -> str:
    """Return the currently active OCR model ID, defaulting to rapidocr if available."""
    global _active_ocr_model_id
    if _active_ocr_model_id and _active_ocr_model_id in OCR_MODELS_INFO:
        return _active_ocr_model_id

    cfg_id = getattr(CONFIG, "ocr_model_id", "rapidocr")
    models_status = get_ocr_models_status()

    # 1. Check if configured model is ready
    for m in models_status:
        if m["id"] == cfg_id and m["ready"]:
            _active_ocr_model_id = cfg_id
            return cfg_id

    # 2. Prefer rapidocr if available
    for m in models_status:
        if m["id"] == "rapidocr" and m["ready"]:
            _active_ocr_model_id = "rapidocr"
            return "rapidocr"

    # 3. Check first ready model
    for m in models_status:
        if m["ready"]:
            _active_ocr_model_id = m["id"]
            return m["id"]

    # 4. Check first installed model
    for m in models_status:
        if m["installed"]:
            _active_ocr_model_id = m["id"]
            return m["id"]

    _active_ocr_model_id = cfg_id
    return cfg_id


def set_active_ocr_model(model_id: str) -> dict:
    """Set the active OCR model ID and return updated status."""
    global _active_ocr_model_id
    if model_id not in OCR_MODELS_INFO:
        found = False
        for k in OCR_MODELS_INFO:
            if model_id.lower() in k.lower() or k.lower() in model_id.lower():
                model_id = k
                found = True
                break
        if not found:
            raise ValueError(f"Unknown OCR model ID '{model_id}'. Available: {list(OCR_MODELS_INFO.keys())}")

    _active_ocr_model_id = model_id
    setattr(CONFIG, "ocr_model_id", model_id)
    invalidate_ocr_cache()
    return get_ocr_status()


def get_cached_vlm_files(target_model_id: str | None = None) -> tuple[Path | None, Path | None]:
    """
    Find the cached VLM GGUF file and its corresponding mmproj file for an OCR family model.
    Only models belonging to the OCR family (glm-ocr, deepseek-ocr-3b, dots-ocr) are accepted.
    """
    try:
        from oprel.downloader.aliases import OFFICIAL_REPOS, resolve_model_id
        from oprel.downloader.hub import _find_cached_model_for_repo
        from oprel.runtime.backends.vision import detect_mmproj_file

        ocr_repos: dict[str, str] = OFFICIAL_REPOS.get("OCR") or OFFICIAL_REPOS.get("Ocr", {})

        model_id = target_model_id or get_active_ocr_model_id()
        quant = getattr(CONFIG, "ocr_model_quant", "Q8_0")
        repo_id = ocr_repos.get(model_id) or resolve_model_id(model_id)

        # 1. Search via hub cache lookup for configured model
        cached_model = _find_cached_model_for_repo(repo_id, CONFIG.cache_dir, quant)
        if not cached_model:
            cached_model = _find_cached_model_for_repo(repo_id, CONFIG.cache_dir, None)

        if cached_model:
            mmproj = detect_mmproj_file(cached_model.parent)
            if mmproj:
                return cached_model, mmproj

        # 2. Dynamic scan across ALL models in the OCR family if target not specified
        if not target_model_id and CONFIG.cache_dir.exists():
            for ocr_alias, ocr_repo in ocr_repos.items():
                m = _find_cached_model_for_repo(ocr_repo, CONFIG.cache_dir, None)
                if m:
                    mp = detect_mmproj_file(m.parent)
                    if mp:
                        return m, mp

        # 3. Filesystem search fallback strictly for target model or OCR family
        if CONFIG.cache_dir.exists():
            search_repos = {model_id: repo_id} if target_model_id else ocr_repos
            for ocr_alias, ocr_repo in search_repos.items():
                repo_clean = ocr_repo.replace("/", "--").lower()
                for p in CONFIG.cache_dir.rglob("*.gguf"):
                    fname_lower = p.name.lower()
                    if "mmproj" in fname_lower or fname_lower.startswith("vision-") or fname_lower.startswith("clip-"):
                        continue
                    if repo_clean in str(p).lower() or ocr_alias.lower() in fname_lower:
                        mp = detect_mmproj_file(p.parent)
                        if mp:
                            return p, mp

        return None, None
    except Exception as exc:
        logger.debug(f"Error checking cached OCR VLM files: {exc}")
        return None, None


def get_ocr_models_status() -> list[dict[str, Any]]:
    """Scan and return status for each model in the OCR family."""
    global _ocr_models_status_cache, _ocr_models_status_time
    import time as _time
    now = _time.time()
    if _ocr_models_status_cache is not None and (now - _ocr_models_status_time < 30.0):
        return _ocr_models_status_cache

    from oprel.downloader.aliases import OFFICIAL_REPOS, resolve_model_id
    from oprel.downloader.hub import _find_cached_model_for_repo
    from oprel.runtime.backends.vision import detect_mmproj_file

    ocr_repos: dict[str, str] = OFFICIAL_REPOS.get("OCR") or OFFICIAL_REPOS.get("Ocr", {})
    results: list[dict[str, Any]] = []

    for model_id, info in OCR_MODELS_INFO.items():
        if model_id in ("rapidocr", "auto"):
            avail = is_rapidocr_available()
            results.append({
                "id": model_id,
                "name": info["name"],
                "repo_id": info["repo_id"],
                "architecture": info["architecture"],
                "params": info["params"],
                "quant": info["default_quant"],
                "description": info["description"],
                "installed": avail,
                "has_mmproj": True,
                "ready": avail,
                "size_mb": 15 if avail else 0,
                "file_path": "python:rapidocr_onnxruntime" if avail else None,
            })
            continue

        repo_id = ocr_repos.get(model_id, info["repo_id"])
        model_path = _find_cached_model_for_repo(repo_id, CONFIG.cache_dir, None)
        mmproj_path = None

        if model_path:
            mmproj_path = detect_mmproj_file(model_path.parent)
        elif CONFIG.cache_dir.exists():
            repo_clean = repo_id.replace("/", "--").lower()
            for p in CONFIG.cache_dir.rglob("*.gguf"):
                fname_lower = p.name.lower()
                if "mmproj" in fname_lower or fname_lower.startswith("vision-") or fname_lower.startswith("clip-"):
                    continue
                if repo_clean in str(p).lower() or model_id.lower() in fname_lower:
                    model_path = p
                    mmproj_path = detect_mmproj_file(p.parent)
                    break

        installed = bool(model_path and model_path.exists())
        has_mmproj = bool(mmproj_path and mmproj_path.exists())
        ready = installed and has_mmproj

        size_mb = 0
        if model_path and model_path.exists():
            size_mb += model_path.stat().st_size // (1024 * 1024)
        if mmproj_path and mmproj_path.exists():
            size_mb += mmproj_path.stat().st_size // (1024 * 1024)

        # Detect quantization from filename
        quant = info["default_quant"]
        if model_path:
            for q in ["Q8_0", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "f16", "BF16"]:
                if q.lower() in model_path.name.lower():
                    quant = q.upper()
                    break

        results.append({
            "id": model_id,
            "name": info["name"],
            "repo_id": repo_id,
            "architecture": info["architecture"],
            "params": info["params"],
            "quant": quant,
            "description": info["description"],
            "installed": installed,
            "has_mmproj": has_mmproj,
            "ready": ready,
            "size_mb": size_mb,
            "file_path": str(model_path) if model_path else None,
        })

    _ocr_models_status_cache = results
    _ocr_models_status_time = now
    return results


def is_ocr_ready() -> bool:
    """Return True when RapidOCR or any OCR family model is ready in cache."""
    if is_rapidocr_available():
        return True
    model_path, mmproj_path = get_cached_vlm_files()
    if model_path and mmproj_path and model_path.exists() and mmproj_path.exists():
        return True
    models_status = get_ocr_models_status()
    return any(m["ready"] for m in models_status)


def get_ocr_status() -> dict:
    """Return whether OCR is ready and current cache stats for OCR family."""
    models_status = get_ocr_models_status()
    active_id = get_active_ocr_model_id()
    active_info = OCR_MODELS_INFO.get(active_id, OCR_MODELS_INFO["rapidocr"])

    if active_id in ("rapidocr", "auto"):
        avail = is_rapidocr_available()
        return {
            "ready": avail,
            "installed": avail,
            "active_model_id": active_id,
            "active_model_name": active_info["name"],
            "model_id": active_id,
            "model_name": active_info["name"],
            "model_dir": str(CONFIG.cache_dir),
            "size_mb": 15 if avail else 0,
            "gpu": detect_gpu(),
            "models": models_status,
        }

    model_path, mmproj_path = get_cached_vlm_files(active_id)
    active_ready = bool(model_path and mmproj_path and model_path.exists() and mmproj_path.exists())
    active_installed = bool(model_path and model_path.exists())
    any_ready = any(m["ready"] for m in models_status)
    any_installed = any(m["installed"] for m in models_status)

    size_mb = 0
    if model_path and model_path.exists():
        size_mb += model_path.stat().st_size // (1024 * 1024)
    if mmproj_path and mmproj_path.exists():
        size_mb += mmproj_path.stat().st_size // (1024 * 1024)

    return {
        "ready": active_ready or any_ready,
        "installed": active_installed or any_installed,
        "active_model_id": active_id,
        "active_model_name": active_info["name"],
        "model_id": active_id,
        "model_name": active_info["name"],
        "model_dir": str(CONFIG.cache_dir),
        "size_mb": size_mb,
        "gpu": detect_gpu(),
        "models": models_status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Download / setup (Native GGUF + mmproj download)
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
    """Background task: download GGUF VLM and mmproj projection via hub downloader."""
    global _download_error
    try:
        from oprel.downloader.aliases import resolve_model_id

        model_id = getattr(CONFIG, "ocr_model_id", "glm-ocr")
        quant = getattr(CONFIG, "ocr_model_quant", "Q8_0")
        repo_id = resolve_model_id(model_id)

        _emit("prep", f"Preparing VLM download for {model_id} ({quant})...")
        _emit("models", f"Downloading {repo_id} GGUF and vision projector...")

        # download_model automatically ensures the mmproj projection file is downloaded
        download_model(
            model_id=repo_id,
            quantization=quant,
            cache_dir=CONFIG.cache_dir,
            force_download=False,
        )

        _emit("verify", "Verifying GGUF model and vision projection files...")
        model_path, mmproj_path = get_cached_vlm_files()
        if not model_path or not mmproj_path:
            raise RuntimeError(
                f"VLM download completed but model or mmproj could not be verified in {CONFIG.cache_dir}"
            )

        invalidate_ocr_cache()
        _emit("done", "OCR VLM setup complete ✓", done=True)
    except Exception as exc:
        _download_error = str(exc)
        _emit("error", str(exc), done=True, error=str(exc))
        logger.error(f"OCR VLM download failed: {exc}", exc_info=True)
    finally:
        _download_done.set()


def start_download_async() -> None:
    """Start the VLM download in a background thread. Idempotent."""
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
            while sent < len(_download_events):
                evt = _download_events[sent]
                yield f"data: {json.dumps(evt)}\n\n"
                sent += 1
            return
        time.sleep(0.25)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing & Bounding Box Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(val: float, min_v: float = 0.0, max_v: float = 1.0) -> float:
    return max(min_v, min(max_v, val))


def parse_vlm_ocr_response(raw_text: str, img_w: int, img_h: int) -> list[dict]:
    """
    Parse VLM output into structured OCR results with normalized and pixel bounding boxes.
    Supports:
    1. JSON array of objects: [{"text": "...", "bbox_2d": [ymin, xmin, ymax, xmax]}]
       or dots.ocr layout JSON: [{"category": "Title", "bbox": [x1, y1, x2, y2], "text": "..."}]
    2. Grounding coordinate tags: <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
    3. Plain text line fallback with vertically distributed bounding boxes.
    """
    results: list[dict] = []

    # 1. Attempt JSON parsing
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
        cleaned = cleaned.strip()

    json_data = None
    # Look for a top-level or embedded JSON array [...]
    array_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            json_data = json.loads(array_match.group(0))
        except Exception:
            pass

    # Look for a top-level or embedded JSON object {...}
    if json_data is None:
        obj_match = re.search(r"\{\s*\"(?:layout|elements|items|results|data|boxes|annotations)\".*\}", cleaned, re.DOTALL)
        if obj_match:
            try:
                parsed_obj = json.loads(obj_match.group(0))
                for key in ["elements", "layout", "items", "results", "data", "boxes", "annotations"]:
                    if isinstance(parsed_obj.get(key), list):
                        json_data = parsed_obj[key]
                        break
            except Exception:
                pass

    if json_data is None:
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                json_data = parsed
            elif isinstance(parsed, dict):
                for key in ["elements", "layout", "items", "results", "data", "boxes", "annotations"]:
                    if isinstance(parsed.get(key), list):
                        json_data = parsed[key]
                        break
                if json_data is None:
                    for v in parsed.values():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            json_data = v
                            break
        except Exception:
            pass

    if json_data is None:
        try:
            import ast
            parsed_ast = ast.literal_eval(cleaned)
            if isinstance(parsed_ast, list):
                json_data = parsed_ast
            elif isinstance(parsed_ast, dict):
                for key in ["elements", "layout", "items", "results", "data", "boxes", "annotations"]:
                    if isinstance(parsed_ast.get(key), list):
                        json_data = parsed_ast[key]
                        break
        except Exception:
            pass

    if isinstance(json_data, list) and json_data:
        for item in json_data:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("content") or item.get("label") or "").strip()
            category = str(item.get("category") or item.get("type") or "").strip() or None
            if not text and category:
                if category.lower() != "picture":
                    continue
                text = f"[{category}]"
            elif not text:
                continue

            conf = round(float(item.get("confidence", 0.95)), 4)

            # Box parsing
            box = item.get("bbox") or item.get("bbox_2d") or item.get("box_2d") or item.get("box")
            left, top, width, height = 0.05, 0.05, 0.90, 0.05

            if isinstance(box, (list, tuple)) and len(box) == 4:
                b0, b1, b2, b3 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                max_val = max(b0, b1, b2, b3)
                if max_val <= 1.0:
                    scale_x, scale_y = 1.0, 1.0
                elif max_val <= 1000.0:
                    scale_x, scale_y = 1000.0, 1000.0
                else:
                    scale_x, scale_y = float(img_w), float(img_h)

                # Format detection:
                # dots.ocr and standard document annotations use [x1, y1, x2, y2]
                # Some VLMs (e.g. Qwen2-VL) use [y1, x1, y2, x2]
                span_02 = abs(b2 - b0)
                span_13 = abs(b3 - b1)
                if category is not None or span_02 >= span_13:
                    xmin, xmax = min(b0, b2) / scale_x, max(b0, b2) / scale_x
                    ymin, ymax = min(b1, b3) / scale_y, max(b1, b3) / scale_y
                else:
                    ymin, ymax = min(b0, b2) / scale_y, max(b0, b2) / scale_y
                    xmin, xmax = min(b1, b3) / scale_x, max(b1, b3) / scale_x

                left = _clamp(xmin)
                top = _clamp(ymin)
                width = _clamp(xmax - xmin, min_v=0.005)
                height = _clamp(ymax - ymin, min_v=0.005)

            x1 = int(left * img_w)
            y1 = int(top * img_h)
            x2 = int((left + width) * img_w)
            y2 = int((top + height) * img_h)

            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            bbox_norm = {
                "left": round(left, 6),
                "top": round(top, 6),
                "width": round(width, 6),
                "height": round(height, 6),
            }

            res_entry = {
                "text": text,
                "confidence": conf,
                "bbox": bbox,
                "bbox_norm": bbox_norm,
                "grounded": True,
            }
            if category:
                res_entry["category"] = category
            results.append(res_entry)

        if results:
            return results

    # 2. Attempt Grounding coordinate tags: (ymin,xmin),(ymax,xmax)
    coord_pattern = re.compile(
        r"(?:<\|box_start\|>)?\s*\((\d+),\s*(\d+)\)\s*,\s*\((\d+),\s*(\d+)\)\s*(?:<\|box_end\|>)?\s*([^\n\(\)]+)"
    )
    for match in coord_pattern.finditer(cleaned):
        y1_i, x1_i, y2_i, x2_i, text = match.groups()
        text = text.strip()
        if not text:
            continue
        v0, v1, v2, v3 = float(y1_i), float(x1_i), float(y2_i), float(x2_i)
        if abs(v2 - v0) >= abs(v3 - v1):
            xmin, xmax = min(v0, v2) / 1000.0, max(v0, v2) / 1000.0
            ymin, ymax = min(v1, v3) / 1000.0, max(v1, v3) / 1000.0
        else:
            ymin, ymax = min(v0, v2) / 1000.0, max(v0, v2) / 1000.0
            xmin, xmax = min(v1, v3) / 1000.0, max(v1, v3) / 1000.0

        left = _clamp(xmin)
        top = _clamp(ymin)
        width = _clamp(xmax - xmin, min_v=0.005)
        height = _clamp(ymax - ymin, min_v=0.005)

        x1 = int(left * img_w)
        y1 = int(top * img_h)
        x2 = int((left + width) * img_w)
        y2 = int((top + height) * img_h)

        results.append({
            "text": text,
            "confidence": 0.95,
            "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "bbox_norm": {
                "left": round(left, 6),
                "top": round(top, 6),
                "width": round(width, 6),
                "height": round(height, 6),
            },
            "grounded": True,
        })

    if results:
        return results

    # 3. Fallback: Parse plain text lines with vertically sequential normalized bands
    raw_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    # Strip markdown headers, lists, or bullets if present
    cleaned_lines = []
    for line in raw_lines:
        line_clean = re.sub(r"^[-*#•\d+\.]\s*", "", line).strip()
        if line_clean:
            cleaned_lines.append(line_clean)

    total_lines = max(len(cleaned_lines), 1)
    for i, line_text in enumerate(cleaned_lines):
        top = round(i / total_lines, 6)
        height = round(0.85 / total_lines, 6)
        left = 0.05
        width = 0.90

        x1 = int(left * img_w)
        y1 = int(top * img_h)
        x2 = int((left + width) * img_w)
        y2 = int((top + height) * img_h)

        results.append({
            "text": line_text,
            "confidence": 0.95,
            "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "bbox_norm": {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            },
            "grounded": False,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Extraction Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_rapidocr(image_bytes: bytes) -> dict:
    """
    Run pixel-grounded OCR detection and recognition via RapidOCR (ONNX).
    Uses DBNet text detection to produce tight, non-bleeding bounding boxes around text contours.
    """
    pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = pil_img.size

    engine = get_rapidocr_engine()
    raw_result, _ = engine(image_bytes)

    results: list[dict] = []
    for line in (raw_result or []):
        if not line:
            continue
        bbox_points, text, conf = line
        xs = [float(p[0]) for p in bbox_points]
        ys = [float(p[1]) for p in bbox_points]
        x_min = max(0.0, min(xs))
        x_max = min(float(img_w), max(xs))
        y_min = max(0.0, min(ys))
        y_max = min(float(img_h), max(ys))

        left = round(x_min / img_w, 6)
        top = round(y_min / img_h, 6)
        width = round(max(0.002, (x_max - x_min) / img_w), 6)
        height = round(max(0.002, (y_max - y_min) / img_h), 6)

        bbox = [[int(round(p[0])), int(round(p[1]))] for p in bbox_points]
        bbox_norm = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

        # Structural category detection heuristics
        category = "Text"
        if y_min < 0.05 * img_h:
            category = "Page-header"
        elif x_min < 0.22 * img_w and width < 0.22:
            category = "List-item"

        results.append({
            "text": str(text),
            "confidence": round(float(conf), 4),
            "bbox": bbox,
            "bbox_norm": bbox_norm,
            "grounded": True,
            "category": category,
            "engine": "rapidocr",
        })

    return {
        "width": img_w,
        "height": img_h,
        "engine": "rapidocr",
        "results": results,
    }


async def run_vlm_ocr(image_bytes: bytes, target_model_id: str | None = None) -> dict:
    """Run OCR on raw image bytes using the GGUF Vision-Language Model."""
    # Decode bytes → Image dimensions
    pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = pil_img.size

    # Convert to base64 JPEG
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=95)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    from oprel.server.services.generation import generate_text, GenerateParams

    chosen_model_id = target_model_id or get_active_ocr_model_id()
    model_path, mmproj_path = get_cached_vlm_files(chosen_model_id)

    model_id = chosen_model_id
    if model_path:
        from oprel.downloader.metadata import get_repo_id_from_filename
        from oprel.downloader.aliases import get_best_alias_for_repo
        repo_id = get_repo_id_from_filename(CONFIG.cache_dir, model_path.name)
        if repo_id:
            model_id = get_best_alias_for_repo(repo_id) or repo_id
        elif "glm-ocr" in model_path.name.lower():
            model_id = "glm-ocr"
        elif "deepseek-ocr" in model_path.name.lower():
            model_id = "deepseek-ocr-3b"
        elif "dots.ocr" in model_path.name.lower():
            model_id = "dots-ocr"

    # Specialized OCR models use task-driven triggers; general VLMs use instructional prompts
    m_lower = model_id.lower()
    if "glm-ocr" in m_lower or "glm" in m_lower:
        ocr_prompt = "Text Recognition:"
        system_prompt = None
    elif "deepseek-ocr" in m_lower or "deepseek" in m_lower:
        ocr_prompt = "<|grounding|>OCR this image."
        system_prompt = None
    elif "dots" in m_lower:
        ocr_prompt = (
            "Please output the layout information from the image, including each layout element's bbox, its category, and the corresponding text content within the bbox.\n\n"
            "1. Bbox format: [x1, y1, x2, y2]\n"
            "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n"
            "3. Text Extraction & Formatting Rules:\n"
            "   - Picture: For the 'Picture' category, the text field should be omitted.\n"
            "   - Formula: Format its text as LaTeX.\n"
            "   - Table: Format its text as HTML.\n"
            "   - All Others (Text, Title, etc.): Format their text as Markdown.\n"
            "4. Constraints:\n"
            "   - The output text must be the original text from the image, with no translation.\n"
            "   - All layout elements must be sorted according to human reading order.\n"
            "5. Final Output: The entire output must be a single JSON object."
        )
        system_prompt = None
    else:
        ocr_prompt = (
            "Transcribe all text from this image accurately in natural reading order.\n"
            "Requirements:\n"
            "1. Extract ALL text in ALL languages and scripts present verbatim (including Tamil, Hindi, other Indic scripts, English, etc.) without omission or translation.\n"
            "2. Capture all names, dates, numbers, codes, labels, headers, and footers in their original script.\n"
            "3. For each text segment, provide the 2D bounding box [xmin, ymin, xmax, ymax] scaled between 0 and 1000.\n"
            "Return ONLY a JSON array of objects:\n"
            '[\n  {"text": "transcribed text in native script", "bbox_2d": [xmin, ymin, xmax, ymax]}\n]'
        )
        system_prompt = (
            "You are an expert multilingual document OCR engine. "
            "Transcribe all printed and handwritten text verbatim in its native Unicode script "
            "(including Tamil, Hindi, and all regional scripts) with accurate bounding boxes. "
            "Do not omit, translate, or abbreviate any text."
        )

    params = GenerateParams(
        model_id=model_id,
        prompt=ocr_prompt,
        max_tokens=4096,
        temperature=0.05,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        stream=False,
        images=[b64_str],
        conversation_id=None,
        system_prompt=system_prompt,
        reset_conversation=False,
        thinking=False,
        rag=False,
    )

    old_opt = CONFIG.vision_optimization_enabled
    try:
        # Preserve original image resolution for accurate character recognition
        CONFIG.vision_optimization_enabled = False
        gen_res = await generate_text(params)
    finally:
        CONFIG.vision_optimization_enabled = old_opt

    raw_text = gen_res.text if hasattr(gen_res, "text") else str(gen_res)
    results = parse_vlm_ocr_response(raw_text, img_w, img_h)
    for r in results:
        r.setdefault("engine", model_id)
    return {"width": img_w, "height": img_h, "engine": model_id, "results": results}


def _calculate_iou(b1: dict[str, float], b2: dict[str, float]) -> float:
    """Calculate Intersection-over-Union (IoU) between two normalized boxes."""
    l1, t1, w1, h1 = b1["left"], b1["top"], b1["width"], b1["height"]
    l2, t2, w2, h2 = b2["left"], b2["top"], b2["width"], b2["height"]

    r1, bot1 = l1 + w1, t1 + h1
    r2, bot2 = l2 + w2, t2 + h2

    inter_l = max(l1, l2)
    inter_t = max(t1, t2)
    inter_r = min(r1, r2)
    inter_b = min(bot1, bot2)

    if inter_r <= inter_l or inter_b <= inter_t:
        return 0.0

    inter_area = (inter_r - inter_l) * (inter_b - inter_t)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


async def run_auto_ocr(image_bytes: bytes) -> dict:
    """
    Auto (Hybrid DBNet + VLM) mode:
    1. Fast pass: Run RapidOCR for 100% pixel-grounded, tight bounding boxes (DBNet).
    2. Semantic pass: If a VLM (dots-ocr) is available in cache, run VLM to extract document layout,
       hierarchical categories (Title, Table, Section-header, List-item), and reading structure.
    3. Spatial Fusion: Enrich RapidOCR's pixel-accurate bounding boxes with the VLM's semantic layout classes.
    """
    rapid_res = run_rapidocr(image_bytes)
    rapid_results = rapid_res.get("results", [])

    # Check if a VLM is installed and ready in cache
    m_path, p_path = get_cached_vlm_files("dots-ocr")
    vlm_ready = bool(m_path and p_path and m_path.exists() and p_path.exists())

    if not vlm_ready:
        rapid_res["engine"] = "auto"
        for r in rapid_results:
            r["engine"] = "auto"
        return rapid_res

    # VLM is available: run semantic enrichment pass
    try:
        logger.info("Auto Hybrid: Running VLM semantic enrichment pass (dots-ocr)...")
        vlm_res = await run_vlm_ocr(image_bytes, target_model_id="dots-ocr")
        vlm_items = vlm_res.get("results", [])

        # Enrich each RapidOCR box with VLM's semantic category
        for r in rapid_results:
            r["engine"] = "auto"
            r_norm = r.get("bbox_norm")
            if not r_norm:
                continue
            cx = r_norm["left"] + r_norm["width"] / 2.0
            cy = r_norm["top"] + r_norm["height"] / 2.0

            best_category = None
            best_iou = 0.0

            for v in vlm_items:
                v_norm = v.get("bbox_norm")
                if not v_norm:
                    continue
                v_cat = v.get("category")
                if not v_cat or v_cat == "Text":
                    continue

                vl = v_norm["left"]
                vt = v_norm["top"]
                vr = vl + v_norm["width"]
                vb = vt + v_norm["height"]

                # Prioritize containment (RapidOCR text line is inside VLM layout element like Table or Header)
                if vl <= cx <= vr and vt <= cy <= vb:
                    best_category = v_cat
                    break

                # Overlap metric fallback
                iou = _calculate_iou(r_norm, v_norm)
                if iou > best_iou and iou > 0.10:
                    best_iou = iou
                    best_category = v_cat

            if best_category:
                r["category"] = best_category

        rapid_res["engine"] = "auto"
        return rapid_res
    except Exception as exc:
        logger.warning(f"Auto Hybrid VLM pass skipped due to error: {exc}")
        rapid_res["engine"] = "auto"
        for r in rapid_results:
            r["engine"] = "auto"
        return rapid_res


async def run_ocr(image_bytes: bytes, target_model_id: str | None = None) -> dict:
    """
    Main OCR pipeline supporting Dual-Mode architecture:
    1. RapidOCR (Default) - Pixel detection for screenshots and documents.
    2. dots-ocr / VLM - Visual reasoning and Markdown extraction.
    3. Auto - RapidOCR pixel bounding boxes with auto engine tagging.
    """
    if not is_ocr_ready():
        raise RuntimeError("OCR engine is not ready. Please install RapidOCR or an OCR model first.")

    chosen_model_id = (target_model_id or get_active_ocr_model_id()).lower()

    if "rapid" in chosen_model_id or chosen_model_id == "rapidocr":
        return run_rapidocr(image_bytes)

    if chosen_model_id == "auto":
        return await run_auto_ocr(image_bytes)

    return await run_vlm_ocr(image_bytes, target_model_id=chosen_model_id)


def run_ocr_sync(image_bytes: bytes) -> dict:
    """Synchronous wrapper for run_ocr."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, run_ocr(image_bytes)).result()
        return loop.run_until_complete(run_ocr(image_bytes))
    except RuntimeError:
        return asyncio.run(run_ocr(image_bytes))


def image_to_data_url(image_bytes: bytes, content_type: str = "image/png") -> str:
    """Convert raw bytes to a base64 data URL for storage in the DB."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{content_type};base64,{b64}"
