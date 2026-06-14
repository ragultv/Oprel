"""OCR API routes — /v1/ocr/*"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from oprel.server import db
from oprel.server.services import ocr_service

router = APIRouter()

# ── Status ─────────────────────────────────────────────────────────────────────

@router.get("/v1/ocr/status")
async def ocr_status():
    """Return whether OCR models are ready and disk usage."""
    return ocr_service.get_ocr_status()


# ── Setup / Download ────────────────────────────────────────────────────────────

@router.post("/v1/ocr/setup")
async def ocr_setup():
    """Kick off async model download. Returns immediately."""
    if ocr_service.is_ocr_ready():
        return {"status": "already_ready"}
    ocr_service.start_download_async()
    return {"status": "downloading"}


@router.get("/v1/ocr/setup/progress")
async def ocr_setup_progress():
    """SSE stream of download progress events."""
    async def _generator():
        for chunk in ocr_service.iter_download_progress():
            yield chunk

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Extraction ─────────────────────────────────────────────────────────────────

@router.post("/v1/ocr/extract")
async def ocr_extract(file: UploadFile = File(...)):
    """
    Accept an uploaded image, run PaddleOCR, persist to DB, return result.

    Response shape:
      {
        id: str,
        filename: str,
        results: [{ text, confidence, bbox }],
        full_text: str,
        word_count: int,
        created_at: str
      }
    """
    if not ocr_service.is_ocr_ready():
        raise HTTPException(status_code=503, detail="OCR models not ready. Run /v1/ocr/setup first.")

    # Validate content type
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp", "image/tiff"}
    content_type = file.content_type or ""
    if content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Allowed: {', '.join(allowed)}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB).")

    try:
        ocr_data = ocr_service.run_ocr(image_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    results = ocr_data["results"]
    img_width = ocr_data["width"]
    img_height = ocr_data["height"]

    # Build plain text + word count
    full_text = "\n".join(r["text"] for r in results)
    word_count = len(full_text.split())

    # Persist to DB
    job_id = f"ocr_{uuid.uuid4().hex[:12]}"
    image_data = ocr_service.image_to_data_url(image_bytes, content_type)
    result_json = json.dumps(results)

    db.add_ocr_job(
        job_id=job_id,
        filename=file.filename or "upload",
        image_data=image_data,
        result_json=result_json,
        full_text=full_text,
        word_count=word_count,
    )

    return {
        "id": job_id,
        "filename": file.filename or "upload",
        "img_width": img_width,
        "img_height": img_height,
        "results": results,
        "full_text": full_text,
        "word_count": word_count,
    }


# ── History ─────────────────────────────────────────────────────────────────────

@router.get("/v1/ocr/history")
async def ocr_history(limit: int = 50):
    """Return recent OCR jobs (newest first)."""
    jobs = db.get_ocr_jobs(limit=limit)
    # Parse result_json back to list for the frontend
    for job in jobs:
        try:
            job["results"] = json.loads(job.pop("result_json"))
        except Exception:
            job["results"] = []
    return jobs


@router.delete("/v1/ocr/history/{job_id}")
async def ocr_delete_job(job_id: str):
    """Delete a single OCR job."""
    db.delete_ocr_job(job_id)
    return {"deleted": job_id}
