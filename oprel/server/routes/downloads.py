from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from oprel.server.schemas.common import PullRequest
from oprel.server.services import downloads as download_service

router = APIRouter()


@router.post("/pull")
async def pull_model_endpoint(request: PullRequest):
    try:
        return download_service.start_download(request.model_id, request.quantization)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start download: {str(exc)}")


@router.get("/downloads/progress")
async def stream_download_progress(id: str):
    try:
        stream = download_service.stream_download_progress(id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return StreamingResponse(
        stream.iterator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/downloads")
async def list_downloads():
    return download_service.list_downloads()


@router.get("/download-logs")
async def list_download_logs(limit: int = 100):
    return download_service.list_download_logs(limit=limit)


@router.get("/downloads/{download_id}")
async def get_download(download_id: str):
    try:
        return download_service.get_download(download_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
