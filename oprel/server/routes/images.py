from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from oprel.server.schemas.images import ImageGenerationRequest, ImageGenerationResponse
from oprel.server.services import downloads as download_service
from oprel.server.services.images import (
    generate_image,
    get_image_generation_job,
    list_image_models,
    start_image_generation,
    stream_image_generation_progress,
)

router = APIRouter()


@router.post("/v1/images/generations")
async def v1_images_generations(request: ImageGenerationRequest):
    try:
        data = await generate_image(
            prompt=request.prompt,
            model=request.model,
            response_format=request.response_format,
            size=request.size,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            sampler=request.sampler,
        )
        return ImageGenerationResponse(**data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/v1/images/generations/start")
async def v1_images_generations_start(request: ImageGenerationRequest):
    try:
        return start_image_generation(
            prompt=request.prompt,
            model=request.model,
            response_format=request.response_format,
            size=request.size,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            sampler=request.sampler,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/v1/images/generations/jobs/{job_id}")
async def v1_images_generation_job(job_id: str):
    try:
        return get_image_generation_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/v1/images/generations/progress")
async def v1_images_generation_progress(id: str):
    async def iterator():
        async for event in stream_image_generation_progress(id):
            yield event

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/v1/images/models")
async def v1_images_models():
    try:
        return list_image_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/v1/images/models/pull")
async def v1_images_models_pull(payload: dict):
    model_id = str(payload.get("model_id") or "").strip()
    quantization = str(payload.get("quantization") or "").strip() or None
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    try:
        return download_service.start_image_download(model_id, quantization=quantization)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start image download: {str(exc)}")
