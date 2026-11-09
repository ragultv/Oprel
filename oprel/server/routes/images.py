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


# ── Image Chat History Endpoints ──────────────────────────────────────────

@router.get("/v1/images/chats")
async def list_image_chats():
    try:
        from oprel.server import db
        return db.get_image_chats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/v1/images/chats")
async def create_new_image_chat(payload: dict = None):
    try:
        from oprel.server import db
        title = (payload or {}).get("title") or "New Image Chat"
        chat_id = (payload or {}).get("id")
        return db.create_image_chat(title=title, chat_id=chat_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/v1/images/chats/{chat_id}")
async def get_single_image_chat(chat_id: str):
    try:
        from oprel.server import db
        chat = db.get_image_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Image chat not found")
        return chat
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/v1/images/chats/{chat_id}/messages")
async def add_message_to_image_chat(chat_id: str, payload: dict):
    try:
        from oprel.server import db
        prompt = str(payload.get("prompt") or "").strip()
        image_url = str(payload.get("image_url") or "").strip()
        model = str(payload.get("model") or "").strip()
        size = str(payload.get("size") or "1024x1024").strip()
        negative_prompt = str(payload.get("negative_prompt") or "").strip()
        steps = int(payload.get("steps") or 15)
        cfg_scale = float(payload.get("cfg_scale") or 7.0)
        seed = int(payload.get("seed") or -1)
        sampler = str(payload.get("sampler") or "euler").strip()

        if not prompt or not image_url:
            raise HTTPException(status_code=400, detail="prompt and image_url are required")

        return db.add_image_message(
            chat_id=chat_id,
            prompt=prompt,
            image_url=image_url,
            model=model,
            size=size,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            sampler=sampler,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/v1/images/chats/{chat_id}")
async def remove_image_chat(chat_id: str):
    try:
        from oprel.server import db
        db.delete_image_chat(chat_id)
        return {"status": "ok", "deleted": chat_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/v1/images/chats")
async def clear_all_image_chats():
    try:
        from oprel.server import db
        db.clear_image_chats()
        return {"status": "ok", "cleared": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

