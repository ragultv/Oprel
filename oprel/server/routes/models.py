from __future__ import annotations

from fastapi import APIRouter, HTTPException

from oprel.server.schemas.common import LoadRequest, LoadResponse, ModelInfo, UnloadRequest, UnloadResponse
from oprel.server.services import models as model_service

router = APIRouter()


@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    return [ModelInfo(**m.__dict__) for m in model_service.list_models()]


@router.get("/registry/models")
async def list_registry_models():
    return model_service.list_registry_models()


@router.get("/models/info/{model_id:path}")
async def get_model_detailed_info(model_id: str):
    try:
        return model_service.get_model_detailed_info(model_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model info: {str(exc)}")


@router.get("/models/{model_id:path}/local-quants")
async def get_local_quantizations(model_id: str):
    try:
        return model_service.get_local_quantizations(model_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch local quantizations: {str(exc)}")


@router.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest):
    try:
        result = model_service.load_model(
            request.model_id,
            quantization=request.quantization,
            max_memory_mb=request.max_memory_mb,
            backend=request.backend,
        )
        return LoadResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(exc)}")


@router.post("/unload", response_model=UnloadResponse)
async def unload_model_post(request: UnloadRequest):
    try:
        result = model_service.unload_model(request.model_id)
        return UnloadResponse(**result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/unload/{model_id:path}", response_model=UnloadResponse)
async def unload_model(model_id: str):
    try:
        result = model_service.unload_model(model_id)
        return UnloadResponse(**result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/models/{model_id:path}/quant/{quantization}")
async def delete_model_quant(model_id: str, quantization: str):
    try:
        return model_service.delete_model_quant(model_id, quantization)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
