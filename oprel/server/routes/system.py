from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from oprel.server.services.model_state import cleanup_models
from oprel.downloader.cache import get_storage_info, set_storage_root

router = APIRouter()


class StorageUpdateRequest(BaseModel):
    path: str
    migrate: bool = False


@router.get("/system/storage")
async def get_storage():
    """Get active storage root path and drive capacity/free space."""
    return get_storage_info()


@router.post("/system/storage")
async def update_storage(req: StorageUpdateRequest):
    """Set custom storage path, optionally migrating existing cached models/data."""
    try:
        new_path = set_storage_root(req.path, migrate=req.migrate)
        return {"success": True, "info": get_storage_info()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set storage path: {e}")


@router.post("/shutdown")
async def shutdown_server():
    async def shutdown():
        await asyncio.sleep(0.5)
        cleanup_models()
        os._exit(0)

    asyncio.create_task(shutdown())
    return {"status": "shutting down"}
