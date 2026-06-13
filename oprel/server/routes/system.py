from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter

from oprel.server.services.model_state import cleanup_models

router = APIRouter()


@router.post("/shutdown")
async def shutdown_server():
    async def shutdown():
        await asyncio.sleep(0.5)
        cleanup_models()
        os._exit(0)

    asyncio.create_task(shutdown())
    return {"status": "shutting down"}
