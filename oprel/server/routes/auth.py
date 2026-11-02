from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from oprel.server.security.local_auth import (
    TOKEN_HEADER,
    get_local_token,
    is_trusted_browser_request,
)

router = APIRouter()


@router.get("/auth/session")
async def get_session(request: Request):
    """
    Hand the per-process API token to the web UI.

    Only answered for same-origin / explicitly trusted origins, so a third-party page
    cannot obtain it even if CORS has been opened up with OPREL_CORS_ORIGINS=*.
    """
    if not is_trusted_browser_request(request):
        raise HTTPException(status_code=403, detail="Untrusted origin")
    return JSONResponse(
        {"token": get_local_token(), "header": TOKEN_HEADER},
        headers={"Cache-Control": "no-store"},
    )
