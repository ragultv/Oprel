"""
Authentication & Hugging Face Connection API routes — /v1/auth/*
"""
from __future__ import annotations

import logging
import os
import requests
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from oprel.server import db
from oprel.downloader.hub import _enable_fast_transfer

logger = logging.getLogger(__name__)

router = APIRouter()


class HfTokenRequest(BaseModel):
    token: str = Field(..., description="Hugging Face User Access Token (starts with hf_)")


def _mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"


@router.get("/v1/auth/hf/status")
async def get_hf_status():
    """Return whether Hugging Face account is connected and user profile details."""
    auth = db.get_hf_auth()
    if not auth or not auth.get("token"):
        return {
            "connected": False,
            "username": None,
            "display_name": None,
            "avatar_url": None,
            "email": None,
            "masked_token": None,
            "connected_at": None,
        }

    token = auth.get("token", "")
    return {
        "connected": True,
        "username": auth.get("username"),
        "display_name": auth.get("display_name") or auth.get("username"),
        "avatar_url": auth.get("avatar_url"),
        "email": auth.get("email"),
        "masked_token": _mask_token(token),
        "connected_at": auth.get("connected_at"),
    }


@router.post("/v1/auth/hf/token")
async def connect_hf_token(body: HfTokenRequest):
    """
    Validate a Hugging Face user access token against HF API,
    store in local database, and activate fast transfer.
    """
    raw_token = body.token.strip()
    if not raw_token:
        raise HTTPException(status_code=400, detail="Token cannot be empty")

    # Validate against Hugging Face whoami API
    whoami_url = "https://huggingface.co/api/whoami-v2"
    headers = {
        "Authorization": f"Bearer {raw_token}",
        "User-Agent": "Oprel-Studio/1.0",
    }

    try:
        resp = requests.get(whoami_url, headers=headers, timeout=10.0)
    except Exception as exc:
        logger.error(f"Failed to connect to Hugging Face API: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach Hugging Face API to verify token: {exc}"
        )

    if resp.status_code in (401, 403):
        raise HTTPException(
            status_code=400,
            detail="Invalid Hugging Face token. Please check that the token is active and has 'read' access."
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Hugging Face API returned error ({resp.status_code}): {resp.text[:200]}"
        )

    data = resp.json()
    username = data.get("name") or data.get("username") or "HF User"
    display_name = data.get("fullname") or username
    avatar_url = data.get("avatarUrl") or ""
    email = data.get("email") or ""

    # Save to local database
    db.save_hf_auth(
        token=raw_token,
        username=username,
        display_name=display_name,
        avatar_url=avatar_url,
        email=email,
    )

    # Export to environment for huggingface_hub and enable fast transfer
    os.environ["HF_TOKEN"] = raw_token
    _enable_fast_transfer()

    logger.info(f"Successfully connected Hugging Face account: @{username}")

    return {
        "success": True,
        "connected": True,
        "username": username,
        "display_name": display_name,
        "avatar_url": avatar_url,
        "email": email,
        "masked_token": _mask_token(raw_token),
    }


@router.delete("/v1/auth/hf/token")
async def disconnect_hf():
    """Disconnect stored Hugging Face credentials."""
    db.delete_hf_auth()
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    logger.info("Disconnected Hugging Face account")
    return {"success": True, "connected": False}
