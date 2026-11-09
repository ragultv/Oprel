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
    """
    Live verification of Hugging Face access token status.
    Returns status: 'valid' (verified active), 'expired' (401/403 unauthorized), or 'missing'.
    """
    auth = db.get_hf_auth()
    token = (auth.get("token") if auth else None) or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token or not token.strip():
        return {
            "connected": False,
            "status": "missing",
            "message": "No Hugging Face token configured.",
            "username": None,
            "display_name": None,
            "avatar_url": None,
            "email": None,
            "masked_token": None,
            "connected_at": None,
        }

    token = token.strip()
    whoami_url = "https://huggingface.co/api/whoami-v2"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Oprel-Studio/1.0",
    }

    try:
        resp = requests.get(whoami_url, headers=headers, timeout=4.0)
        if resp.status_code == 200:
            data = resp.json()
            username = data.get("name") or data.get("username") or (auth.get("username") if auth else "HF User")
            display_name = data.get("fullname") or username
            avatar_url = data.get("avatarUrl") or (auth.get("avatar_url") if auth else "")
            email = data.get("email") or (auth.get("email") if auth else "")

            # Sync updated details into local DB cache
            if auth and (auth.get("username") != username or auth.get("display_name") != display_name):
                db.save_hf_auth(
                    token=token,
                    username=username,
                    display_name=display_name,
                    avatar_url=avatar_url,
                    email=email,
                )

            return {
                "connected": True,
                "status": "valid",
                "message": "Hugging Face token is active and valid.",
                "username": username,
                "display_name": display_name,
                "avatar_url": avatar_url,
                "email": email,
                "masked_token": _mask_token(token),
                "connected_at": auth.get("connected_at") if auth else None,
            }
        elif resp.status_code in (401, 403):
            # Token is explicitly expired or revoked
            logger.warning("Configured Hugging Face token returned 401/403 Unauthorized (expired)")
            return {
                "connected": False,
                "status": "expired",
                "message": "Hugging Face token has expired or is unauthorized. Please update it.",
                "username": auth.get("username") if auth else None,
                "display_name": auth.get("display_name") if auth else None,
                "avatar_url": auth.get("avatar_url") if auth else None,
                "email": auth.get("email") if auth else None,
                "masked_token": _mask_token(token),
                "connected_at": auth.get("connected_at") if auth else None,
            }
        else:
            # Server error on Hugging Face side; assume cached token valid
            logger.debug(f"HF whoami returned status {resp.status_code}, falling back to cached state")
    except Exception as exc:
        # Offline or network timeout: fall back gracefully to cached state so offline inference works
        logger.debug(f"Hugging Face live status check skipped (offline/timeout): {exc}")

    # Fallback to cached state as valid if token is present
    return {
        "connected": True,
        "status": "valid",
        "message": "Cached token active (offline/unreachable).",
        "username": auth.get("username") if auth else "HF User",
        "display_name": auth.get("display_name") if auth else (auth.get("username") if auth else "HF User"),
        "avatar_url": auth.get("avatar_url") if auth else None,
        "email": auth.get("email") if auth else None,
        "masked_token": _mask_token(token),
        "connected_at": auth.get("connected_at") if auth else None,
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

    existing_user = db.get_user()
    role = existing_user.get("role", "Hugging Face User") if existing_user else "Hugging Face User"
    db.set_user(name=display_name, role=role)

    # Export to environment for huggingface_hub and enable fast transfer
    os.environ["HF_TOKEN"] = raw_token
    _enable_fast_transfer()

    # Auto-configure the AI Provider for cloud inference
    try:
        db.upsert_provider({
            "id": "huggingface",
            "name": "Hugging Face Inference",
            "type": "huggingface",
            "api_key": raw_token,
            "base_url": "https://router.huggingface.co/v1",
            "enabled": True,
            "enabled_model_ids": [],
            "available_model_ids": []
        })
    except Exception as e:
        logger.error(f"Failed to auto-configure HF Inference provider: {e}")

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
