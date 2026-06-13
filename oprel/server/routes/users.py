from __future__ import annotations

from fastapi import APIRouter

from oprel.server.schemas.common import UserProfile, UserSettings
from oprel.server.services import users as user_service

router = APIRouter()


@router.get("/user", response_model=UserProfile | None)
async def get_user_profile():
    return user_service.get_user_profile()


@router.post("/user", response_model=UserProfile)
async def update_user_profile(user: UserProfile):
    return user_service.update_user_profile(user.name, user.role)


@router.get("/user/settings", response_model=UserSettings)
async def get_user_settings():
    settings = user_service.get_user_settings() or {}
    if not settings:
        return UserSettings()
    return UserSettings(**settings)


@router.post("/user/settings", response_model=UserSettings)
async def update_user_settings(settings: UserSettings):
    user_service.update_user_settings(settings.dict())
    return settings
