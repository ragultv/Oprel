from __future__ import annotations

from oprel.server import db


def get_user_profile():
    return db.get_user()


def update_user_profile(name: str, role: str):
    return db.set_user(name, role)


def get_user_settings():
    settings = db.get_user_settings()
    return settings


def update_user_settings(settings: dict):
    db.set_user_settings(settings)
    return settings
