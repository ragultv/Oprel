"""One scene recipe per conversation, versioned so the UI knows when to reload the mesh."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from oprel.server.db import get_db


def get_scene(conversation_id: str) -> dict[str, Any]:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT objects_json, version, updated_at FROM scene3d_scenes WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return {"conversation_id": conversation_id, "version": 0, "objects": [], "updated_at": None}
    return {
        "conversation_id": conversation_id,
        "version": row["version"],
        "objects": json.loads(row["objects_json"] or "[]"),
        "updated_at": row["updated_at"],
    }


def save_scene(conversation_id: str, objects: list[dict[str, Any]]) -> int:
    """Store the recipe and return the new version number."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO scene3d_scenes (conversation_id, objects_json, version, updated_at)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(conversation_id)
               DO UPDATE SET objects_json = excluded.objects_json, version = version + 1,
                             updated_at = excluded.updated_at""",
            (conversation_id, json.dumps(objects), now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT version FROM scene3d_scenes WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
    finally:
        conn.close()
    return int(row["version"])
