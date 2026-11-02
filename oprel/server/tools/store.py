"""
SQLite persistence for integration connections, tool policies, pending approvals
and the tool audit log. Tables are created in oprel.server.db.init_db().

Nothing in here ever sees a secret value — only `secret_ref` and a display hint.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from oprel.server.db import get_db

DEFAULT_USER_ID = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def _row_to_connection(row) -> dict[str, Any]:
    d = dict(row)
    d["config"] = _loads(d.pop("config_json", None), {})
    d["enabled"] = bool(d.get("enabled"))
    return d


def get_connection(connection_id: str, user_id: int = DEFAULT_USER_ID) -> dict[str, Any] | None:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM integration_connections WHERE id = ? AND user_id = ?", (connection_id, user_id)
        ).fetchone()
        return _row_to_connection(row) if row else None
    finally:
        conn.close()


def get_connection_by_type(integration_type: str, user_id: int = DEFAULT_USER_ID) -> dict[str, Any] | None:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM integration_connections WHERE integration_type = ? AND user_id = ?",
            (integration_type, user_id),
        ).fetchone()
        return _row_to_connection(row) if row else None
    finally:
        conn.close()


def list_connections(user_id: int = DEFAULT_USER_ID) -> list[dict[str, Any]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM integration_connections WHERE user_id = ? ORDER BY created_at", (user_id,)
        ).fetchall()
        return [_row_to_connection(r) for r in rows]
    finally:
        conn.close()


def upsert_connection(
    *,
    integration_type: str,
    display_name: str,
    config: dict[str, Any],
    secret_ref: str | None,
    secret_hint: str,
    status: str,
    status_message: str,
    user_id: int = DEFAULT_USER_ID,
) -> dict[str, Any]:
    now = _now()
    verified_at = now if status == "connected" else None
    existing = get_connection_by_type(integration_type, user_id)
    conn = get_db()
    try:
        if existing:
            conn.execute(
                """UPDATE integration_connections
                   SET display_name = ?, config_json = ?, secret_ref = ?, secret_hint = ?, status = ?,
                       status_message = ?, last_verified_at = COALESCE(?, last_verified_at), updated_at = ?
                   WHERE id = ?""",
                (display_name, json.dumps(config), secret_ref, secret_hint, status, status_message,
                 verified_at, now, existing["id"]),
            )
            connection_id = existing["id"]
        else:
            connection_id = f"conn_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """INSERT INTO integration_connections
                   (id, user_id, integration_type, display_name, config_json, secret_ref, secret_hint,
                    status, status_message, enabled, last_verified_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (connection_id, user_id, integration_type, display_name, json.dumps(config), secret_ref,
                 secret_hint, status, status_message, verified_at, now, now),
            )
        conn.commit()
    finally:
        conn.close()
    return get_connection(connection_id, user_id)  # type: ignore[return-value]


def update_connection_status(connection_id: str, status: str, message: str, config: dict[str, Any] | None = None) -> None:
    now = _now()
    conn = get_db()
    try:
        if config is not None:
            conn.execute(
                "UPDATE integration_connections SET status = ?, status_message = ?, config_json = ?, "
                "last_verified_at = CASE WHEN ? = 'connected' THEN ? ELSE last_verified_at END, updated_at = ? WHERE id = ?",
                (status, message, json.dumps(config), status, now, now, connection_id),
            )
        else:
            conn.execute(
                "UPDATE integration_connections SET status = ?, status_message = ?, "
                "last_verified_at = CASE WHEN ? = 'connected' THEN ? ELSE last_verified_at END, updated_at = ? WHERE id = ?",
                (status, message, status, now, now, connection_id),
            )
        conn.commit()
    finally:
        conn.close()


def set_connection_enabled(connection_id: str, enabled: bool) -> None:
    conn = get_db()
    try:
        conn.execute(
            "UPDATE integration_connections SET enabled = ?, updated_at = ? WHERE id = ?",
            (1 if enabled else 0, _now(), connection_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_connection(connection_id: str) -> None:
    conn = get_db()
    try:
        conn.execute("DELETE FROM tool_policies WHERE connection_id = ?", (connection_id,))
        conn.execute("DELETE FROM tool_approvals WHERE connection_id = ? AND status = 'pending'", (connection_id,))
        conn.execute("DELETE FROM integration_connections WHERE id = ?", (connection_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool policies
# ---------------------------------------------------------------------------

def get_policies(connection_id: str) -> dict[str, dict[str, Any]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT tool_name, enabled, approval_mode FROM tool_policies WHERE connection_id = ?", (connection_id,)
        ).fetchall()
        return {r["tool_name"]: {"enabled": bool(r["enabled"]), "approval_mode": r["approval_mode"]} for r in rows}
    finally:
        conn.close()


def set_policy(connection_id: str, tool_name: str, enabled: bool, approval_mode: str) -> None:
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO tool_policies (connection_id, tool_name, enabled, approval_mode, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(connection_id, tool_name)
               DO UPDATE SET enabled = excluded.enabled, approval_mode = excluded.approval_mode,
                             updated_at = excluded.updated_at""",
            (connection_id, tool_name, 1 if enabled else 0, approval_mode, _now()),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pending approvals
# ---------------------------------------------------------------------------

APPROVAL_TTL = timedelta(minutes=15)


def _row_to_approval(row) -> dict[str, Any]:
    d = dict(row)
    d["arguments"] = _loads(d.pop("args_json", None), {})
    d["loop_state"] = _loads(d.pop("loop_state_json", None), {})
    return d


def create_approval(
    *,
    conversation_id: str | None,
    connection_id: str | None,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    loop_state: dict[str, Any],
    user_id: int = DEFAULT_USER_ID,
) -> str:
    approval_id = f"apr_{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc)
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO tool_approvals
               (id, user_id, conversation_id, connection_id, tool_call_id, tool_name, args_json, status,
                loop_state_json, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
            (approval_id, user_id, conversation_id, connection_id, tool_call_id, tool_name,
             json.dumps(arguments), json.dumps(loop_state), now.isoformat(), (now + APPROVAL_TTL).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return approval_id


def get_approval(approval_id: str, user_id: int = DEFAULT_USER_ID) -> dict[str, Any] | None:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM tool_approvals WHERE id = ? AND user_id = ?", (approval_id, user_id)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    approval = _row_to_approval(row)
    if approval["status"] == "pending" and datetime.fromisoformat(approval["expires_at"]) < datetime.now(timezone.utc):
        claim_approval(approval_id, "expired")
        approval["status"] = "expired"
    return approval


def claim_approval(approval_id: str, new_status: str) -> bool:
    """Atomically move a pending approval to `new_status`. False if it was already decided."""
    conn = get_db()
    try:
        cur = conn.execute(
            "UPDATE tool_approvals SET status = ?, decided_at = ? WHERE id = ? AND status = 'pending'",
            (new_status, _now(), approval_id),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def list_pending_approvals(conversation_id: str | None = None, user_id: int = DEFAULT_USER_ID) -> list[dict[str, Any]]:
    conn = get_db()
    try:
        if conversation_id:
            rows = conn.execute(
                "SELECT * FROM tool_approvals WHERE user_id = ? AND conversation_id = ? AND status = 'pending' "
                "ORDER BY created_at",
                (user_id, conversation_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tool_approvals WHERE user_id = ? AND status = 'pending' ORDER BY created_at",
                (user_id,),
            ).fetchall()
    finally:
        conn.close()
    now = datetime.now(timezone.utc)
    out = []
    for r in rows:
        a = _row_to_approval(r)
        if datetime.fromisoformat(a["expires_at"]) >= now:
            a.pop("loop_state", None)  # internal; not for API consumers
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def add_audit_entry(
    *,
    conversation_id: str | None,
    connection_id: str | None,
    tool_name: str,
    arguments: dict[str, Any],
    status: str,
    approval: str,
    duration_ms: float,
    result_summary: str,
    user_id: int = DEFAULT_USER_ID,
) -> None:
    from oprel.utils.redaction import redact

    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO tool_audit_log
               (id, user_id, conversation_id, connection_id, tool_name, args_json, status, approval,
                duration_ms, result_summary, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"aud_{uuid.uuid4().hex[:16]}", user_id, conversation_id, connection_id, tool_name,
             redact(json.dumps(arguments))[:8000], status, approval, duration_ms,
             redact(result_summary)[:2000], _now()),
        )
        conn.commit()
    finally:
        conn.close()


def list_audit_entries(limit: int = 100, user_id: int = DEFAULT_USER_ID) -> list[dict[str, Any]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM tool_audit_log WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, int(limit)),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["arguments"] = _loads(d.pop("args_json", None), {})
        out.append(d)
    return out
