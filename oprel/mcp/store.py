"""SQLite persistence for MCP connectors, discovered tools/prompts, per-tool permissions and the audit log."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from oprel.server import db
from oprel.utils import secrets_store


def init_tables() -> None:
    conn = db.get_db()
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS mcpc_connectors (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            catalog_id TEXT,
            package TEXT,
            version TEXT,
            command TEXT NOT NULL,
            args TEXT NOT NULL DEFAULT '[]',
            env TEXT NOT NULL DEFAULT '{}',
            network TEXT NOT NULL DEFAULT 'unknown',
            enabled INTEGER NOT NULL DEFAULT 1,
            trust_read_only INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'disconnected',
            created_at TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS mcpc_call_logs (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            connector_id TEXT NOT NULL,
            tool TEXT NOT NULL,
            args TEXT,
            decision TEXT NOT NULL,
            ok INTEGER,
            duration_ms REAL,
            error TEXT
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS mcpc_tools (
            connector_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            input_schema TEXT NOT NULL DEFAULT '{}',
            read_only INTEGER NOT NULL DEFAULT 0,
            destructive INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (connector_id, name)
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS mcpc_prompts (
            connector_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (connector_id, name)
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS mcpc_tool_perms (
            connector_id TEXT NOT NULL,
            tool TEXT NOT NULL,
            mode TEXT NOT NULL CHECK (mode IN ('allow','ask','deny')),
            updated_at TEXT,
            PRIMARY KEY (connector_id, tool)
        )"""
    )
    # Older builds stored "always allow" decisions in mcpc_grants: carry them over, then drop the table.
    if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='mcpc_grants'").fetchone():
        c.execute("INSERT OR IGNORE INTO mcpc_tool_perms (connector_id, tool, mode, updated_at) "
                  "SELECT connector_id, tool, 'allow', created_at FROM mcpc_grants")
        c.execute("DROP TABLE mcpc_grants")
    cols = [r[1] for r in c.execute("PRAGMA table_info(mcpc_connectors)")]
    if "use_in_chat" not in cols:
        c.execute("ALTER TABLE mcpc_connectors ADD COLUMN use_in_chat INTEGER NOT NULL DEFAULT 1")
    c.execute("CREATE INDEX IF NOT EXISTS idx_mcpc_logs_ts ON mcpc_call_logs(ts DESC)")
    conn.commit()
    conn.close()


def _row(r) -> Dict[str, Any]:
    d = dict(r)
    d["args"] = json.loads(d.get("args") or "[]")
    d["env"] = json.loads(d.get("env") or "{}")
    d["enabled"] = bool(d["enabled"])
    d.pop("trust_read_only", None)  # superseded by per-tool permissions
    d["use_in_chat"] = bool(d.get("use_in_chat", 1))
    return d


def public(connector: Dict[str, Any]) -> Dict[str, Any]:
    """Connector as shown over the API: secret values never leave the daemon."""
    d = dict(connector)
    d["env"] = sorted(d.get("env", {}).keys())
    return d


def add_connector(
    name: str,
    command: str,
    args: List[str],
    *,
    catalog_id: Optional[str] = None,
    package: Optional[str] = None,
    version: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    network: str = "unknown",
) -> Dict[str, Any]:
    cid = uuid.uuid4().hex[:12]
    stored_env = {k: secrets_store.protect(f"mcp:{cid}:{k}", v) for k, v in (env or {}).items()}
    conn = db.get_db()
    conn.execute(
        """INSERT INTO mcpc_connectors
           (id, name, catalog_id, package, version, command, args, env, network, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (cid, name, catalog_id, package, version, command, json.dumps(args),
         json.dumps(stored_env), network, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return get_connector(cid)


def get_connector(cid: str) -> Optional[Dict[str, Any]]:
    conn = db.get_db()
    r = conn.execute("SELECT * FROM mcpc_connectors WHERE id = ?", (cid,)).fetchone()
    conn.close()
    return _row(r) if r else None


def list_connectors() -> List[Dict[str, Any]]:
    conn = db.get_db()
    rows = conn.execute("SELECT * FROM mcpc_connectors ORDER BY created_at").fetchall()
    conn.close()
    return [_row(r) for r in rows]


def prune_unknown_catalog_connectors() -> int:
    """Remove installed connectors whose catalog entry was retired (their stored secrets go too).
    Custom connectors (no catalog id) are never touched."""
    from oprel.mcp.registry import get_catalog_entry

    removed = 0
    for c in list_connectors():
        if c.get("catalog_id") and get_catalog_entry(c["catalog_id"]) is None:
            delete_connector(c["id"])
            removed += 1
    return removed


def resolve_env(connector: Dict[str, Any]) -> Dict[str, str]:
    return {k: secrets_store.reveal(v) for k, v in connector.get("env", {}).items()}


def update_connector(cid: str, **fields: Any) -> None:
    allowed = {"enabled", "use_in_chat", "status"}
    sets = {k: (int(v) if isinstance(v, bool) else v) for k, v in fields.items() if k in allowed}
    if not sets:
        return
    conn = db.get_db()
    conn.execute(
        f"UPDATE mcpc_connectors SET {', '.join(k + ' = ?' for k in sets)} WHERE id = ?",
        (*sets.values(), cid),
    )
    conn.commit()
    conn.close()


def delete_connector(cid: str) -> None:
    c = get_connector(cid)
    if c:
        for v in c["env"].values():
            secrets_store.forget(v)
    conn = db.get_db()
    conn.execute("DELETE FROM mcpc_connectors WHERE id = ?", (cid,))
    for table in ("mcpc_tools", "mcpc_prompts", "mcpc_tool_perms"):
        conn.execute(f"DELETE FROM {table} WHERE connector_id = ?", (cid,))
    conn.commit()
    conn.close()


# -- discovered tools and prompts (cached so the UI can show them when a connector is off) -------

def save_tools(cid: str, tools: List[Dict[str, Any]]) -> None:
    """Replace the cached tool list of a connector with what the server reported."""
    conn = db.get_db()
    conn.execute("DELETE FROM mcpc_tools WHERE connector_id = ?", (cid,))
    for t in tools:
        ann = t.get("annotations") or {}
        conn.execute(
            "INSERT OR REPLACE INTO mcpc_tools (connector_id, name, description, input_schema, read_only, destructive)"
            " VALUES (?,?,?,?,?,?)",
            (cid, t["name"], t.get("description", ""), json.dumps(t.get("input_schema") or {}),
             int(ann.get("readOnlyHint") is True), int(ann.get("destructiveHint") is True)),
        )
    conn.commit()
    conn.close()


def get_tools(cid: str) -> List[Dict[str, Any]]:
    conn = db.get_db()
    rows = conn.execute("SELECT * FROM mcpc_tools WHERE connector_id = ? ORDER BY name", (cid,)).fetchall()
    conn.close()
    return [
        {"name": r["name"], "description": r["description"], "input_schema": json.loads(r["input_schema"] or "{}"),
         "read_only": bool(r["read_only"]), "destructive": bool(r["destructive"])}
        for r in rows
    ]


def save_prompts(cid: str, prompts: List[Dict[str, Any]]) -> None:
    conn = db.get_db()
    conn.execute("DELETE FROM mcpc_prompts WHERE connector_id = ?", (cid,))
    for p in prompts:
        conn.execute("INSERT OR REPLACE INTO mcpc_prompts (connector_id, name, description) VALUES (?,?,?)",
                     (cid, p["name"], p.get("description", "")))
    conn.commit()
    conn.close()


def get_prompts(cid: str) -> List[Dict[str, Any]]:
    conn = db.get_db()
    rows = conn.execute("SELECT name, description FROM mcpc_prompts WHERE connector_id = ? ORDER BY name", (cid,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -- per-tool permissions -------------------------------------------------------------------------
# Modes: "allow" (run without asking), "ask" (approval each time), "deny" (never run).
# Without a stored mode: read-only tools of a catalog connector are allowed, everything else asks.

MODES = ("allow", "ask", "deny")


def default_mode(connector: Dict[str, Any], read_only: bool) -> str:
    return "allow" if connector.get("catalog_id") and read_only else "ask"


def stored_modes(cid: str) -> Dict[str, str]:
    conn = db.get_db()
    rows = conn.execute("SELECT tool, mode FROM mcpc_tool_perms WHERE connector_id = ?", (cid,)).fetchall()
    conn.close()
    return {r["tool"]: r["mode"] for r in rows}


def effective_mode(connector: Dict[str, Any], tool: str, read_only: bool) -> str:
    return stored_modes(connector["id"]).get(tool) or default_mode(connector, read_only)


def set_tool_mode(cid: str, tool: str, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"Unknown permission mode: {mode}")
    conn = db.get_db()
    conn.execute(
        "INSERT OR REPLACE INTO mcpc_tool_perms (connector_id, tool, mode, updated_at) VALUES (?,?,?,?)",
        (cid, tool, mode, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def set_group_mode(cid: str, read_only: bool, mode: str) -> int:
    """Apply one mode to every cached tool in a group (read-only vs write/delete)."""
    names = [t["name"] for t in get_tools(cid) if t["read_only"] == read_only]
    for name in names:
        set_tool_mode(cid, name, mode)
    return len(names)


# -- audit log ---------------------------------------------------------------

def log_call(cid: str, tool: str, args: str, decision: str, ok: Optional[bool],
             duration_ms: float = 0.0, error: str = "") -> None:
    conn = db.get_db()
    conn.execute(
        "INSERT INTO mcpc_call_logs (id, ts, connector_id, tool, args, decision, ok, duration_ms, error)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (uuid.uuid4().hex, datetime.now().isoformat(), cid, tool, args, decision,
         None if ok is None else int(ok), duration_ms, error[:500]),
    )
    conn.commit()
    conn.close()


def list_logs(limit: int = 100) -> List[Dict[str, Any]]:
    conn = db.get_db()
    rows = conn.execute("SELECT * FROM mcpc_call_logs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
