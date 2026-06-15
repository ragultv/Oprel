from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from oprel.server import db
from oprel.mcp.registry import get_builtin_connector

logger = logging.getLogger("oprel.mcp")

# connector_id → {"session": ClientSession, "task": asyncio.Task}
_sessions: dict[str, dict[str, Any]] = {}
_locks: dict[str, asyncio.Lock] = {}


def _lock(cid: str) -> asyncio.Lock:
    if cid not in _locks:
        _locks[cid] = asyncio.Lock()
    return _locks[cid]


# ── transport helpers ─────────────────────────────────────────────────────────

async def _get_fresh_token(connector: dict, builtin: dict) -> str:
    """Returns a valid access token, refreshing if needed."""
    from oprel.mcp.oauth import refresh_token
    oauth_cfg = builtin.get("oauth")
    if oauth_cfg:
        return await refresh_token(connector["id"], oauth_cfg)
    cfg = connector.get("config", {})
    return cfg.get("oauth_token") or cfg.get("api_key", "")


@asynccontextmanager
async def _stdio_ctx(connector: dict, builtin: dict):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters

    config = connector.get("config", {})
    command = builtin.get("command") or config.get("command", "")
    template_args: list[str] = builtin.get("args") or []
    args = [a.format(**{k: config.get(k, "") for k in config}) for a in template_args]

    token = await _get_fresh_token(connector, builtin)

    env = dict(os.environ)
    for k, v_tmpl in (builtin.get("env_vars") or {}).items():
        v = v_tmpl.replace("{api_key}", token)
        for ck, cv in config.items():
            v = v.replace(f"{{{ck}}}", str(cv))
        env[k] = v

    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def _sse_ctx(connector: dict, builtin: dict):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    config = connector.get("config", {})
    url = config.get("url") or builtin.get("url", "")
    headers: dict[str, str] = {}
    token = await _get_fresh_token(connector, builtin)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with sse_client(url, headers=headers) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def _http_ctx(connector: dict, builtin: dict):
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        config = connector.get("config", {})
        url = config.get("url") or builtin.get("url", "")
        headers: dict[str, str] = {}
        token = await _get_fresh_token(connector, builtin)
        if token:
            headers["Authorization"] = f"Bearer {token}"
            # Figma requires its own header in addition to Authorization
            if "figma" in url:
                headers["X-Figma-Token"] = token

        async with streamablehttp_client(url, headers=headers) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                yield session
    except ImportError:
        # Fall back to SSE if streamable_http is not available
        async with _sse_ctx(connector, builtin) as session:
            yield session


def _get_transport_ctx(connector: dict, builtin: dict):
    transport = connector.get("transport") or builtin.get("transport", "stdio")
    if transport == "stdio":
        return _stdio_ctx(connector, builtin)
    if transport == "sse":
        return _sse_ctx(connector, builtin)
    if transport == "streamable_http":
        return _http_ctx(connector, builtin)
    raise ValueError(f"Unknown MCP transport: {transport!r}")


# ── session lifecycle ─────────────────────────────────────────────────────────

_RETRY_DELAYS = [2, 5, 15, 30, 60]


async def _run_session(cid: str) -> None:
    attempt = 0
    while True:
        connector = db.get_mcp_connector(cid)
        if not connector or not connector.get("enabled"):
            break

        builtin = get_builtin_connector(connector["builtin_id"]) or {}
        db.set_mcp_connector_status(cid, "connecting")

        try:
            async with _get_transport_ctx(connector, builtin) as session:
                db.set_mcp_connector_status(cid, "connected")
                attempt = 0
                logger.info(f"MCP [{connector['name']}] connected")

                tools_resp = await session.list_tools()
                tools = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
                    }
                    for t in tools_resp.tools
                ]
                db.save_mcp_tools(cid, tools)
                logger.info(f"MCP [{connector['name']}] {len(tools)} tools discovered")

                # Preserve the task reference so disconnect() can cancel it
                existing_task = _sessions.get(cid, {}).get("task")
                _sessions[cid] = {"session": session, "task": existing_task}

                # Keep alive with periodic pings
                while True:
                    await asyncio.sleep(30)
                    try:
                        await session.send_ping()
                    except Exception:
                        break

        except asyncio.CancelledError:
            break
        except Exception as exc:
            # Unwrap ExceptionGroup (raised by asyncio.TaskGroup inside the MCP library)
            inner: BaseException = exc
            if hasattr(exc, "exceptions") and exc.exceptions:
                inner = exc.exceptions[0]
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            logger.warning(f"MCP [{connector.get('name', cid)}] error (retry in {delay}s): {inner}")
            db.set_mcp_connector_status(cid, "error")
            attempt += 1
            await asyncio.sleep(delay)
        finally:
            _sessions.pop(cid, None)

    db.set_mcp_connector_status(cid, "disconnected")


async def connect(cid: str) -> None:
    async with _lock(cid):
        if cid in _sessions:
            return
        task = asyncio.create_task(_run_session(cid), name=f"mcp-{cid}")
        _sessions[cid] = {"task": task}


async def disconnect(cid: str) -> None:
    async with _lock(cid):
        entry = _sessions.pop(cid, None)
        if entry and "task" in entry:
            entry["task"].cancel()
            try:
                await entry["task"]
            except (asyncio.CancelledError, Exception):
                pass
    db.set_mcp_connector_status(cid, "disconnected")


async def reconnect(cid: str) -> None:
    await disconnect(cid)
    await asyncio.sleep(0.3)
    await connect(cid)


def get_live_session(cid: str):
    entry = _sessions.get(cid)
    return entry["session"] if entry and "session" in entry else None


def get_all_live_connector_ids() -> list[str]:
    return [cid for cid, e in _sessions.items() if "session" in e]


# ── startup / shutdown ────────────────────────────────────────────────────────

async def startup() -> None:
    db.init_mcp_tables()
    connectors = db.list_mcp_connectors()
    for c in connectors:
        if c.get("enabled"):
            await connect(c["id"])
    logger.info(f"MCP startup: {len(connectors)} connector(s) loaded")


async def shutdown() -> None:
    for cid in list(_sessions.keys()):
        await disconnect(cid)
    logger.info("MCP shutdown complete")


# ── tool execution ────────────────────────────────────────────────────────────

async def call_tool(
    cid: str,
    tool_name: str,
    arguments: dict,
    conversation_id: str | None = None,
    timeout: float = 60.0,
) -> dict:
    session = get_live_session(cid)
    if session is None:
        raise RuntimeError(
            f"MCP connector '{cid}' is not connected. "
            "Go to Settings → Connectors to connect it first."
        )

    t0 = time.perf_counter()
    error_msg: str | None = None

    try:
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments=arguments),
            timeout=timeout,
        )
        content = []
        for block in result.content:
            if hasattr(block, "text"):
                content.append({"type": "text", "text": block.text})
            elif hasattr(block, "data"):
                content.append({"type": "image", "data": block.data, "mimeType": getattr(block, "mimeType", "image/png")})
            else:
                content.append({"type": "unknown", "raw": str(block)})
        payload = {"content": content, "is_error": bool(getattr(result, "isError", False))}

    except asyncio.TimeoutError:
        error_msg = f"Tool '{tool_name}' timed out after {timeout}s"
        payload = {"content": [{"type": "text", "text": error_msg}], "is_error": True}
    except Exception as exc:
        error_msg = str(exc)
        payload = {"content": [{"type": "text", "text": f"Tool error: {error_msg}"}], "is_error": True}

    db.log_mcp_tool_call(
        connector_id=cid,
        tool_name=tool_name,
        arguments=arguments,
        result=payload,
        error=error_msg,
        duration_ms=(time.perf_counter() - t0) * 1000,
        conversation_id=conversation_id,
    )
    return payload


async def test_connection(cid: str) -> dict:
    connector = db.get_mcp_connector(cid)
    if not connector:
        return {"ok": False, "tools": [], "error": "Connector not found"}

    builtin = get_builtin_connector(connector["builtin_id"]) or {}
    try:
        async with _get_transport_ctx(connector, builtin) as session:
            resp = await asyncio.wait_for(session.list_tools(), timeout=15.0)
            tools = [{"name": t.name, "description": t.description or ""} for t in resp.tools]
            return {"ok": True, "tools": tools, "error": None}
    except Exception as exc:
        return {"ok": False, "tools": [], "error": str(exc)}
