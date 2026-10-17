"""
MCP client sessions over stdio.

Each enabled connector runs in its own asyncio task (the MCP SDK requires the
transport context to be entered and exited in the same task).  Servers are
started with a minimal environment: they do NOT inherit the daemon's
environment, so API keys and tokens held by Oprel are not visible to them.

Not implemented yet: OS-level sandboxing (Windows Job Object / AppContainer),
HTTP transports and OAuth.  Timeouts and output caps are enforced here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from oprel.mcp import store

logger = logging.getLogger("oprel.mcp")

CALL_TIMEOUT_SECONDS = 60
START_TIMEOUT_SECONDS = 90  # first run may download the package
MAX_START_ATTEMPTS = 3

# Environment variables a server legitimately needs to start; nothing else is inherited.
_SAFE_ENV_KEYS = (
    "PATH", "SystemRoot", "SYSTEMROOT", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR",
    "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "LANG", "LC_ALL",
)

# connector id -> {"session": ClientSession, "task": Task, "tools": [dict]}
_sessions: Dict[str, Dict[str, Any]] = {}


def minimal_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
    env.update(extra or {})
    return env


PUBLIC_NPM_REGISTRY = "https://registry.npmjs.org"


def resolve_npm_version(package: str, timeout: int = 30) -> str:
    """Ask the public npm registry for the current version so it can be pinned and shown."""
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if not npm:
        raise RuntimeError("Node.js is required for this connector. Install it from nodejs.org and try again.")
    out = subprocess.run(
        [npm, "view", package, "version", f"--registry={PUBLIC_NPM_REGISTRY}"],
        capture_output=True, text=True, timeout=timeout, env=minimal_env(),
    )
    version = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if out.returncode != 0 or not version:
        raise RuntimeError(f"Could not look up the latest version of {package}.")
    return version


def build_npx_command(package: str, version: str, extra_args: List[str]) -> tuple[str, List[str]]:
    """Command and args that run an exact, pinned npm package version."""
    return "npx", ["--yes", f"--registry={PUBLIC_NPM_REGISTRY}", f"{package}@{version}", *extra_args]


def build_command(entry: Dict[str, Any], version: str, extra_args: List[str]) -> tuple[str, List[str]]:
    """Command and args for a catalog entry, pinned to `version`."""
    return build_npx_command(entry["package"], version, extra_args)


def _resolve_executable(command: str, args: List[str]) -> tuple[str, List[str]]:
    path = shutil.which(command) or (shutil.which(command + ".cmd") if sys.platform == "win32" else None)
    if not path:
        raise RuntimeError(f"Could not find '{command}'. Is it installed?")
    if sys.platform == "win32" and path.lower().endswith((".cmd", ".bat")):
        return os.environ.get("COMSPEC", "cmd.exe"), ["/c", path, *args]
    return path, args


def _unwrap(exc: BaseException) -> BaseException:
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return exc


async def _run_session(cid: str, ready: "asyncio.Future[None]") -> None:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    attempt = 0
    while attempt < MAX_START_ATTEMPTS:
        connector = store.get_connector(cid)
        if not connector or not connector["enabled"]:
            break
        store.update_connector(cid, status="connecting")
        try:
            command, args = _resolve_executable(connector["command"], connector["args"])
            params = StdioServerParameters(
                command=command, args=args, env=minimal_env(store.resolve_env(connector))
            )
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as session:
                    await asyncio.wait_for(session.initialize(), START_TIMEOUT_SECONDS)
                    listed = await asyncio.wait_for(session.list_tools(), 30)
                    tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "input_schema": getattr(t, "inputSchema", None) or {},
                            "annotations": t.annotations.model_dump() if getattr(t, "annotations", None) else {},
                        }
                        for t in listed.tools
                    ]
                    _sessions[cid].update(session=session, tools=tools)
                    store.save_tools(cid, tools)
                    try:
                        listed_prompts = await asyncio.wait_for(session.list_prompts(), 15)
                        store.save_prompts(
                            cid, [{"name": p.name, "description": p.description or ""} for p in listed_prompts.prompts]
                        )
                    except Exception:  # not every server offers prompts
                        store.save_prompts(cid, [])
                    store.update_connector(cid, status="connected")
                    logger.info(f"MCP [{connector['name']}] connected, {len(tools)} tools")
                    if not ready.done():
                        ready.set_result(None)
                    await asyncio.Event().wait()  # hold the session open until cancelled
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - includes ExceptionGroup from the SDK
            inner = _unwrap(exc)
            attempt += 1
            logger.warning(f"MCP [{connector['name'] if connector else cid}] failed: {type(inner).__name__}: {inner}")
            if not ready.done() and attempt >= MAX_START_ATTEMPTS:
                ready.set_exception(RuntimeError(f"Could not start connector: {inner}"))
            await asyncio.sleep(min(2 * attempt, 10))
        finally:
            entry = _sessions.get(cid)
            if entry:
                entry.pop("session", None)
                entry["tools"] = []
    store.update_connector(cid, status="error" if attempt else "disconnected")


async def connect(cid: str) -> None:
    """Start a connector and wait until its tools are discovered (or it fails)."""
    if cid in _sessions and "session" in _sessions[cid]:
        return
    await disconnect(cid)
    ready: "asyncio.Future[None]" = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(_run_session(cid, ready), name=f"mcp-{cid}")
    _sessions[cid] = {"task": task, "tools": []}
    try:
        await asyncio.wait_for(asyncio.shield(ready), START_TIMEOUT_SECONDS + 30)
    except Exception:
        await disconnect(cid)
        raise


async def disconnect(cid: str) -> None:
    entry = _sessions.pop(cid, None)
    if entry and entry.get("task"):
        entry["task"].cancel()
        try:
            await entry["task"]
        except BaseException:  # noqa: BLE001
            pass
    if store.get_connector(cid):
        store.update_connector(cid, status="disconnected")


def get_tools(cid: str) -> List[Dict[str, Any]]:
    return list(_sessions.get(cid, {}).get("tools", []))


def all_live_tools() -> List[Dict[str, Any]]:
    """Tools of every connected connector the user switched on for chat, tagged with the connector."""
    out = []
    for c in store.list_connectors():
        if c["enabled"] and c["use_in_chat"] and "session" in _sessions.get(c["id"], {}):
            for t in get_tools(c["id"]):
                out.append({**t, "connector_id": c["id"], "connector_name": c["name"]})
    return out


def has_live_tools() -> bool:
    return any("session" in e and e.get("tools") for e in _sessions.values())


async def call_tool(cid: str, tool: str, arguments: Dict[str, Any]) -> tuple[bool, str, float]:
    """Run one tool call. Returns (ok, text, duration_ms). Never raises."""
    session = _sessions.get(cid, {}).get("session")
    if session is None:
        return False, "The connector is not running.", 0.0
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(session.call_tool(tool, arguments), CALL_TIMEOUT_SECONDS)
        parts = [getattr(b, "text", "") for b in result.content if getattr(b, "type", "") == "text"]
        text = "\n".join(p for p in parts if p)
        return (not getattr(result, "isError", False)), text, (time.perf_counter() - started) * 1000
    except asyncio.TimeoutError:
        return False, f"The tool took longer than {CALL_TIMEOUT_SECONDS} seconds and was stopped.", CALL_TIMEOUT_SECONDS * 1000.0
    except Exception as exc:  # noqa: BLE001
        return False, f"The tool failed: {_unwrap(exc)}", (time.perf_counter() - started) * 1000


async def startup() -> None:
    store.init_tables()
    store.prune_unknown_catalog_connectors()
    for c in store.list_connectors():
        store.update_connector(c["id"], status="disconnected")
        if c["enabled"]:
            asyncio.create_task(_safe_connect(c["id"]))


async def _safe_connect(cid: str) -> None:
    try:
        await connect(cid)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MCP connector {cid} did not start: {exc}")


async def shutdown() -> None:
    for cid in list(_sessions):
        await disconnect(cid)
