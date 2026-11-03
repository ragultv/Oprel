"""
MCP servers built from Oprel integrations, so other apps (Claude Desktop, VS Code,
Cursor, any MCP client) can use the same tools Oprel's own chat uses.

    python -m oprel.mcp_servers serve azure_devops            # stored Oprel connection
    python -m oprel.mcp_servers serve azure_devops --read-only
    python -m oprel.mcp_servers config azure_devops           # print client config JSON

This module must not import the MCP SDK: the daemon uses `client_config` to show the
setup snippet even when the optional `mcp` extra isn't installed.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any


def mcp_sdk_installed() -> bool:
    return importlib.util.find_spec("mcp") is not None


def server_key(integration_id: str) -> str:
    return integration_id.replace("_", "-")


def launch_command(integration_id: str, *, read_only: bool = False) -> tuple[str, list[str]]:
    """Command an MCP client runs to start the server. Contains no secrets."""
    args = ["-m", "oprel.mcp_servers", "serve", integration_id]
    if read_only:
        args.append("--read-only")
    return sys.executable, args


def client_config(integration_id: str, *, read_only: bool = False) -> dict[str, Any]:
    command, args = launch_command(integration_id, read_only=read_only)
    key = server_key(integration_id)
    return {
        "server_name": key,
        "command": command,
        "args": args,
        "mcp_installed": mcp_sdk_installed(),
        "install_hint": "pip install \"oprel[mcp]\"",
        "clients": {
            # Claude Desktop, Cursor, Windsurf and most other clients
            "mcpServers": {"mcpServers": {key: {"command": command, "args": args}}},
            # VS Code: .vscode/mcp.json or the user-level mcp.json
            "vscode": {"servers": {key: {"type": "stdio", "command": command, "args": args}}},
        },
    }
