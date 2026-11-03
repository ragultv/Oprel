"""Command line for the Oprel MCP servers (`python -m oprel.mcp_servers` / `oprel mcp`)."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="mcp_command")

    serve = sub.add_parser("serve", help="Run an integration as an MCP server over stdio")
    serve.add_argument("integration", help="Integration id, e.g. azure_devops")
    serve.add_argument("--read-only", action="store_true", help="Only expose tools that don't change data")
    serve.add_argument(
        "--env", action="store_true",
        help="Read credentials from environment variables (e.g. AZURE_DEVOPS_PAT) instead of Oprel's keychain",
    )
    serve.add_argument("--user-id", type=int, default=1, help="Oprel user whose connection to use")

    config = sub.add_parser("config", help="Print the JSON to add this server to an MCP client")
    config.add_argument("integration")
    config.add_argument("--client", choices=("mcpServers", "vscode"), default="mcpServers",
                        help="mcpServers: Claude Desktop, Cursor, Windsurf; vscode: VS Code mcp.json")
    config.add_argument("--read-only", action="store_true")

    sub.add_parser("list", help="List integrations that can run as MCP servers")


def run(args: argparse.Namespace) -> int:
    command = getattr(args, "mcp_command", None)
    if command == "serve":
        return _serve(args)
    if command == "config":
        from oprel.mcp_servers import client_config

        integration = _integration(args.integration)
        if not integration:
            return 2
        print(json.dumps(client_config(integration.id, read_only=args.read_only)["clients"][args.client], indent=2))
        return 0
    if command == "list":
        from oprel.integrations.registry import list_integrations, load_builtin_integrations

        load_builtin_integrations()
        for i in list_integrations():
            print(f"{i.id:20} {i.name}")
        return 0
    print("Usage: oprel mcp {serve,config,list} ...", file=sys.stderr)
    return 2


def _integration(integration_id: str):
    from oprel.integrations.registry import get_integration, list_integrations, load_builtin_integrations

    load_builtin_integrations()
    integration = get_integration(integration_id) or get_integration(integration_id.replace("-", "_"))
    if not integration:
        known = ", ".join(i.id for i in list_integrations())
        print(f"Unknown integration '{integration_id}'. Available: {known}", file=sys.stderr)
    return integration


def _serve(args: argparse.Namespace) -> int:
    # stdout carries the MCP protocol: anything printed there by an import would corrupt it.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with contextlib.redirect_stdout(sys.stderr):
        from oprel.mcp_servers import mcp_sdk_installed

        if not mcp_sdk_installed():
            print('The MCP SDK is not installed. Run: pip install "oprel[mcp]"', file=sys.stderr)
            return 1

        import anyio

        from oprel.mcp_servers.bridge import IntegrationMCPServer
        from oprel.mcp_servers.credentials import CredentialError, EnvCredentials, StoredConnection
        from oprel.utils.redaction import install_log_redaction

        install_log_redaction()
        integration = _integration(args.integration)
        if not integration:
            return 2
        try:
            credentials = EnvCredentials(integration) if args.env else StoredConnection(integration, args.user_id)
        except CredentialError as exc:
            print(f"Cannot start the {integration.name} MCP server: {exc}", file=sys.stderr)
            return 1
        server = IntegrationMCPServer(integration, credentials, read_only=args.read_only)
        logging.getLogger("oprel.mcp_servers").info(
            f"{integration.name} MCP server using {credentials.description}"
            f"{' (read-only)' if args.read_only else ''}; tools: {', '.join(t.name for t in server.visible_tools())}"
        )

    try:
        anyio.run(server.run_stdio)
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m oprel.mcp_servers", description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))
