# Connectors: Developer Guide

A **connector** links Oprel to an outside app (Azure DevOps today) so a model can act there: create tasks, update work items, search, comment. Each connector is written once and used in two ways:

1. **In Oprel's own chat**, through native tool calling with approvals (the Tools toggle in chat).
2. **From other apps** (Claude Desktop, VS Code, Cursor, any MCP client), through an MCP server that Oprel generates from the same code.

The UI for both is **Settings → Connectors**.

> Not to be confused with [`oprel/mcp/`](../../oprel/mcp/) (see [mcp_connectors.md](mcp_connectors.md)). That is the parked MCP *client*, which lets Oprel use third-party MCP servers. This guide covers Oprel's own connectors and the MCP *server* that exposes them.

---

## 1. Layout

```
oprel/tools/                  tool primitives, no server imports
  base.py                     Tool, ToolContext, ToolCall, ToolResult, ToolError
  validation.py               argument validation + coercion for sloppy local-model output

oprel/integrations/           one package per connector
  base.py                     Integration ABC, AuthField (the connect form is generated from these)
  registry.py                 BUILTIN_INTEGRATIONS
  azure_devops/               client.py (REST), tools.py (5 tools), markdown_lite.py

oprel/server/tools/           Oprel chat: connections, policies, approvals, agent loop, audit
oprel/server/routes/integrations.py   /integrations/* API (local token required)

oprel/mcp_servers/            MCP server for any Integration
  bridge.py                   Integration -> MCP Server (tools, annotations, redaction, timeouts)
  credentials.py              StoredConnection (default) / EnvCredentials (--env)
  cli.py                      `python -m oprel.mcp_servers ...` and `oprel mcp ...`
```

Adding a connector means writing an `Integration` subclass and registering it in `BUILTIN_INTEGRATIONS`. It then appears in Settings → Connectors, works in chat, and runs as an MCP server, with no frontend or MCP code.

## 2. The MCP server

```
python -m oprel.mcp_servers serve azure_devops              # use the connection saved in Oprel
python -m oprel.mcp_servers serve azure_devops --read-only  # get/search tools only
python -m oprel.mcp_servers serve azure_devops --env        # AZURE_DEVOPS_ORGANIZATION / _PROJECT / _PAT
python -m oprel.mcp_servers config azure_devops [--client vscode]   # print client config JSON
oprel mcp list | serve | config                              # same, via the main CLI
```

It needs the optional extra `pip install "oprel[mcp]"` (MCP SDK 2.x, Python ≥ 3.10). The daemon does not need to be running.

Clients start the server as `<python> -m oprel.mcp_servers serve <id>`. Settings → Connectors → *Use in other apps (MCP)* shows the ready-made snippet (`GET /integrations/{id}/mcp-config`).

### Credentials

| Mode | Where the secret lives | Notes |
|---|---|---|
| **Stored connection** (default) | OS keychain, via the connection saved in Oprel | Nothing secret in the client's config. Re-read on every call, so token rotation and Pause take effect at once. |
| `--env` | Environment variables `<INTEGRATION_ID>_<FIELD_KEY>` | For machines without an Oprel connection. The token then sits in the client's config file. |

### Behaviour and safety

- **Same tools as chat:** `bridge.py` wraps the `Tool` objects from `Integration.tools()`. Arguments go through `oprel.tools.validation` before any request is sent.
- **Approvals belong to the client:** Claude Desktop, VS Code and others ask before running a tool. Oprel's per-tool *Ask first / Auto* setting does not apply here, because there is no Oprel UI to ask in.
  - Tools advertise `readOnlyHint` and `destructiveHint` (`ado_update_work_item` is destructive), so clients can pick sensible defaults.
  - `--read-only` removes the write tools completely.
- **Oprel's tool on/off switches do apply:** a tool switched off is not listed and is refused if called.
- **Output:**
  - Results are JSON, redacted (registered secret values plus credential patterns) and capped at 20,000 characters.
  - Errors are returned as `isError` results and never include credentials.
  - Unexpected crashes are logged to stderr, and the client gets a generic message.
- **Audit:** every call goes into Oprel's tool audit log with `approval = "mcp_client"`. A 401 from Azure marks the connection *expired*, as it does in chat.
- **Prompt injection:** the server instructions repeat the connector's `system_hint` and warn that work-item text is untrusted data.
- **stdout is the protocol:** keep imports quiet. `cli._serve` redirects stdout during startup, and the SDK diverts fd 1 to stderr while serving. `oprel/server/__init__.py` is lazy, so importing `oprel.server.db` does not build the FastAPI app.

## 3. Tests

`tests/unit/tools/`:

- `test_primitives.py`, `test_azure_devops.py`
- `test_api_security.py`: the PAT never leaks; token and origin guards
- `test_tool_loop.py`: Oprel chat approvals
- `test_mcp_server.py`: a real in-process MCP client against the bridge, covering stored and env credentials, tool switches, read-only mode, 401 handling, the audit log and the config snippet

```
pytest tests/unit/tools
```
