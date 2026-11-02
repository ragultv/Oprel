# MCP Connectors: Developer Guide (living document)

**Status: built, tested, and parked.** All of the code lives in [`oprel/mcp/`](../../oprel/mcp/). Nothing else in Oprel imports it. The web UI shows "Connectors: coming soon". This guide explains how the feature works, what we learned building it, and exactly how to switch it back on.

Update this file whenever you change anything under `oprel/mcp/`. If a statement here and the code disagree, the code wins; fix the document.

---

## 1. What it is

MCP (Model Context Protocol) lets an AI app discover and call tools exposed by separate programs ("MCP servers"). Oprel's connectors let a local or cloud model use those tools, with the user in control:

- Connectors are **local stdio MCP servers** started and supervised by the Oprel daemon.
- Every tool has a **permission**: *Always allow*, *Needs approval*, or *Blocked*.
- Every call is **validated, permission-checked, optionally approved by the user, time-limited, sanitised and logged**.
- The model's reply always **ends with a plain-text answer**; tool calls appear as collapsed cards.

What it is *not*: a way to make anything "offline". A local control plane (Oprel, the policy, the approvals, the audit log) says nothing about what a connector does. A connector can talk to the internet, and the chat model may be a cloud model that receives tool results. The catalog marks each connector `network: none` or `internet`, and the approval dialog says so.

Default catalog (`registry.py`): **Files** (`@modelcontextprotocol/server-filesystem`), **Memory** (`@modelcontextprotocol/server-memory`), **Step-by-step thinking** (`@modelcontextprotocol/server-sequential-thinking`). All run through `npx`.

---

## 2. Layout

```
oprel/mcp/
  registry.py        catalog of installable connectors (+ usage hints) and the tool-list preview loader
  catalog_tools.json snapshot of each catalog server's real tools/prompts (used before connecting)
  store.py           SQLite: connectors, cached tools/prompts, per-tool permissions, audit log
  manager.py         starts/stops stdio sessions, discovers tools, runs tool calls (timeout, minimal env)
  policy.py          pure helpers: schema validation, redaction, output sanitising, card stripping
  approvals.py       in-memory queue of calls waiting for the user's decision
  tool_loop.py       the model <-> tool loop (local and cloud variants) + prompt + resilience
  routes.py          FastAPI router (prefix /mcp) used by the UI
  integration.py     glue for Oprel's generation/provider code (see section 9)
  webui/             the React pieces (see section 8)
  tests/             68 tests, including a real stdio MCP server end to end
```

Run the tests with `pytest oprel/mcp/tests` (they are outside the default `testpaths`, so CI does not run them while the feature is parked).

Packaging: `pyproject.toml` excludes `oprel.mcp*` from the wheel and the `mcp` extra was removed. Restore both when shipping (section 9).

Runtime dependencies of the parked code: `mcp>=1.2` (the official Python SDK; 1.26 was used), `jsonschema>=4`, and Node.js for the catalog servers.

---

## 3. Data model (SQLite, same file as chats: `chat_history.db`)

| Table | Purpose |
| --- | --- |
| `mcpc_connectors` | one row per installed connector: `catalog_id`, pinned `version`, `command`, `args`, `env` (secret values are keyring references), `network`, `enabled`, `use_in_chat`, `status` |
| `mcpc_tools` | tools reported by the server on connect (`description`, `input_schema`, `read_only`, `destructive`); replaced at every connect |
| `mcpc_prompts` | prompts the server offers (none of the three defaults do) |
| `mcpc_tool_perms` | the user's per-tool mode: `allow`, `ask`, `deny` |
| `mcpc_call_logs` | audit log: time, connector, tool, redacted arguments, decision, success, duration |

Notes:

- The prefix is `mcpc_` because an earlier experiment (branch `implement-mcp-connectors`) created `mcp_*` tables with different columns in some developers' databases. `CREATE TABLE IF NOT EXISTS` silently does nothing on a clashing table, which crashed startup once. Never reuse the old names.
- `init_tables()` is idempotent, adds columns when needed, and migrates the old `mcpc_grants` table into `mcpc_tool_perms`.
- Secrets: connector environment values go through `oprel/utils/secrets_store.py` (OS keyring via `keyring`, with a plain-text fallback if no keyring exists). The API never returns env values, only their names.

---

## 4. Permissions

Effective mode of a tool = stored mode, else the default (`store.default_mode`):

- Catalog connector and the server declares the tool read-only (`readOnlyHint`, not destructive) -> **allow**.
- Everything else -> **ask**.
- Custom connectors always ask: a server's hints about itself are not trusted.

`deny` hides the tool from the model (it is not in the prompt) and blocks it if called anyway. "Always allow" in the approval dialog stores `allow`. The UI can set a mode per tool or for a group ("Read-only tools", "Write/delete tools").

Custom commands (`POST /mcp/connectors/custom`) are refused unless the environment variable `OPREL_MCP_ALLOW_CUSTOM=1` is set, because a connector is a program running with the user's rights.

---

## 5. Life of a tool call

`tool_loop.execute_call()`:

1. **Find** the tool by exact connector and tool name (no fuzzy matching).
2. **Validate** the arguments against the tool's JSON schema (`policy.validate_arguments`). Invalid calls never reach the server; the model is told why and that nothing changed.
3. **Permission**: `deny` -> blocked; `ask` -> `approvals.request()` waits for the UI (`GET/POST /mcp/approvals`), 120 s timeout counts as deny; `allow` -> run.
4. **Call** via `manager.call_tool()`: 60 s timeout, never raises.
5. **Sanitise** the output for the model (`policy.sanitize_output`): redact secrets, truncate to 8,000 characters, flag instruction-like text, fence it in `<tool_result>` and neutralise markers that could fake a call.
6. **Log** to `mcpc_call_logs` with redacted arguments.
7. **Card**: build the UI-only `<toolcall>BASE64JSON</toolcall>` block (result kept to 4,000 characters).

Servers start with a **minimal environment** (`manager.minimal_env`): only variables needed to run, plus the connector's own env. The daemon's environment (API keys, tokens) is not inherited. There is **no OS sandbox yet** (no Windows Job Object or AppContainer); that is future work.

Installing pins the exact version: `npm view <pkg> version --registry=https://registry.npmjs.org`, then `npx --yes --registry=... pkg@VERSION`. Versions are never hard-coded in the catalog. (A private npm feed once reported `1.0.0` for everything, hence the explicit public registry.)

---

## 6. The tool protocol (why it is text, not native tool calling)

Tools are described to the model **as text in the system prompt**, and the model asks for one with:

```
<tool_call>{"connector": "Memory", "tool": "read_graph", "arguments": {}}</tool_call>
```

This works with every model and provider, but it is less reliable than each provider's native tool-calling API (Claude, OpenAI, Groq and Gemini pass tool schemas as structured data and the model is trained to use them). Much of `tool_loop.py` is resilience work for that gap:

- **Prompt** (`prompt_block`): each tool with a compact argument shape such as `{entities*: [{name*: string, entityType*: string, observations*: [string]}]}`, up to 500 characters of the tool's own description (it often says what to do next), plus a per-connector `usage_hint` from the catalog. Large tool lists get shorter descriptions to protect small context windows.
- **Parsing** (`extract_tool_calls`): tagged JSON; the whole reply being a bare JSON call (or in a code fence); the whole reply being `Connector.tool({...})` (small models), including `{names: ['x']}` style arguments.
- **Provider quirks**: gpt-oss on Groq emits its own tool format and the provider rejects it (`tool_use_failed`, `output_parse_failed`, "Tool choice is none"). Rejected calls are recovered from the error (`integration.recover_native_tool_call`), parse failures get a corrective retry, and errors become plain-language messages (`friendly_error`).
- **Retries**: unreadable or empty replies get up to 2 corrective prompts (`FORMAT_NUDGE`).
- **Use a tool first**: if the reply claims "saved/noted" or says "I don't know" and no tool ran, one nudge (`TOOL_FIRST_NUDGE`) asks it to use a tool or say it did not.
- **Follow-through**: a result containing "skipped / already exists / could not / failed" gets a note for the model (`INCOMPLETE_NOTE`); if it still says "done", one nudge (`RESULT_NUDGE`).
- **Always an answer**: after the tool budget (6 rounds, 3 calls each) or an empty reply, one forced call (`FINAL_NUDGE`) asks for the final text.
- **Small local models** (under about 3B parameters) usually fail at all of this; the empty-reply message then says so and recommends a 4B+ model.

**Recommended next step:** use native tool calling where available (OpenAI-compatible APIs, Gemini, OpenRouter, and llama.cpp with `--jinja` for local models), and keep this text protocol only as the fallback. That would remove most of the nudging above.

---

## 7. HTTP API (`routes.py`, prefix `/mcp`)

| Method and path | Purpose |
| --- | --- |
| `GET /catalog` | catalog entries + tool/prompt preview + `installed_id` |
| `GET /connectors` | installed connectors (counts, status) |
| `GET /connectors/{id}` | live tools with permission mode, prompts |
| `POST /connectors` | install a catalog connector `{catalog_id, paths?}` (pins the version, connects, rolls back on failure) |
| `POST /connectors/custom` | custom command (disabled unless `OPREL_MCP_ALLOW_CUSTOM=1`) |
| `PATCH /connectors/{id}` | `enabled`, `use_in_chat` |
| `PUT /connectors/{id}/permissions` | `{tool|group, mode}` |
| `DELETE /connectors/{id}` | disconnect and remove (secrets and permissions too) |
| `GET /approvals`, `POST /approvals/{id}` | pending approvals and the user's decision |
| `GET /logs` | audit log |

All of these sit behind the daemon's request guard (`oprel/server/security.py`: loopback host, no cross-site origins, bearer token for non-browser clients).

---

## 8. Web UI (`oprel/mcp/webui/`)

These files were moved out of `oprel/webui-react/` and are **not compiled** while parked. They use the `@/` import alias of the Next.js app.

| File | Destination when enabled | What it is |
| --- | --- | --- |
| `mcp.ts` | `webui-react/services/` | typed client for the API above |
| `ConnectorsView.tsx` | `webui-react/components/` | Settings page: Yours/Discover tabs, search, catalog detail (tools, prompts, Connect), connected detail (permissions grouped Read-only / Write-delete, mode per tool, Turn off, Disconnect) |
| `ApprovalPrompt.tsx` | `webui-react/components/` (mounted in `AppShell`) | approval card above the composer: connector, action, read-only/changes-data chip, internet chip, colored JSON details, countdown, Deny / Always allow / Allow once, Esc denies |
| `ToolCallCard.tsx` | `webui-react/components/` | `splitToolCalls`, `groupToolCalls`, `ToolCallGroup` (collapsed "Used Memory" / "3 tool calls", rows with Parameters and Result), JSON syntax colors |
| `chat_menu.snippet.tsx` | `ChatView.tsx` | the `+` menu "Connectors" flyout with per-connector switches (replaces the "Soon" item) |
| `chat_state.snippet.tsx` | `ChatView.tsx` | state and handlers for the flyout |
| `chat_render.snippet.tsx` | `ChatView.tsx` | renders a reply as text segments and tool-call groups (replaces the plain markdown block) |

While parked: Settings -> Connectors shows "coming soon", the chat menu has a disabled-looking "Connectors - Soon" item, and `ChatView` hides any `<toolcall>` blocks left in old chats.

---

## 9. Turning it back on

1. **Dependencies**: add to `pyproject.toml` an extra `mcp = ["mcp>=1.2.0", "jsonschema>=4.0.0"]` (or make them core), remove `exclude = ["oprel.mcp*"]` from `[tool.setuptools.packages.find]`, and add `"mcp/catalog_tools.json"` to `package-data`.
2. **Server startup** (`oprel/server/app.py`): import the router (`from oprel.mcp import routes as mcp_routes`; `app.include_router(mcp_routes.router)`) and in `lifespan`, after the idle-monitor task: `await manager.startup()` (wrapped in try/except so connectors never block the daemon) and `await manager.shutdown()` after `yield`.
3. **Local models** (`services/generation.py`): after the Supermemory block, `effective_system_prompt, tools_on = integration.local_system_prompt(effective_system_prompt)`; where tokens are produced, if `tools_on`, iterate `tool_loop.tool_aware_generate(client=model._client, prompt=full_prompt, gen_kwargs=...)` instead of the plain generate call (streaming and non-streaming).
4. **Cloud models** (`services/providers.py`): `prepared_messages, mcp_active = integration.with_mcp_block(_drop_empty_messages(prepared_messages))`; when `mcp_active`, build `generate_once` with `integration.call_provider_once(...)` over the existing `_call_*` functions and consume `tool_loop.tool_aware_generate_messages(...)`, emitting each piece as `data: {"choices":[{"delta":{"content": piece}}]}` and a final `data: [DONE]`. Also switch the non-streaming parsers (`_call_openai`, `_call_openrouter`, `_call_groq`) back to `integration.message_text` so provider-accepted native calls are converted.
5. **Web UI**: move the files in section 8 back, mount `<ApprovalPrompt />` in `AppShell`, replace the Settings "coming soon" panel with `<ConnectorsView embedded />`, and apply the three chat snippets.
6. **Tests**: move `oprel/mcp/tests` to `tests/` (or add it to `testpaths`) and add it to the CI workflow. They do not need Node.js: the stdio test starts a small Python MCP server.
7. **Startup cleanup**: `store.prune_unknown_catalog_connectors()` removes installed connectors whose catalog entry was retired; keep calling it from `manager.startup()`.

---

## 10. Lessons learned (read before changing things)

- **Never decode "JSON-looking" chat messages.** `db.get_conversation_messages` used to `json.loads` any content starting with `{` or `[`. A reply that was a bare JSON tool call became an object, was sent to Groq as the message content, and every later message in that chat failed with `400: messages.N.content must be a string or array`. Only a list of typed parts (multimodal) is decoded now. This fix is in the main code, not parked.
- **Drop empty or malformed turns before calling a provider** (`_drop_empty_messages`): failed replies leave blank assistant messages that providers reject. Also in main code.
- **Do not block the event loop.** Loading a model inside an `async` handler froze the whole server (and the approval polling). Loads run in a worker thread and embedding loads are serialised. Main code.
- **Tests must never touch the real keyring.** A test once overwrote and then deleted the developer's real Hugging Face token. `tests/conftest.py` and `oprel/mcp/tests/conftest.py` give every test an in-memory keyring. Keep it autouse.
- **Do not trust what a model says it did.** Weak models answer "saved" without calling anything, or after a result that says "skipped". That is why there are use-a-tool-first and follow-through nudges, and why cards show the real result.
- **Tool descriptions matter.** They were once cut to 200 characters and lost "use add_observations for existing entities"; now up to 500 for small tool sets.
- **Pin versions, resolve them at install time.** The catalog never hard-codes a version.
- **A connector that fails to start is removed** at install so the user can retry cleanly.
- **Cards are UI-only.** They are stored with the reply but stripped before history goes to a model or to memory.
- **The Excel connector (community) cannot create workbooks, only edit existing ones.** The Office/Figma/Canva/Google/n8n connectors were tried and removed from the catalog for now; Word, PowerPoint, Excel, n8n, Figma and Google Workspace servers all start over stdio. Canva needed `mcp-remote` and an OAuth browser flow, which is not implemented.

---

## 11. Known gaps and ideas

- Native tool calling (section 6) is the biggest reliability win.
- No OS-level sandboxing or resource limits beyond timeouts and output caps (Windows Job Object / AppContainer, or containers).
- No HTTP/SSE transports and no OAuth flow (needed for remote connectors such as Canva or Figma's hosted server).
- No "offline mode" switch that blocks connector and provider traffic.
- Connector packages are pinned by version but not by hash/integrity.
- Tool results can exceed a small model's context; consider summarising or paging.
- Only the local daemon can approve actions; a headless API client cannot answer approvals (they time out as deny).
