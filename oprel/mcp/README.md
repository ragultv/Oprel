# oprel/mcp: MCP connectors (parked)

Everything for the MCP connectors feature lives in this folder. **Nothing outside it imports it**, and the feature is not shipped in the wheel yet. The web UI shows "Connectors: coming soon".

- How it works, how to turn it back on, and what we learned: [`docs/internal/mcp_connectors.md`](../../docs/internal/mcp_connectors.md)
- Run its tests: `pytest oprel/mcp/tests`
- React code and chat snippets: [`webui/`](webui/)
- Glue for the generation and provider code: [`integration.py`](integration.py)
