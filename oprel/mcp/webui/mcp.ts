const API_BASE =
  typeof window !== "undefined" && window.location.port === "3000" ? "http://127.0.0.1:11435" : ""

export type PermissionMode = "allow" | "ask" | "deny"

export interface PreviewTool {
  name: string
  description: string
  read_only: boolean
  destructive: boolean
}

export interface Prompt {
  name: string
  description: string
}

export interface CatalogEntry {
  id: string
  name: string
  description: string
  package: string
  needs_paths?: boolean
  network: "none" | "internet"
  risk: "read" | "write"
  installed_id: string | null
  tools: PreviewTool[]
  prompts: Prompt[]
  checked_version: string | null
}

export interface Connector {
  id: string
  name: string
  description: string
  catalog_id: string | null
  package: string | null
  version: string | null
  network: string
  enabled: boolean
  use_in_chat: boolean
  status: string
  tool_count: number
  tools: string[]
}

export interface ConnectorTool {
  name: string
  description: string
  read_only: boolean
  mode: PermissionMode
  customized: boolean
}

export interface ConnectorDetail extends Omit<Connector, "tools"> {
  risk: string | null
  tools: ConnectorTool[]
  prompts: Prompt[]
}

export interface PendingApproval {
  id: string
  connector_id: string
  connector_name: string
  network: string
  tool: string
  read_only: boolean
  arguments: string
  created_at: number
  expires_in: number
}

export interface CallLog {
  id: string
  ts: string
  connector_id: string
  tool: string
  args: string
  decision: string
  ok: number | null
  error: string
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}/mcp${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    let detail = "Something went wrong."
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* keep default */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const MCP = {
  catalog: () => request<CatalogEntry[]>("/catalog"),
  connectors: () => request<Connector[]>("/connectors"),
  detail: (id: string) => request<ConnectorDetail>(`/connectors/${id}`),
  install: (catalog_id: string, paths: string[]) =>
    request<Connector>("/connectors", { method: "POST", body: JSON.stringify({ catalog_id, paths }) }),
  patch: (id: string, body: { enabled?: boolean; use_in_chat?: boolean }) =>
    request<Connector>(`/connectors/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  remove: (id: string) => request<{ success: boolean }>(`/connectors/${id}`, { method: "DELETE" }),
  setPermission: (id: string, target: { tool: string } | { group: "read" | "write" }, mode: PermissionMode) =>
    request<{ tools: ConnectorTool[] }>(`/connectors/${id}/permissions`, {
      method: "PUT",
      body: JSON.stringify({ ...target, mode }),
    }),
  approvals: () => request<PendingApproval[]>("/approvals"),
  decide: (id: string, decision: "once" | "always" | "deny") =>
    request<{ success: boolean }>(`/approvals/${id}`, { method: "POST", body: JSON.stringify({ decision }) }),
  logs: (limit = 50) => request<CallLog[]>(`/logs?limit=${limit}`),
}
