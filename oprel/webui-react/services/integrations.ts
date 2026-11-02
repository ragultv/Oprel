/**
 * Integrations & tool-calling client.
 *
 * Security: every call here goes through `authFetch`, which attaches the daemon's
 * per-process token (X-Oprel-Token). The token is fetched from /auth/session and
 * kept in memory only — never in localStorage. Secrets (PATs) are sent once on
 * connect and are never returned by the server.
 */
import { API_BASE } from './api'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface AuthField {
  key: string
  label: string
  type: 'text' | 'url' | 'secret'
  required: boolean
  placeholder: string
  help: string
  help_url: string
}

export interface ToolInfo {
  name: string
  title: string
  description: string
  kind: 'read' | 'write'
}

export interface ToolPolicy extends ToolInfo {
  enabled: boolean
  approval_mode: 'manual' | 'auto'
}

export interface CatalogEntry {
  id: string
  name: string
  description: string
  icon: string
  docs_url: string
  auth_fields: AuthField[]
  tools: ToolInfo[]
  connection_id: string | null
  secure_storage: string
  secure_storage_available: boolean
}

export interface Connection {
  id: string
  integration_type: string
  integration_name: string
  display_name: string
  config: Record<string, any>
  has_secret: boolean
  secret_hint: string
  status: 'connected' | 'error' | 'expired' | 'unverified'
  status_message: string
  enabled: boolean
  last_verified_at: string | null
  tools: ToolPolicy[]
  details?: Record<string, any>
}

/** Events streamed by the server-side tool loop (see oprel/server/tools/loop.py). */
export type ToolEvent =
  | { type: 'tool_call'; call_id: string; tool: string; title: string; kind: 'read' | 'write'; integration: string; arguments: Record<string, any>; approval: 'auto' | 'manual'; invalid?: boolean }
  | { type: 'approval_required'; approval_id: string; call_id: string; tool: string; title: string; kind: 'read' | 'write'; integration: string; arguments: Record<string, any>; expires_in: number; reason: 'policy' | 'write_limit' }
  | { type: 'tool_result'; call_id: string; tool: string; ok: boolean; data: Record<string, any>; duration_ms: number; rejected?: boolean }
  | { type: 'error'; message: string }
  | { type: 'done'; finish_reason: 'stop' | 'approval_required' | 'max_iterations' | 'error'; conversation_id: string | null }

// ─── Token handling ───────────────────────────────────────────────────────────

let tokenPromise: Promise<string> | null = null

async function getToken(forceRefresh = false): Promise<string> {
  if (forceRefresh) tokenPromise = null
  if (!tokenPromise) {
    tokenPromise = fetch(`${API_BASE}/auth/session`, { cache: 'no-store' })
      .then(async res => {
        if (!res.ok) throw new Error('Could not authenticate with the Oprel daemon')
        const data = await res.json()
        return data.token as string
      })
      .catch(err => {
        tokenPromise = null
        throw err
      })
  }
  return tokenPromise
}

/** fetch() with the local API token. Retries once with a fresh token (daemon restarted). */
export async function authFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const send = async (token: string) =>
    fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { ...(init.headers || {}), 'X-Oprel-Token': token },
    })
  let res = await send(await getToken())
  if (res.status === 401) res = await send(await getToken(true))
  return res
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    const detail = typeof err.detail === 'string'
      ? err.detail
      : Array.isArray(err.detail) ? err.detail.map((d: any) => d.msg).join(', ') : `Request failed (${res.status})`
    throw new Error(detail)
  }
  return res.json()
}

const JSON_HEADERS = { 'Content-Type': 'application/json' }

// ─── Integrations API ─────────────────────────────────────────────────────────

export const IntegrationsAPI = {
  catalog: async (): Promise<CatalogEntry[]> => json(await authFetch('/integrations/catalog')),

  list: async (): Promise<Connection[]> => json(await authFetch('/integrations')),

  connect: async (integrationType: string, values: Record<string, string>): Promise<Connection> =>
    json(await authFetch(`/integrations/${encodeURIComponent(integrationType)}/connect`, {
      method: 'POST',
      headers: JSON_HEADERS,
      body: JSON.stringify({ values }),
    })),

  test: async (connectionId: string): Promise<{ ok: boolean; message: string; connection: Connection }> =>
    json(await authFetch(`/integrations/${encodeURIComponent(connectionId)}/test`, { method: 'POST' })),

  setEnabled: async (connectionId: string, enabled: boolean): Promise<Connection> =>
    json(await authFetch(`/integrations/${encodeURIComponent(connectionId)}`, {
      method: 'PATCH',
      headers: JSON_HEADERS,
      body: JSON.stringify({ enabled }),
    })),

  updateTools: async (
    connectionId: string,
    tools: { name: string; enabled?: boolean; approval_mode?: 'manual' | 'auto' }[],
  ): Promise<Connection> =>
    json(await authFetch(`/integrations/${encodeURIComponent(connectionId)}/tools`, {
      method: 'PUT',
      headers: JSON_HEADERS,
      body: JSON.stringify({ tools }),
    })),

  disconnect: async (connectionId: string): Promise<void> => {
    await json(await authFetch(`/integrations/${encodeURIComponent(connectionId)}`, { method: 'DELETE' }))
  },
}

// ─── Tool-enabled chat streaming ──────────────────────────────────────────────

export interface StreamHandlers {
  onToken: (token: string) => void
  onEvent: (event: ToolEvent) => void
  onConversationId?: (id: string) => void
}

async function consumeStream(res: Response, handlers: StreamHandlers, signal?: AbortSignal) {
  if (!res.ok) {
    await json(res) // throws with the server's message
  }
  const convId = res.headers.get('X-Conversation-ID')
  if (convId) handlers.onConversationId?.(convId)
  if (!res.body) return

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const handleLine = (line: string) => {
    const trimmed = line.trim()
    if (!trimmed.startsWith('data: ') || trimmed === 'data: [DONE]') return
    let chunk: any
    try {
      chunk = JSON.parse(trimmed.slice(6))
    } catch {
      return
    }
    if (chunk.oprel_event) {
      handlers.onEvent(chunk.oprel_event as ToolEvent)
      return
    }
    const content = chunk.choices?.[0]?.delta?.content
    if (content) handlers.onToken(content)
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    if (signal?.aborted) {
      await reader.cancel()
      return
    }
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''
    lines.forEach(handleLine)
  }
  if (buffer.trim()) handleLine(buffer)
}

export async function toolChatStream(
  payload: Record<string, any>,
  enableTools: string[],
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await authFetch('/v1/chat/completions', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ ...payload, stream: true, enable_tools: enableTools }),
    signal,
  })
  await consumeStream(res, handlers, signal)
}

export async function decideApproval(
  approvalId: string,
  decision: 'approve' | 'reject',
  handlers: StreamHandlers,
  opts: { arguments?: Record<string, any>; reason?: string } = {},
  signal?: AbortSignal,
): Promise<void> {
  const res = await authFetch(`/tool-approvals/${encodeURIComponent(approvalId)}`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ decision, stream: true, ...opts }),
    signal,
  })
  await consumeStream(res, handlers, signal)
}
