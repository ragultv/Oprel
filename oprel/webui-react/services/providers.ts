/**
 * External AI Provider service for OPREL.
 *
 * Storage strategy:
 *   - Provider configs (id, name, type, api_key, base_url, enabled, model lists)
 *     are persisted in the BACKEND SQLite DB via /providers REST API.
 *   - The model list cache (available_model_ids) is also stored in the DB so it
 *     survives page refreshes without needing to re-fetch on every load.
 *   - We do NOT store provider model lists in the local models cache —
 *     those are synthetic AIModel objects derived at runtime.
 *
 * What IS stored in DB: conversations, inference_log, api_config (provider rows)
 * What is NOT stored in DB: downloaded local model files (managed by the FS)
 */

import type { AIModel } from './data'

// ─── Types ───────────────────────────────────────────────────────────────────

/** All supported provider type slugs */
export type ProviderType =
  | 'openai'
  | 'gemini'
  | 'openai-compatible'
  | 'nvidia'
  | 'groq'
  | 'openrouter'

export interface ProviderConfig {
  /** Stable slug, e.g. "openai", "my-groq", "local-llm" */
  id: string
  /** Human readable label */
  name: string
  type: ProviderType
  apiKey: string
  /** For openai-compatible / custom endpoints */
  baseUrl?: string
  /** Only the model IDs the user has turned ON */
  enabledModelIds: string[]
  /** Full list fetched from the provider API */
  availableModelIds: string[]
  enabled: boolean
  /** ISO timestamp of last successful model list fetch */
  lastFetched?: string
}

// ─── Well-known provider presets ─────────────────────────────────────────────

export interface ProviderPreset {
  type: ProviderType
  name: string
  /** Effective base URL used for /models and /chat/completions */
  baseUrl: string
  docsUrl: string
  /** Tailwind-compatible hex accent colour */
  color: string
  /** Short description shown in the UI */
  description: string
}

export const PROVIDER_PRESETS: Record<ProviderType, ProviderPreset> = {
  openai: {
    type: 'openai',
    name: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    docsUrl: 'https://platform.openai.com/api-keys',
    color: '#10a37f',
    description: 'GPT-4o, o1, o3-mini and more',
  },
  gemini: {
    type: 'gemini',
    name: 'Google Gemini',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
    docsUrl: 'https://aistudio.google.com/app/apikey',
    color: '#4285F4',
    description: 'Gemini 2.0 Flash, Pro and Ultra',
  },
  nvidia: {
    type: 'nvidia',
    name: 'NVIDIA NIM',
    baseUrl: 'https://integrate.api.nvidia.com/v1',
    docsUrl: 'https://developer.nvidia.com/nim',
    color: '#76b900',
    description: 'LLMs accelerated on NVIDIA infrastructure',
  },
  groq: {
    type: 'groq',
    name: 'Groq',
    baseUrl: 'https://api.groq.com/openai/v1',
    docsUrl: 'https://console.groq.com/keys',
    color: '#f55036',
    description: 'Ultra-fast LPU-powered inference',
  },
  openrouter: {
    type: 'openrouter',
    name: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    docsUrl: 'https://openrouter.ai/keys',
    color: '#7c3aed',
    description: '200+ models from many providers',
  },
  'openai-compatible': {
    type: 'openai-compatible',
    name: 'Custom (OpenAI-compatible)',
    baseUrl: '',
    docsUrl: '',
    color: '#f97316',
    description: 'Any server that speaks the OpenAI API',
  },
}

// ─── API_BASE (mirrors api.ts) ────────────────────────────────────────────────

const API_BASE =
  typeof window !== 'undefined' && window.location.port === '3000'
    ? 'http://localhost:11435'
    : ''

// ─── DB-backed CRUD (via backend REST) ───────────────────────────────────────

/** Map snake_case DB row → camelCase ProviderConfig */
function fromDbRow(row: any): ProviderConfig {
  return {
    id: row.id,
    name: row.name,
    type: row.type as ProviderType,
    apiKey: row.api_key ?? '',
    baseUrl: row.base_url ?? '',
    enabled: Boolean(row.enabled),
    enabledModelIds: Array.isArray(row.enabled_model_ids) ? row.enabled_model_ids : [],
    availableModelIds: Array.isArray(row.available_model_ids) ? row.available_model_ids : [],
    lastFetched: row.last_fetched ?? undefined,
  }
}

/** Map camelCase ProviderConfig → DB row shape */
function toDbRow(p: ProviderConfig) {
  return {
    id: p.id,
    name: p.name,
    type: p.type,
    api_key: p.apiKey,
    base_url: p.baseUrl ?? '',
    enabled: p.enabled,
    enabled_model_ids: p.enabledModelIds,
    available_model_ids: p.availableModelIds,
    last_fetched: p.lastFetched ?? null,
  }
}

export async function loadAllProviders(): Promise<ProviderConfig[]> {
  try {
    const res = await fetch(`${API_BASE}/providers`)
    if (!res.ok) return []
    const rows: any[] = await res.json()
    return rows.map(fromDbRow)
  } catch {
    return []
  }
}

export async function saveProvider(p: ProviderConfig): Promise<ProviderConfig> {
  const res = await fetch(`${API_BASE}/providers/${encodeURIComponent(p.id)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toDbRow(p)),
  })
  if (!res.ok) throw new Error(`Failed to save provider: HTTP ${res.status}`)
  return fromDbRow(await res.json())
}

export async function deleteProvider(id: string): Promise<void> {
  await fetch(`${API_BASE}/providers/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

// ─── Fetch model list from external provider ──────────────────────────────────

export async function fetchProviderModels(provider: ProviderConfig): Promise<string[]> {
  const res = await fetch(`${API_BASE}/providers/${encodeURIComponent(provider.id)}/models`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(err?.detail || `${provider.name}: Failed to fetch models`)
  }
  return res.json()
}

// ─── Stream a chat completion through an external provider ────────────────────

export async function providerChatStream(
  provider: ProviderConfig,
  modelId: string,
  messages: any[],
  options: { max_tokens?: number; temperature?: number; top_p?: number; conversation_id?: string; rag?: boolean; skill?: { id: string; name: string } },
  onToken: (t: string) => void,
  onConversationId?: (id: string) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE}/providers/${encodeURIComponent(provider.id)}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: modelId,
      messages,
      stream: true,
      max_tokens: options.max_tokens,
      temperature: options.temperature,
      top_p: options.top_p,
      conversation_id: options.conversation_id,
      rag: options.rag,
      skill: options.skill,
    }),
    signal,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(err?.detail || `${provider.name} request failed (${res.status})`)
  }

  // Extract created conversation ID if present (for new chats)
  const convId = res.headers.get('X-Conversation-ID')
  if (convId && onConversationId) {
    onConversationId(convId)
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done || signal?.aborted) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed === 'data: [DONE]') continue
      if (trimmed.startsWith('data: ')) {
        try {
          const json = JSON.parse(trimmed.slice(6))
          
          if (json.error) {
            throw new Error(typeof json.error === 'string' ? json.error : JSON.stringify(json.error))
          }

          let token = ''
          if (provider.type === 'gemini') {
            token = json?.candidates?.[0]?.content?.parts?.[0]?.text || ''
          } else {
            token = json?.choices?.[0]?.delta?.content || ''
          }
          if (token) onToken(token)
        } catch (e) {
          if (e instanceof Error && e.message.includes('API Error')) {
            throw e // Propagate API errors
          }
          // ignore other SSE parse errors
        }
      }
    }
  }
}

// ─── Convert provider configs → AIModel entries for the model selector ────────

export function providerModelsToAIModels(providers: ProviderConfig[]): AIModel[] {
  const result: AIModel[] = []
  for (const p of providers) {
    if (!p.enabled) continue
    const enabledModelIds = Array.isArray(p.enabledModelIds) ? p.enabledModelIds : []
    for (const modelId of enabledModelIds) {
      const preset = PROVIDER_PRESETS[p.type] || PROVIDER_PRESETS['openai-compatible']
      result.push({
        /** Composite ID: "{providerId}::{modelId}" */
        id: `${p.id}::${modelId}`,
        name: formatModelName(modelId),
        family: p.name,
        size: '—',
        quantization: 'API',
        contextLength: 0,
        ramRequired: 0,
        status: 'available' as const,
        tags: [p.name, p.type, 'External'],
        description: `${p.name} cloud model`,
        publisher: p.name,
        architecture: p.type,
        parameters: '—',
        license: 'Proprietary',
        compatibility: 'compatible' as const,
        downloaded: true,
        modelRepoId: `${p.id}::${modelId}`,
        // Set category to 'external' for grouping in ModelsView
        category: 'external',
      })
    }
  }
  return result
}

/** Extract a display name from a raw model ID like "gpt-4o-mini" or "meta/llama-3.1-70b-instruct" */
function formatModelName(modelId: string): string {
  const parts = modelId.split('/')
  const base = parts[parts.length - 1]
  return base
    .replace(/-/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
}

/** Check if an AIModel ID belongs to an external provider */
export function isProviderModel(modelId: string): boolean {
  // Provider models use "{providerId}::{rawModelId}" format,
  // while local models use "{hf_repo_id}::QUANT" (quant is uppercase)
  // We can distinguish by checking if the part after :: is NOT a known quant pattern
  const QUANT_RE = /^(Q[2-9]|IQ|F16|F32|BF16)/i
  const parts = modelId.split('::')
  if (parts.length < 2) return false
  return !QUANT_RE.test(parts[1])
}

/** Extract {providerId, rawModelId} from a composite provider model ID */
export function parseProviderModelId(compositeId: string): { providerId: string; rawModelId: string } | null {
  if (!isProviderModel(compositeId)) return null
  const idx = compositeId.indexOf('::')
  return {
    providerId: compositeId.slice(0, idx),
    rawModelId: compositeId.slice(idx + 2),
  }
}
