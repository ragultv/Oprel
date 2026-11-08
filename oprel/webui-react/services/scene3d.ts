/**
 * 3D scene client (see oprel/integrations/scene3d and oprel/server/routes/scene3d.py).
 * The model edits the scene through scene_* tools; the UI only reads the compiled mesh.
 */
import { authFetch } from './integrations'

/** Built-in integration id to put in `enable_tools`. */
export const SCENE3D_TOOLSET = 'scene3d'

export interface SceneObjectSpec {
  id: string
  shape: string
  params?: Record<string, any>
  parent?: string
  label?: string
  color?: string
  /** On a group: its parts are fused into one solid. */
  merge?: boolean
}

export interface MergeResult extends SceneDocument {
  notes: string[]
  warnings: string[]
  /** Merged group id -> how many separate bodies it came out as (1 = one solid part). */
  merged: Record<string, { bodies: number; parts: string[] }>
}

export interface StlDownload {
  data: ArrayBuffer
  version: number
  /** Separate pieces in the fused scene: 1 means one solid part. */
  bodies: number
  watertight: boolean
}

export interface SceneDocument {
  conversation_id: string
  version: number
  objects: SceneObjectSpec[]
  updated_at: string | null
}

/** True for a tool event that changed the scene; carries the new version. */
export function sceneVersionFromEvent(event: { type: string; tool?: string; ok?: boolean; data?: any }): number | null {
  if (event.type !== 'tool_result' || !event.ok || !event.tool?.startsWith('scene_')) return null
  const v = event.data?.scene_version
  return typeof v === 'number' ? v : null
}

const enc = encodeURIComponent

export const Scene3DAPI = {
  get: async (conversationId: string): Promise<SceneDocument> => {
    const res = await authFetch(`/scene3d/${enc(conversationId)}`)
    if (!res.ok) throw new Error(`Could not load the 3D scene (${res.status})`)
    return res.json()
  },

  /** Compiled scene as GLB, or null when the scene has nothing to draw. */
  glb: async (conversationId: string): Promise<{ data: ArrayBuffer | null; version: number }> => {
    const res = await authFetch(`/scene3d/${enc(conversationId)}/scene.glb`, { cache: 'no-store' })
    const version = Number(res.headers.get('X-Scene-Version') || 0)
    if (res.status === 204) return { data: null, version }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(typeof err.detail === 'string' ? err.detail : `Could not build the 3D scene (${res.status})`)
    }
    return { data: await res.arrayBuffer(), version }
  },

  /** Fuse a group (or, with no group, the whole scene) into one solid; merge=false splits it again. */
  merge: async (conversationId: string, merge: boolean, group?: string): Promise<MergeResult> => {
    const res = await authFetch(`/scene3d/${enc(conversationId)}/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ merge, group: group ?? null }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(typeof err.detail === 'string' ? err.detail : `Could not merge (${res.status})`)
    }
    return res.json()
  },

  /** The whole scene fused into one solid, as STL (metres), or null when nothing is drawn. */
  stl: async (conversationId: string): Promise<StlDownload | null> => {
    const res = await authFetch(`/scene3d/${enc(conversationId)}/scene.stl`, { cache: 'no-store' })
    if (res.status === 204) return null
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(typeof err.detail === 'string' ? err.detail : `Could not export STL (${res.status})`)
    }
    return {
      data: await res.arrayBuffer(),
      version: Number(res.headers.get('X-Scene-Version') || 0),
      bodies: Number(res.headers.get('X-Solid-Bodies') || 0),
      watertight: res.headers.get('X-Solid-Watertight') === 'true',
    }
  },

  clear: async (conversationId: string): Promise<SceneDocument> => {
    const res = await authFetch(`/scene3d/${enc(conversationId)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`Could not clear the 3D scene (${res.status})`)
    return res.json()
  },
}

