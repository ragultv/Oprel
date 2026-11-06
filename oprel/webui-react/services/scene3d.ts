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

  clear: async (conversationId: string): Promise<SceneDocument> => {
    const res = await authFetch(`/scene3d/${enc(conversationId)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`Could not clear the 3D scene (${res.status})`)
    return res.json()
  },
}
