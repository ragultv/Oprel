'use client'

/**
 * 3D scene panel. The model draws through the scene_* tools; the server compiles the recipe
 * into a GLB (one node per object id, `id#n` for pattern copies) and this panel renders it.
 * Load with next/dynamic (ssr: false): three.js is client-only and large.
 */
import React, { Suspense, useCallback, useEffect, useMemo, useState } from 'react'
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber'
import { Bounds, GizmoHelper, GizmoViewport, Grid, OrbitControls, useBounds } from '@react-three/drei'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { Box, Download, Loader2, Maximize, Trash2, X, Grid3x3 } from 'lucide-react'
import { Scene3DAPI, type SceneObjectSpec } from '@/services/scene3d'
import { cn } from '@/services/utils'

interface ScenePanelProps {
  conversationId: string | null
  /** Bumped by ChatView whenever a scene_* tool changes the scene. */
  version: number
  onVersionChange?: (version: number) => void
  onClose: () => void
}

const objectId = (nodeName: string) => nodeName.split('#')[0]

function disposeTree(root: THREE.Object3D | null) {
  root?.traverse(obj => {
    const mesh = obj as THREE.Mesh
    if (!mesh.isMesh) return
    mesh.geometry?.dispose()
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    mats.forEach(m => m?.dispose())
  })
}

function parseGlb(data: ArrayBuffer): Promise<THREE.Group> {
  return new Promise((resolve, reject) => {
    new GLTFLoader().parse(data, '', gltf => {
      // Own material per mesh, so highlighting one object never lights up another with the same colour.
      gltf.scene.traverse(obj => {
        const mesh = obj as THREE.Mesh
        if (mesh.isMesh && !Array.isArray(mesh.material)) mesh.material = mesh.material.clone()
      })
      resolve(gltf.scene)
    }, reject)
  })
}

type View = 'iso' | 'top' | 'front' | 'side' | 'bottom'
const VIEWS: { id: View; label: string; dir: [number, number, number] }[] = [
  { id: 'iso', label: '3D', dir: [1, 0.8, 1.2] },
  { id: 'top', label: 'Top', dir: [0, 1, 0.001] },      // tiny z keeps OrbitControls stable at the pole
  { id: 'front', label: 'Front', dir: [0, 0.15, 1] },
  { id: 'side', label: 'Side', dir: [1, 0.15, 0] },
  { id: 'bottom', label: 'Bottom', dir: [0, -1, 0.001] },
]

/** Jump the camera to a preset view around the drawing. */
function CameraRig({ root, request }: { root: THREE.Group | null; request: { view: View; n: number } | null }) {
  const { camera, controls } = useThree() as unknown as {
    camera: THREE.PerspectiveCamera
    controls: { target: THREE.Vector3; update: () => void } | null
  }
  useEffect(() => {
    if (!request || !root) return
    const box = new THREE.Box3().setFromObject(root)
    const centre = box.getCenter(new THREE.Vector3())
    const radius = box.getSize(new THREE.Vector3()).length() / 2 || 1
    const dist = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.1
    const dir = new THREE.Vector3(...VIEWS.find(v => v.id === request.view)!.dir).normalize()
    camera.position.copy(centre).addScaledVector(dir, dist)
    camera.up.set(0, 1, 0)
    camera.lookAt(centre)
    if (controls) {
      controls.target.copy(centre)
      controls.update()
    }
    // Only on a button press: a reloaded drawing is re-framed by <FitOnChange> instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request])
  return null
}

/** Re-frame the camera whenever the drawing changes. */
function FitOnChange({ token }: { token: unknown }) {
  const bounds = useBounds()
  useEffect(() => {
    bounds.refresh().clip().fit()
  }, [token, bounds])
  return null
}

function SceneContent({
  root, selected, wireframe, fitToken, onSelect,
}: {
  root: THREE.Group
  selected: string | null
  wireframe: boolean
  fitToken: unknown
  onSelect: (id: string | null) => void
}) {
  useEffect(() => {
    root.traverse(obj => {
      const mesh = obj as THREE.Mesh
      if (!mesh.isMesh) return
      const mat = mesh.material as THREE.MeshStandardMaterial
      mat.wireframe = wireframe
      if (mat.emissive) mat.emissive.set(selected && objectId(mesh.name) === selected ? '#3b2a14' : '#000000')
    })
  }, [root, selected, wireframe])

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation()
    onSelect(objectId(e.object.name))
  }

  return (
    <Bounds margin={1.25} maxDuration={0.4}>
      <FitOnChange token={fitToken} />
      <primitive object={root} onClick={handleClick} />
    </Bounds>
  )
}

/** Grid cell size that suits the drawing: 1 cm for a pen, 1 m for a house. */
function gridScale(root: THREE.Group | null): number {
  if (!root) return 1
  const size = new THREE.Box3().setFromObject(root).getSize(new THREE.Vector3())
  const largest = Math.max(size.x, size.z, 0.01)
  return Math.pow(10, Math.floor(Math.log10(largest)))
}

function formatParams(params?: Record<string, any>): string {
  if (!params) return ''
  return Object.entries(params)
    .filter(([k]) => k !== 'segments')
    .map(([k, v]) => `${k.replace(/_/g, ' ')} ${typeof v === 'number' ? +v.toFixed(4) : JSON.stringify(v)}`)
    .join(' · ')
}

function ToolbarButton({ title, onClick, active, disabled, children }: {
  title: string; onClick: () => void; active?: boolean; disabled?: boolean; children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      disabled={disabled}
      className={cn(
        'w-7 h-7 flex items-center justify-center rounded-md transition-all disabled:opacity-40',
        active ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:bg-secondary hover:text-foreground',
      )}
    >
      {children}
    </button>
  )
}

export default function ScenePanel({ conversationId, version, onVersionChange, onClose }: ScenePanelProps) {
  const [root, setRoot] = useState<THREE.Group | null>(null)
  const [glb, setGlb] = useState<ArrayBuffer | null>(null)
  const [objects, setObjects] = useState<SceneObjectSpec[]>([])
  const [loadedVersion, setLoadedVersion] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [wireframe, setWireframe] = useState(false)
  const [fitToken, setFitToken] = useState(0)
  const [viewRequest, setViewRequest] = useState<{ view: View; n: number } | null>(null)

  // Load the compiled mesh and the recipe whenever the version moves.
  useEffect(() => {
    if (!conversationId) return
    let cancelled = false
    setLoading(true)
    setError(null)
    Promise.all([Scene3DAPI.glb(conversationId), Scene3DAPI.get(conversationId)])
      .then(async ([mesh, doc]) => {
        const next = mesh.data ? await parseGlb(mesh.data) : null
        if (cancelled) { disposeTree(next); return }
        setRoot(next)  // the previous tree is disposed by the cleanup effect below
        setGlb(mesh.data)
        setObjects(doc.objects)
        setLoadedVersion(doc.version)
        setFitToken(t => t + 1)
        setViewRequest(null)
        setSelected(s => (s && doc.objects.some(o => o.id === s) ? s : null))
      })
      .catch(err => { if (!cancelled) setError(err.message || 'Could not load the scene') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [conversationId, version])

  useEffect(() => () => disposeTree(root), [root])

  const selectedSpec = useMemo(() => objects.find(o => o.id === selected) || null, [objects, selected])
  const cell = useMemo(() => gridScale(root), [root])

  const download = useCallback(() => {
    if (!glb) return
    const url = URL.createObjectURL(new Blob([glb], { type: 'model/gltf-binary' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `scene-v${loadedVersion}.glb`
    a.click()
    URL.revokeObjectURL(url)
  }, [glb, loadedVersion])

  const clear = useCallback(async () => {
    if (!conversationId || !window.confirm('Clear the whole 3D scene?')) return
    try {
      const doc = await Scene3DAPI.clear(conversationId)
      onVersionChange?.(doc.version)
    } catch (err: any) {
      setError(err.message)
    }
  }, [conversationId, onVersionChange])

  return (
    <div className="flex flex-col h-full bg-[#141414] overflow-hidden">
      {/* Top bar */}
      <div className="h-12 flex items-center justify-between px-4 border-b border-border/40 shrink-0 gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <Box size={15} className="text-primary shrink-0" />
          <span className="text-sm font-semibold text-foreground">3D Scene</span>
          <span className="text-[10px] text-muted-foreground truncate">
            {objects.length} object{objects.length === 1 ? '' : 's'}{loadedVersion ? ` · v${loadedVersion}` : ''}
          </span>
          {loading && <Loader2 size={12} className="animate-spin text-muted-foreground" />}
        </div>
        <div className="flex items-center gap-0.5">
          <ToolbarButton title="Fit to view" onClick={() => setFitToken(t => t + 1)} disabled={!root}><Maximize size={14} /></ToolbarButton>
          <ToolbarButton title="Wireframe" onClick={() => setWireframe(w => !w)} active={wireframe} disabled={!root}><Grid3x3 size={14} /></ToolbarButton>
          <ToolbarButton title="Download GLB" onClick={download} disabled={!glb}><Download size={14} /></ToolbarButton>
          <ToolbarButton title="Clear scene" onClick={clear} disabled={!objects.length}><Trash2 size={14} /></ToolbarButton>
          <div className="h-4 w-px bg-border mx-1" />
          <ToolbarButton title="Close" onClick={onClose}><X size={14} /></ToolbarButton>
        </div>
      </div>

      {/* Viewport */}
      <div className="relative flex-1 min-h-0">
        <Canvas
          camera={{ position: [4, 3, 6], fov: 45, near: 0.001, far: 5000 }}
          gl={{ antialias: true, preserveDrawingBuffer: true }}
          onPointerMissed={() => setSelected(null)}
        >
          <color attach="background" args={['#141414']} />
          <hemisphereLight args={['#ffffff', '#6a6a6a', 0.9]} />
          <directionalLight position={[5, 10, 7]} intensity={1.6} />
          <directionalLight position={[-6, 4, -5]} intensity={0.5} />
          {/* From below too, so the underside isn't black when orbiting under the ground. */}
          <directionalLight position={[2, -8, 3]} intensity={0.6} />
          <Grid
            side={THREE.DoubleSide}
            infiniteGrid
            cellSize={cell / 10}
            sectionSize={cell}
            cellColor="#2b2b2b"
            sectionColor="#454545"
            fadeDistance={cell * 60}
            fadeStrength={1.5}
          />
          <Suspense fallback={null}>
            {root && (
              <SceneContent root={root} selected={selected} wireframe={wireframe} fitToken={fitToken} onSelect={setSelected} />
            )}
          </Suspense>
          <OrbitControls makeDefault enableDamping minPolarAngle={0} maxPolarAngle={Math.PI} />
          <CameraRig root={root} request={viewRequest} />
          <GizmoHelper alignment="bottom-right" margin={[64, 64]}>
            <GizmoViewport axisColors={['#e5484d', '#46a758', '#3e63dd']} labelColor="#fff" />
          </GizmoHelper>
        </Canvas>

        {root && (
          <div className="absolute top-3 left-3 flex items-center gap-0.5 rounded-lg border border-border/60 bg-[#1a1a1a]/90 backdrop-blur p-0.5">
            {VIEWS.map(v => (
              <button
                key={v.id}
                onClick={() => setViewRequest(r => ({ view: v.id, n: (r?.n ?? 0) + 1 }))}
                title={`${v.label} view`}
                className={cn(
                  'px-2 py-1 rounded-md text-[11px] font-medium transition-colors',
                  viewRequest?.view === v.id ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground hover:bg-secondary',
                )}
              >
                {v.label}
              </button>
            ))}
          </div>
        )}

        {!root && !loading && !error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none px-8">
            <Box size={28} className="text-muted-foreground/50 mb-3" />
            <p className="text-sm text-muted-foreground">Nothing drawn yet</p>
            <p className="text-xs text-muted-foreground/70 mt-1">Try “a table with four legs and a vase on top”.</p>
          </div>
        )}

        {error && (
          <div className="absolute top-3 left-3 right-3 text-xs text-rose-400 bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
            {error}
          </div>
        )}

        {selectedSpec && (
          <div className="absolute bottom-3 left-3 max-w-[70%] rounded-lg border border-border/60 bg-[#1a1a1a]/95 backdrop-blur px-3 py-2 text-xs">
            <div className="flex items-center gap-2">
              <span className="font-mono font-semibold text-foreground">{selectedSpec.id}</span>
              <span className="text-muted-foreground">{selectedSpec.shape}</span>
              {selectedSpec.parent && <span className="text-muted-foreground/70">in {selectedSpec.parent}</span>}
            </div>
            {selectedSpec.params && (
              <div className="mt-1 text-[11px] text-muted-foreground font-mono break-words">{formatParams(selectedSpec.params)}</div>
            )}
            <div className="mt-1 text-[10px] text-muted-foreground/70">Refer to it by this id in chat, e.g. “make {selectedSpec.id} red”.</div>
          </div>
        )}
      </div>
    </div>
  )
}
